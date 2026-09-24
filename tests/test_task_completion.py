"""A task ends when it's done, not when the orchestrator stops talking
(reported: "if there's an issue the overall process stops, so I need some
check/loop to ensure the task is completed; hopefully AGENTS.md captures it").

Before a final answer ends the task, localforge checks: is every plan item
done, were changed files checked since (the project's tests, per AGENTS.md's
"Definition of done"), and did the answer give up on something? What's
missing goes back to the orchestrator, at most MAX_COMPLETION_CHECKS times.
"""

from unittest.mock import MagicMock, patch

import pytest

import localforge.orchestrator as orch
from localforge import brief, memory
from localforge.hardware import HardwareProfile


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def _call(name, args):
    import json

    msg = MagicMock()
    msg.tool_calls = [MagicMock(id=f"c-{name}", function=MagicMock(arguments=json.dumps(args)))]
    msg.tool_calls[0].function.name = name
    msg.model_dump.return_value = {"role": "assistant", "content": None}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


def _final(text="All done."):
    msg = MagicMock(tool_calls=None, content=text)
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


class Tools:
    """Answers tool calls the way the real dispatcher words its results."""

    def __init__(self):
        self.catalog, self.local_tokens_generated, self.calls = [], 0, []

    def dispatch(self, name, args, on_delegate=None):
        self.calls.append(name)
        if name == "delegate_coding_task":
            return f"Created {args['path']} (+10 -0, 10 lines)."
        if name == "run_command":
            if args.get("declined"):
                return f"The user declined to run: {args['command']}"
            return "exit code 0\n3 passed"
        if name == "read_file":
            return "1: x = 1"
        if name == "update_todos":
            return "Plan updated."
        return "ok"


def _run(replies, brief_text=""):
    sent = []
    replies = iter(replies)

    def completion(**kw):
        sent.append(kw["messages"][-1])
        return next(replies)

    conv = orch.Conversation(brief=brief_text)
    tools = Tools()
    with (
        patch.object(orch, "completion", side_effect=completion),
        patch.object(orch, "Dispatcher", lambda *a, **kw: tools),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        result = orch.run("build it", "gpt-5", hardware=_hw(), conversation=conv)
    checks = [m["content"] for m in sent if str(m.get("content", "")).startswith(orch.COMPLETION_PREFIX)]
    return result, checks, tools


AGENTS = "# Project brief\n\n## Definition of done\nRun `uv run pytest -q` and `uv run ruff check .`\n"


def test_changed_but_unchecked_work_is_sent_back_with_the_projects_own_checks():
    result, checks, tools = _run(
        [
            _call("delegate_coding_task", {"instructions": "x", "path": "app.py"}),
            _final(),  # stops without checking
            _call("run_command", {"command": "uv run pytest -q"}),
            _final("Done; tests pass."),
        ],
        AGENTS,
    )
    assert len(checks) == 1 and "app.py" in checks[0] and "uv run pytest -q" in checks[0]
    assert tools.calls[-1] == "run_command" and result.answer == "Done; tests pass."


def test_unfinished_plan_items_are_sent_back():
    todos = [{"content": "write the API", "status": "completed"}, {"content": "add tests", "status": "pending"}]
    done = [{**t, "status": "completed"} for t in todos]
    result, checks, _ = _run(
        [
            _call("update_todos", {"todos": todos}),
            _final(),
            _call("update_todos", {"todos": done}),
            _final("Everything's done."),
        ]
    )
    assert len(checks) == 1 and "- add tests" in checks[0] and "write the API" not in checks[0]
    assert result.answer == "Everything's done."


def test_reading_back_a_changed_file_counts_as_checking_it():
    _, checks, _ = _run(
        [
            _call("delegate_coding_task", {"instructions": "x", "path": "README.md"}),
            _call("read_file", {"path": "README.md"}),
            _final(),
        ]
    )
    assert checks == []


def test_a_declined_test_run_is_not_asked_again():
    _, checks, _ = _run(
        [
            _call("delegate_coding_task", {"instructions": "x", "path": "app.py"}),
            _call("run_command", {"command": "pytest", "declined": True}),
            _final("Wrote app.py; you declined running the tests."),
        ]
    )
    assert checks == []


def test_giving_up_is_questioned_once_then_accepted():
    result, checks, _ = _run([_final("I couldn't connect to the database."), _final("It needs DB credentials: set DATABASE_URL.")])
    assert len(checks) == 1 and "another way" in checks[0]
    assert result.answer.startswith("It needs DB credentials")


def test_a_chat_reply_is_not_checked():
    result, checks, _ = _run([_final("Hi! What would you like to build?")])
    assert checks == [] and result.answer.startswith("Hi!")


def test_the_check_never_loops_forever():
    stubborn = [_call("delegate_coding_task", {"instructions": "x", "path": "app.py"})] + [_final("Done.")] * 10
    result, checks, _ = _run(stubborn)
    assert len(checks) == orch.MAX_COMPLETION_CHECKS and result.answer == "Done."


def test_completion_checks_are_not_mistaken_for_user_turns():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "build it"}, {"role": "user", "content": orch.COMPLETION_PREFIX + "x"}]
    assert memory._turn_starts(msgs) == [1]


# --- AGENTS.md says what "done" means ------------------------------------------------------


def test_init_asks_for_a_definition_of_done():
    assert "## Definition of done" in brief.PROMPT
    assert "definition of done" in brief.GROUNDING_SECTIONS  # local models see it too


def test_an_older_brief_falls_back_to_running_and_testing():
    old = "# Project brief\n\n## Running and testing\nmake test\n\n## Conventions and decisions\nx\n"
    assert brief.section(old, ("definition of done", "running and testing")) == "make test"


# --- a hiccup in the orchestrator is tried again, patiently ---------------------------------


def test_a_transient_orchestrator_failure_gets_three_tries_with_growing_waits(monkeypatch):
    from localforge import cli_transport

    waits = []
    monkeypatch.setattr(orch.time, "sleep", waits.append)
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise cli_transport.CLINotAvailableError("claude exited 1: Stream closed unexpectedly")
        return cli_transport.CLIResponse(choices=[cli_transport._Choice(message=cli_transport.message_from_reply('{"final_answer": "ok"}'))])

    with patch.object(orch.cli_transport, "complete", side_effect=flaky), patch.object(orch, "_installed_models", return_value=None):
        result = orch.run("hi", "claude-opus-5", hardware=_hw(), cli_provider="anthropic")
    assert result.answer == "ok" and calls["n"] == 3
    assert waits == [orch.RETRY_DELAY_SECONDS, orch.RETRY_DELAY_SECONDS * 3]


def test_it_still_stops_after_three_failed_tries(monkeypatch):
    from localforge import cli_transport

    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    with patch.object(
        orch.cli_transport, "complete", side_effect=cli_transport.CLINotAvailableError("claude exited 1: Stream closed")
    ) as call, patch.object(orch, "_installed_models", return_value=None):
        with pytest.raises(cli_transport.CLINotAvailableError):
            orch.run("hi", "claude-opus-5", hardware=_hw(), cli_provider="anthropic")
    assert call.call_count == orch.MAX_FRONTIER_ATTEMPTS
