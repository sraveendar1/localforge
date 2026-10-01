"""After every task there is a summary of what was done and what went wrong
(reported: "after every work, there should be a summary of activities
performed, and if there is an error that should also be printed out").
"""
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import task_summary as ts
from localforge import trust
from localforge.hardware import HardwareProfile
from localforge.serve import StdioServer
from localforge.theme import get_theme

runner = CliRunner()


# --- the summary itself ---------------------------------------------------------


def _log():
    log = ts.TaskLog()
    log.record("read_file", "src/app.py", "ok", "1: import os")
    log.record("search", "'TODO'", "ok", "src/app.py:3: # TODO")
    log.record("delegate_coding_task", "src/app.py", "ok", "Updated src/app.py (+12 -3)", model="qwen2.5-coder:7b")
    log.record("run_command", "pytest -q", "ok", "exit code 1\n1 failed")
    log.record("edit_file", "src/app.py", "ok", "Updated src/app.py (+1 -1)")
    log.record("run_command", "pytest -q", "ok", "exit code 0\n3 passed")
    log.record("delete_path", "old.py", "declined", "The user declined deleting old.py; nothing was removed.")
    return log


def test_summary_lists_files_delegations_commands_and_declines():
    s = ts.build(_log(), "completed")
    assert [f["path"] for f in s["files"]] == ["src/app.py", "src/app.py"]
    assert s["files"][0]["by"] == "qwen2.5-coder:7b"
    assert s["delegations"][0]["model"] == "qwen2.5-coder:7b"
    assert [c["ok"] for c in s["commands"]] == [False, True]
    assert s["reads"] == 2
    assert s["declined"][0]["what"] == "old.py"
    assert s["steps"] == 7


def test_a_failing_command_is_a_problem_and_marked_recovered_when_a_later_run_passes():
    s = ts.build(_log(), "completed")
    assert len(s["problems"]) == 1
    p = s["problems"][0]
    assert "pytest -q" in p["what"] and p["recovered"] is True


def test_a_failure_nothing_recovered_from_stays_unrecovered_and_the_error_is_kept():
    log = ts.TaskLog()
    log.record("delegate_coding_task", "x.py", "failed", "delegate_coding_task failed: ollama: read timed out")
    s = ts.build(log, "failed", "Something broke")
    assert s["problems"][0]["recovered"] is False
    text = "\n".join(ts.render_lines(s))
    assert "Failed" in text and "read timed out" in text and "Error: Something broke" in text


def test_notes_about_handled_trouble_show_up_as_recovered_problems():
    log = ts.TaskLog()
    log.note("an orchestrator call failed (timeout) and was retried")
    log.record("read_file", "a.py", "ok", "1: x")
    text = "\n".join(ts.render_lines(ts.build(log, "completed")))
    assert "retried" in text and "(recovered)" in text


def test_rendering_reads_like_a_report():
    text = "\n".join(ts.render_lines(ts.build(_log(), "completed")))
    assert "Done" in text and "7 steps" in text
    assert "updated src/app.py  (written by qwen2.5-coder:7b)" in text
    assert "Delegated to: qwen2.5-coder:7b ×1" in text
    assert "✗ pytest -q" in text and "✓ pytest -q" in text
    assert "You declined:" in text and "old.py" in text


def test_a_plain_answer_with_no_steps_is_empty_but_an_error_never_is():
    assert ts.is_empty(ts.build(ts.TaskLog(), "completed"))
    assert not ts.is_empty(ts.build(ts.TaskLog(), "failed", "boom"))
    assert not ts.is_empty(ts.build(None, "failed", "boom"))


def test_long_lists_are_clipped_and_counted():
    log = ts.TaskLog()
    for i in range(30):
        log.record("edit_file", f"f{i}.py", "ok", f"Updated f{i}.py (+1 -1)")
    text = "\n".join(ts.render_lines(ts.build(log, "completed")))
    assert "... and 18 more" in text


# --- the orchestrator fills the log ---------------------------------------------


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def _tool_reply(name, args):
    msg = MagicMock()
    msg.tool_calls = [MagicMock(id="c", function=MagicMock(arguments=json.dumps(args)))]
    msg.tool_calls[0].function.name = name
    msg.model_dump.return_value = {"role": "assistant", "content": None}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


def _final(text="done"):
    msg = MagicMock(tool_calls=None, content=text)
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


class ScriptedDispatcher:
    def __init__(self, results):
        self.results = list(results)
        self.catalog, self.local_tokens_generated = [], 0
        self._last_delegate_model = "qwen2.5-coder:7b"

    def dispatch(self, name, args, on_delegate=None):
        out = self.results.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _run(replies, dispatcher, side=None):
    conv = orch.Conversation()
    with (
        patch.object(orch, "completion", side_effect=side or (lambda **kw: next(replies))),
        patch.object(orch, "Dispatcher", lambda *a, **kw: dispatcher),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        return orch.run("build it", "gpt-5", hardware=_hw(), conversation=conv)


def test_run_logs_each_step_with_its_outcome_and_the_writing_model():
    replies = iter([
        _tool_reply("read_file", {"path": "a.py"}),
        _tool_reply("delegate_coding_task", {"instructions": "x", "path": "a.py"}),
        _tool_reply("run_command", {"command": "pytest -q"}),
        _final(),
    ])
    d = ScriptedDispatcher(["1: x", "Updated a.py (+1 -0)", "exit code 0\nok"])
    result = _run(replies, d)
    steps = result.stats.log.steps
    assert [(s.tool, s.outcome) for s in steps] == [("read_file", "ok"), ("delegate_coding_task", "ok"), ("run_command", "ok")]
    assert steps[1].model == "qwen2.5-coder:7b" and steps[1].target == "a.py"
    assert steps[2].target == "pytest -q"


def test_a_failed_then_retried_step_is_logged_as_a_recovered_problem():
    call = ("delegate_coding_task", {"instructions": "x", "path": "a.py"})
    replies = iter([_tool_reply(*call), _tool_reply(*call)] + [_final()] * 4)  # the completion check sends it back once
    d = ScriptedDispatcher([RuntimeError("ollama: read timed out"), "Created a.py (+3 -0)"])
    summary = ts.build(_run(replies, d).stats.log, "completed")
    assert summary["problems"] and summary["problems"][0]["recovered"] is True
    assert "read timed out" in summary["problems"][0]["error"]
    assert summary["files"][0]["path"] == "a.py"


def test_bookkeeping_tools_are_not_listed():
    replies = iter([_tool_reply("update_todos", {"todos": []}), _final()])
    result = _run(replies, ScriptedDispatcher(["ok"]))
    assert result.stats.log.steps == []


def test_a_transient_orchestrator_failure_that_was_retried_is_noted():
    calls = {"n": 0}

    class APIConnectionError(Exception):  # the type name is what _is_transient() recognises
        pass

    def side(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIConnectionError("connection reset")
        return _final()

    with patch.object(orch.time, "sleep"):
        result = _run(None, ScriptedDispatcher([]), side=side)
    assert any("retried" in n for n in result.stats.log.notes)


def test_an_unexpected_error_still_carries_what_the_task_did():
    replies = iter([_tool_reply("read_file", {"path": "a.py"})])

    def side(**kw):
        try:
            return next(replies)
        except StopIteration:
            raise ValueError("bad key") from None

    with pytest.raises(ValueError) as info:
        _run(None, ScriptedDispatcher(["1: x"]), side=side)
    stats = info.value.stats
    assert [s.tool for s in stats.log.steps] == ["read_file"]


def test_the_log_does_not_leak_into_the_usage_dict():
    import dataclasses

    assert "log" not in dataclasses.asdict(orch.RunStats())


# --- the terminal prints it -----------------------------------------------------


def _capture(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(cli_module, "console", Console(file=buf, width=100, force_terminal=False, theme=get_theme("matrix")))
    return buf


def test_terminal_prints_a_summary_panel_after_a_task(monkeypatch):
    buf = _capture(monkeypatch)
    stats = orch.RunStats()
    stats.log.record("delegate_coding_task", "a.py", "ok", "Created a.py (+3 -0)", model="m1")
    cli_module._print_task_summary(stats, "completed")
    out = buf.getvalue()
    assert "Summary" in out and "created a.py" in out and "m1" in out


def test_terminal_prints_nothing_for_a_plain_answer(monkeypatch):
    buf = _capture(monkeypatch)
    cli_module._print_task_summary(orch.RunStats(), "completed")
    assert buf.getvalue() == ""


def test_terminal_always_prints_an_error_even_with_no_steps(monkeypatch):
    buf = _capture(monkeypatch)
    cli_module._print_task_summary(None, "failed", "AuthenticationError: bad key")
    out = buf.getvalue()
    assert "Failed" in out and "AuthenticationError: bad key" in out


def test_run_prints_the_summary_on_success_and_on_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    trust.trust(tmp_path)
    monkeypatch.setattr(cli_module, "detect_hardware", _hw, raising=False)
    stats = orch.RunStats()
    stats.log.record("run_command", "pytest -q", "ok", "exit code 0\nok")
    monkeypatch.setattr(cli_module, "run_orchestrator", lambda *a, **k: orch.RunResult(answer="all done", stats=stats))
    ok = runner.invoke(cli_module.app, ["run", "go", "--model", "gpt-5", "--yes"])
    assert ok.exit_code == 0 and "Summary" in ok.output and "pytest -q" in ok.output

    def boom(*a, **k):
        err = ValueError("bad key")
        err.stats = stats
        raise err

    monkeypatch.setattr(cli_module, "run_orchestrator", boom)
    bad = runner.invoke(cli_module.app, ["run", "go", "--model", "gpt-5", "--yes"])
    assert bad.exit_code == 1
    assert "Error: bad key" in bad.output and "Summary" in bad.output and "Failed" in bad.output and "pytest -q" in bad.output


# --- the desktop app gets it ----------------------------------------------------


def _server(tmp_path, run_fn):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    trust.trust(tmp_path)
    return StdioServer(tmp_path, "m", out=out, run_fn=run_fn, scratch_root=scratch, conversation=orch.Conversation()), out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


def test_desktop_gets_a_task_summary_event_after_a_run(tmp_path):
    stats = orch.RunStats()
    stats.log.record("delegate_docs_task", "README.md", "ok", "Created README.md (+9 -0)", model="llama3.1:8b")
    server, out = _server(tmp_path, lambda *a, **k: orch.RunResult(answer="ok", stats=stats))
    server._run_turn("go")
    ev = _events(out, "task_summary")[-1]["summary"]
    assert ev["outcome"] == "completed" and ev["files"][0]["path"] == "README.md"


def test_desktop_sends_no_summary_for_a_plain_answer(tmp_path):
    server, out = _server(tmp_path, lambda *a, **k: orch.RunResult(answer="hi", stats=orch.RunStats()))
    server._run_turn("hello")
    assert _events(out, "task_summary") == []


def test_desktop_sends_the_error_in_the_summary_too(tmp_path):
    stats = orch.RunStats()
    stats.log.record("read_file", "a.py", "ok", "1: x")

    def boom(*a, **k):
        err = RuntimeError("model exploded")
        err.stats = stats
        raise err

    server, out = _server(tmp_path, boom)
    server._run_turn("go")
    assert _events(out, "error")
    s = _events(out, "task_summary")[-1]["summary"]
    assert s["outcome"] == "failed" and "model exploded" in s["error"] and s["steps"] == 1


# --- plain English ----------------------------------------------------------------------------

def test_the_judge_warning_from_the_screenshot_reads_as_a_sentence():
    raw = ("[WARNING: the judge model flagged this result (The response is a request for more information instead of "
           "fulfilling the task of creating the new function; it does not add any code.) -- verify before use, or "
           "delegate again with clearer/simpler instructions]")
    text = ts.explain_error(raw, "delegate_coding_task")
    assert "[WARNING" not in text and "--" not in text and "verify before use" not in text
    assert "coding model" in text and "request for more information" in text and "ask again" in text


def test_known_internal_messages_are_explained_and_unknown_ones_are_kept():
    assert "syntax error" in ts.explain_error("delegate_coding_task failed: the local model's a.py doesn't parse, even after one retry", "delegate_coding_task")
    assert "ran out of room" in ts.explain_error("[WARNING: the local coding model's output hit its 4096-token limit and was cut off ...]", "delegate_coding_task")
    assert "low on memory" in ts.explain_error("... so memory is short on this machine right now ...")
    assert ts.explain_error("AuthenticationError: bad key") == "AuthenticationError: bad key"  # unknown: nothing lost


def test_a_usage_limit_keeps_the_reset_time_in_the_sentence():
    text = ts.explain_error("You've hit your session limit · resets 11:40am (America/Chicago)")
    assert "usage limit" in text and "11:40am" in text


def test_friendly_names_for_steps():
    assert ts.friendly_step("delegate_coding_task", "dia.py") == "The coding model (dia.py)"
    assert ts.friendly_step("delegate_docs_task") == "The writing model"
    assert ts.friendly_step("run_command", "npm test") == "Running a command"


def test_problems_in_the_summary_carry_a_plain_sentence_and_keep_the_raw_text():
    log = ts.TaskLog()
    log.record("delegate_coding_task", "dia.py", "failed", "[WARNING: the judge model flagged this result (it only asks a question) -- verify before use, or delegate again]")
    s = ts.build(log, "completed")
    p = s["problems"][0]
    assert p["plain"].startswith("The coding model (dia.py):") and "[WARNING" not in p["plain"]
    assert "[WARNING" in p["error"]
    assert "[WARNING" not in "\n".join(ts.render_lines(s))
