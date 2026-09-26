"""Feedback between the frontier model and the local models (reported: "is
there no feedback between the frontier model and the local models?").

Four gaps closed:
1. A check failure (tests/lint/build) is carried into the next file-writing
   delegation automatically, instead of the orchestrator having to paste it.
2. Remembered preferences (feedback/user/project facts) reach every local
   delegation, not just the orchestrator's own instructions.
3. A local model's corrections (syntax fixes, still-bad results) are logged,
   and a repeated one becomes a lasting memory the memory keeper writes.
4. A task that stops before it's done (Ctrl+C, the step ceiling) is saved,
   and the next session offers to continue it.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from localforge import memory
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import Dispatcher
from localforge.workspace import Workspace

CODER = [ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)]


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[], memory_bandwidth_gbps=400)


class Local:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"type": "text", "content": reply, "tokens": 5}


@pytest.fixture
def project(tmp_path):
    return tmp_path


def _dispatcher(project, approver=lambda *a: True):
    return Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project, approver))


# --- 1. a check failure is carried into the next delegation automatically ---------------


def test_a_failed_check_is_handed_to_the_next_file_write(project):
    d = _dispatcher(project)
    with patch.dict("localforge.tools.BACKENDS", {"stub": Local("some fine output here")}):
        d.dispatch("run_command", {"command": "uv run pytest -q"})  # ws.run_command not stubbed: exit code varies
    # force a known failing result directly through the tracked hook
    d._last_check_failure = "exit code 1\nFAILED tests/test_x.py::test_thing"
    local = Local("```python\nx = 1\n```\nNOTES: none")
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "fix it", "path": "x.py"})
    assert "checks failed after the last change" in local.prompts[0]
    assert "FAILED tests/test_x.py::test_thing" in local.prompts[0]


def test_a_passing_check_clears_the_carried_failure(project):
    d = _dispatcher(project)
    d._last_check_failure = "exit code 1\nold failure"
    with patch.object(Workspace, "run_command", lambda self, command, timeout=300: "exit code 0\nall good"):
        d.dispatch("run_command", {"command": "uv run pytest -q"})
    assert d._last_check_failure is None


def test_the_carried_failure_is_only_shown_once(project):
    d = _dispatcher(project)
    d._last_check_failure = "exit code 1\nboom"
    local = Local("```python\nx = 1\n```\nNOTES: none")
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "fix it", "path": "a.py"})
        d.dispatch("delegate_coding_task", {"instructions": "another change", "path": "b.py"})
    assert "checks failed" in local.prompts[0] and "checks failed" not in local.prompts[1]


def test_a_declined_command_does_not_count_as_a_check_failure(project):
    d = _dispatcher(project, approver=lambda *a: False)  # declines everything
    d.dispatch("run_command", {"command": "uv run pytest -q"})
    assert d._last_check_failure is None


# --- 2. remembered preferences reach every local delegation ------------------------------


def test_remembered_preferences_reach_local_delegations(project):
    memory.remember(project, "no-print", "Never use print(); always use logging.", type="feedback")
    memory.remember(project, "just-a-link", "See https://example.com/api-docs", type="reference")
    local = Local("some fine output here that is long enough")
    d = _dispatcher(project)
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "add a handler"})
    assert "Never use print()" in local.prompts[0]
    assert "example.com/api-docs" not in local.prompts[0]  # a reference isn't a coding rule


def test_no_facts_means_no_extra_section(project):
    local = Local("some fine output here that is long enough")
    d = _dispatcher(project)
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "add a handler"})
    assert "Remembered preferences" not in local.prompts[0]


# --- 3. corrections are logged, and repeats become lasting rules -------------------------


def test_a_syntax_fix_is_logged_as_a_correction(project):
    d = _dispatcher(project)
    local = Local("```python\ndef f(:\n```", "```python\ndef f():\n    return 1\n```\nNOTES: none")
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "write f", "path": "f.py"})
    corrections = memory.recent_corrections(project)
    assert corrections and corrections[0]["kind"] == "syntax" and corrections[0]["path"] == "f.py"


def test_a_repeated_correction_becomes_a_lasting_rule(project):
    for _ in range(3):
        memory.note_correction(project, "syntax", "app.py", "missing colon")
    dispatcher = _dispatcher(project)

    class Keeper:
        def ensure_available(self, name, on_progress=None):
            pass

        def generate(self, name, prompt, on_token=None, **kw):
            assert "missing colon" in prompt  # the log reached the extraction prompt
            reply = json.dumps({"memories": [{"name": "always-close-blocks", "type": "feedback", "content": "Always close code blocks."}], "forget": []})
            return {"type": "text", "content": reply, "tokens": 10}

    conv = MagicMock(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}])
    with patch.dict("localforge.tools.BACKENDS", {"stub": Keeper()}), patch.dict("localforge.memory.BACKENDS", {"stub": Keeper()}):
        changed = memory.extract_facts(conv, dispatcher, project)
    assert changed == 1
    facts = [f["name"] for f in memory.list_facts(project)]
    assert "always-close-blocks" in facts


# --- 4. unfinished work is saved and offered to resume -----------------------------------


def test_a_ctrl_c_saves_open_work_with_the_plan_and_unchecked_files(project, monkeypatch):
    import localforge.orchestrator as orch

    def raises_ctrl_c(**kw):
        raise KeyboardInterrupt

    conv = orch.Conversation()
    with (
        patch.object(orch, "completion", side_effect=raises_ctrl_c),
        patch.object(orch, "Dispatcher", lambda *a, **kw: _StubDispatcher()),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
    ):
        with pytest.raises(orch.TaskCancelled):
            orch.run("build the API", "gpt-5", hardware=_hw(), conversation=conv, workspace=Workspace(project))
    items = memory.open_work(project)
    assert items and items[-1]["task"] == "build the API" and "stopped" in items[-1]["why"]


class _StubDispatcher:
    def __init__(self):
        self.catalog, self.local_tokens_generated = [], 0


def test_a_finished_task_clears_any_saved_open_work(project):
    import localforge.orchestrator as orch

    memory.save_open_work(project, "build the API", "the user stopped it", ["finish tests"], [])

    def final(**kw):
        msg = MagicMock(tool_calls=None, content="All done.")
        msg.model_dump.return_value = {"role": "assistant", "content": "All done."}
        return MagicMock(choices=[MagicMock(message=msg)], usage=None)

    with (
        patch.object(orch, "completion", side_effect=final),
        patch.object(orch, "Dispatcher", lambda *a, **kw: _StubDispatcher()),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        orch.run("build the API", "gpt-5", hardware=_hw(), workspace=Workspace(project))
    assert memory.open_work(project) == []


def test_describe_open_work_reads_plainly():
    entry = {"task": "build the API", "why": "the user stopped it", "when": "2026-09-23 10:00", "open_items": ["add tests"], "unchecked": ["app.py"]}
    text = memory.describe_open_work(entry)
    assert "build the API" in text and "the user stopped it" in text and "add tests" in text and "app.py" in text


def test_session_start_offers_to_continue(project, monkeypatch):
    import localforge.cli as cli_module

    memory.save_open_work(project, "build the API", "the user stopped it", ["add tests"], [])
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 1)
    ran = []
    monkeypatch.setattr(cli_module, "app", lambda argv, standalone_mode=True: ran.append(argv))
    cli_module._offer_open_work_at_start(project)
    assert len(ran) == 1 and ran[0][0] == "run"
    assert "build the API" in ran[0][1] and "add tests" in ran[0][1]


def test_discarding_removes_it(project, monkeypatch):
    import localforge.cli as cli_module

    memory.save_open_work(project, "build the API", "stopped", [], [])
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 3)
    cli_module._offer_open_work_at_start(project)
    assert memory.open_work(project) == []


def test_no_open_work_means_no_question(project, monkeypatch):
    import localforge.cli as cli_module

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda *a: pytest.fail("asked"))
    cli_module._offer_open_work_at_start(project)


# --- /tasks: SLASH_HELP listed it, but no command existed (typing it errored:
# "No such command 'tasks'") -- unlike /queue (tasks waiting behind the one
# running right now, in *this* session), /tasks is unfinished work left from
# *earlier*, already-ended tasks, and only the single most recent one was ever
# surfaced, automatically, at session start. --------------------------------------------


def test_tasks_lists_every_saved_entry_not_just_the_latest(project, monkeypatch, capsys):
    import localforge.cli as cli_module

    memory.save_open_work(project, "build the API", "the user stopped it", ["add tests"], [])
    memory.save_open_work(project, "add auth", "hit the round ceiling", ["wire up JWT"], ["auth.py"])
    monkeypatch.setattr(cli_module, "_session", type("S", (), {"root": project, "runner": None})())
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: False)  # just list, no picker
    cli_module.tasks_command(action=None)
    out = capsys.readouterr().out
    assert "build the API" in out and "add auth" in out
    assert "1." in out and "2." in out


def test_no_unfinished_tasks_says_so(project, monkeypatch, capsys):
    import localforge.cli as cli_module

    monkeypatch.setattr(cli_module, "_session", type("S", (), {"root": project, "runner": None})())
    cli_module.tasks_command(action=None)
    assert "No unfinished work from earlier tasks." in capsys.readouterr().out


def test_tasks_clear_discards_all_of_them(project, monkeypatch, capsys):
    import localforge.cli as cli_module

    memory.save_open_work(project, "build the API", "stopped", [], [])
    memory.save_open_work(project, "add auth", "stopped", [], [])
    monkeypatch.setattr(cli_module, "_session", type("S", (), {"root": project, "runner": None})())
    cli_module.tasks_command(action="clear")
    assert memory.open_work(project) == []
    assert "Discarded 2" in capsys.readouterr().out


def test_choosing_one_from_the_list_can_continue_it(project, monkeypatch):
    import localforge.cli as cli_module

    memory.save_open_work(project, "build the API", "stopped", ["finish tests"], [])
    memory.save_open_work(project, "add auth", "stopped", ["wire up JWT"], [])
    monkeypatch.setattr(cli_module, "_session", type("S", (), {"root": project, "runner": None})())
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module.console, "input", lambda prompt="": "2")  # the second entry
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 1)  # continue it now
    ran = []
    monkeypatch.setattr(cli_module, "app", lambda argv, standalone_mode=True: ran.append(argv))
    cli_module.tasks_command(action=None)
    assert len(ran) == 1 and ran[0][0] == "run"
    assert "add auth" in ran[0][1] and "wire up JWT" in ran[0][1]


def test_continuing_while_a_task_is_already_running_queues_instead_of_running_directly(project, monkeypatch):
    """Reported (elsewhere, same session): a command reading input or
    running a task synchronously while the background worker is already
    busy can corrupt things. Resuming open work must go through the same
    queue a normally-typed task would, not call app(["run", ...]) directly
    on top of whatever the worker thread is doing."""
    import localforge.cli as cli_module

    memory.save_open_work(project, "build the API", "stopped", [], [])
    runner = type("R", (), {"busy": True, "submit": lambda self, task: 1})()
    monkeypatch.setattr(cli_module, "_session", type("S", (), {"root": project, "runner": runner})())
    app_called = []
    monkeypatch.setattr(cli_module, "app", lambda *a, **k: app_called.append(a))
    cli_module._resume_open_work(memory.open_work(project)[0])
    assert not app_called  # never runs synchronously while the worker is busy
