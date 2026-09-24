"""localforge as a Claude-Code-like harness: project tools confined to the
folder, local models writing files behind a diff + permission prompt, a
conversation that persists across messages, structured tool arguments over
the CLI transport, and session memory kept by a local model.

Reported: in a session it couldn't clone/read/write anything ("I don't have
a tool to clone/download a git repository") and forgot the previous message
("why do you not have context").
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import cli_transport, memory
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.orchestrator import Conversation
from localforge.tools import ActivityHooks, Dispatcher, _extract_file_content, build_tool_schemas
from localforge.workspace import Workspace, WorkspaceError


def _hw() -> HardwareProfile:
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


CATALOG = [
    ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1),
    ModelEntry(name="talker", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1),
]


class Local:
    def __init__(self, reply="```python\nprint('hello')\n```"):
        self.reply = reply
        self.prompts = []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        return {"type": "text", "content": self.reply, "tokens": 5}


def _yes(kind, title, detail):
    return True


def _no(kind, title, detail):
    return False


# --- workspace: confinement and the Claude-Code-style tools --------------------


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 1\n")
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x")
    return tmp_path


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "src/../../x"])
def test_paths_cannot_leave_the_project(project, path):
    with pytest.raises(WorkspaceError, match="outside the project"):
        Workspace(project).resolve(path)


def test_a_symlink_pointing_outside_is_refused(project, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere")
    (outside / "secret.txt").write_text("s")
    os.symlink(outside, project / "link")
    with pytest.raises(WorkspaceError, match="outside the project"):
        Workspace(project).read_file("link/secret.txt")


def test_read_list_search(project):
    ws = Workspace(project)
    assert "    1  def main():" in ws.read_file("src/app.py")
    listing = ws.list_files()
    assert "src/app.py" in listing and "README.md" in listing
    assert "node_modules" not in listing and ".git" not in listing
    assert ws.list_files(pattern="*.py") == "src/app.py"
    assert ws.search(r"def \w+") == "src/app.py:1: def main():"


def test_read_file_pages_long_files(project):
    (project / "big.txt").write_text("\n".join(f"l{i}" for i in range(1, 1001)))
    out = Workspace(project).read_file("big.txt")
    assert "(1000 lines)" in out and "pass offset=401" in out


def test_writes_need_approval_and_show_a_diff(project):
    seen = {}

    def approver(kind, title, detail):
        seen.update(kind=kind, title=title, detail=detail)
        return False

    out = Workspace(project, approver).edit_file("src/app.py", "return 1", "return 2")
    assert "declined" in out
    assert (project / "src" / "app.py").read_text().endswith("return 1\n")  # untouched
    assert seen["kind"] == "write" and seen["title"] == "Update src/app.py"
    assert "-    return 1" in seen["detail"] and "+    return 2" in seen["detail"]

    Workspace(project, _yes).edit_file("src/app.py", "return 1", "return 2")
    assert (project / "src" / "app.py").read_text().endswith("return 2\n")


def test_edit_requires_a_unique_match(project):
    (project / "d.py").write_text("x = 1\nx = 1\n")
    with pytest.raises(WorkspaceError, match="appears 2 times"):
        Workspace(project, _yes).edit_file("d.py", "x = 1", "x = 2")
    with pytest.raises(WorkspaceError, match="not found"):
        Workspace(project, _yes).edit_file("d.py", "y = 1", "x = 2")


def test_commands_need_approval_and_run_in_the_project(project):
    assert "declined" in Workspace(project, _no).run_command("echo hi > made.txt")
    assert not (project / "made.txt").exists()
    out = Workspace(project, _yes).run_command("pwd && echo done")
    assert out.startswith("exit code 0") and str(project.resolve()) in out and "done" in out


def test_no_approver_means_no_changes(project):
    assert "declined" in Workspace(project).write_file("new.txt", "x")
    assert not (project / "new.txt").exists()


def test_snapshot_describes_the_project(project):
    snap = Workspace(project).snapshot()
    assert "Git branch: main" in snap and "src/" in snap and "README.md" in snap
    assert "node_modules" not in snap


# --- local models write files ------------------------------------------------------


def test_delegate_with_path_writes_the_local_models_file_after_approval(project):
    local = Local("Here you go:\n```python\ndef main():\n    return 42\n```\nHope that helps!")
    d = Dispatcher(_hw(), catalog=CATALOG, workspace=Workspace(project, _yes))
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        out = d.dispatch("delegate_coding_task", {"instructions": "return 42", "path": "src/app.py"})

    assert (project / "src" / "app.py").read_text() == "def main():\n    return 42\n"
    assert out.startswith("Updated src/app.py")
    # the local model got the current file, since it sees nothing else
    assert "def main():\n    return 1" in local.prompts[0]
    assert "COMPLETE contents of the file `src/app.py`" in local.prompts[0]


def test_a_suspect_local_result_is_not_written(project):
    d = Dispatcher(_hw(), catalog=CATALOG, workspace=Workspace(project, _yes))
    with patch.dict("localforge.tools.BACKENDS", {"stub": Local("")}):
        out = d.dispatch("delegate_coding_task", {"instructions": "x" * 100, "path": "new.py"})
    assert out.startswith("[WARNING:") and "Nothing was written" in out
    assert not (project / "new.py").exists()


def test_extract_file_content_keeps_the_longest_fenced_block():
    assert _extract_file_content("a\n```\nshort\n```\n```py\nlong one\nhere\n```") == "long one\nhere\n"
    assert _extract_file_content("plain text") == "plain text\n"


def test_direct_tools_route_to_the_workspace_and_report(project):
    events = []
    hooks = ActivityHooks(on_tool=lambda t, s: events.append(("tool", t, s)), on_tool_result=lambda t, r: events.append(("result", t)))
    d = Dispatcher(_hw(), catalog=CATALOG, workspace=Workspace(project), hooks=hooks)
    assert "def main" in d.dispatch("read_file", {"path": "src/app.py"})
    assert events == [("tool", "read_file", "src/app.py"), ("result", "read_file")]
    assert "outside the project" in d.dispatch("read_file", {"path": "../x"})  # text, not an exception
    assert "missing the required argument 'pattern'" in d.dispatch("search", {})


def test_update_todos_goes_to_the_hook():
    shown = []
    d = Dispatcher(_hw(), catalog=CATALOG, hooks=ActivityHooks(on_todos=shown.append))
    out = d.dispatch("update_todos", {"todos": [{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}]})
    assert out == "Plan updated (1/2 done)."
    assert [t["content"] for t in shown[0]] == ["a", "b"]


def test_tool_schemas_carry_real_parameters():
    by_name = {s["function"]["name"]: s["function"]["parameters"] for s in build_tool_schemas(_hw(), CATALOG)}
    assert by_name["read_file"]["required"] == ["path"]
    assert set(by_name["delegate_coding_task"]["properties"]) == {"instructions", "path", "context_files"}
    assert {"read_file", "list_files", "search", "edit_file", "run_command", "update_todos"} <= set(by_name)
    assert "write_file" not in by_name  # the orchestrator can't write whole files itself


# --- structured arguments over the CLI transport -------------------------------


def _proc(result_obj):
    return MagicMock(returncode=0, stdout=json.dumps({"result": json.dumps(result_obj), "usage": {}}), stderr="")


@pytest.mark.parametrize(
    "call, expected",
    [
        ({"name": "read_file", "arguments": {"path": "a.py", "offset": 5}}, {"path": "a.py", "offset": 5}),
        ({"name": "read_file", "arguments": '{"path": "a.py"}'}, {"path": "a.py"}),
        ({"name": "delegate_coding_task", "instructions": "do it"}, {"instructions": "do it"}),
    ],
)
def test_cli_transport_parses_tool_arguments(call, expected):
    with patch.object(cli_transport, "available", return_value=True), patch.object(
        cli_transport.subprocess, "run", return_value=_proc({"tool_calls": [call]})
    ):
        resp = cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [])
    assert json.loads(resp.choices[0].message.tool_calls[0].function.arguments) == expected


def test_cli_failure_reports_the_envelopes_own_error():
    proc = MagicMock(returncode=1, stdout='{"subtype": "error_during_execution", "result": "Prompt is too long"}', stderr="")
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", return_value=proc):
        with pytest.raises(cli_transport.CLINotAvailableError, match="Prompt is too long"):
            cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [])


# --- the conversation persists across messages --------------------------------------


def _final(text):
    msg = MagicMock(tool_calls=None, content=text)
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    resp = MagicMock(choices=[MagicMock(message=msg)])
    resp.usage.prompt_tokens = resp.usage.completion_tokens = 1
    return resp


def test_second_message_sees_the_first(project):
    conv = Conversation()
    sent = []

    def fake_completion(**kw):
        sent.append([m["content"] for m in kw["messages"]])
        return _final(f"answer {len(sent)}")

    with (
        patch.object(orch, "completion", side_effect=fake_completion),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        orch.run("clone the repo", "gpt-5", hardware=_hw(), conversation=conv, workspace=Workspace(project))
        orch.run("why do you not have context", "gpt-5", hardware=_hw(), conversation=conv, workspace=Workspace(project))

    second = sent[1]
    assert "clone the repo" in second and "answer 1" in second and second[-1] == "why do you not have context"
    assert "Project folder:" in second[0]  # the system prompt describes the project
    assert sum(1 for c in second if c.startswith("You are the orchestrator")) == 1  # one system message


def test_session_reuses_one_conversation_per_folder(project, monkeypatch, tmp_path_factory):
    monkeypatch.setattr(cli_module.config, "CONFIG_DIR", tmp_path_factory.mktemp("cfg"))
    session = cli_module._SessionState()
    a = session.conversation_for(project)
    assert session.conversation_for(project) is a
    assert session.conversation_for(tmp_path_factory.mktemp("other")) is not a


# --- session memory kept by a local model ------------------------------------------


def _long_conversation(turns=3):
    conv = Conversation()
    conv.messages = [{"role": "system", "content": "sys"}]
    for i in range(turns):
        conv.messages += [{"role": "user", "content": f"task {i}"}, {"role": "assistant", "content": f"did {i}"}]
    return conv


def test_compaction_folds_old_turns_into_memory_and_saves_it(project, monkeypatch, tmp_path):
    monkeypatch.setattr(memory.config, "CONFIG_DIR", tmp_path / "cfg")
    local = Local("## Goal\nbuild the thing")
    d = Dispatcher(_hw(), catalog=CATALOG, workspace=Workspace(project))
    conv = _long_conversation()
    with patch.dict("localforge.memory.BACKENDS", {"stub": local}):
        assert memory.compact(conv, d)

    assert conv.memory == "## Goal\nbuild the thing"
    assert [m["content"] for m in conv.messages] == ["sys", "task 2", "did 2"]  # latest turn kept verbatim
    assert "task 0" in local.prompts[0] and "task 2" not in local.prompts[0]
    assert memory.load(project) == "## Goal\nbuild the thing"  # a new session here starts with it
    assert "Session memory" in conv.system_message()["content"]
    assert d.local_tokens_generated == 5


def test_compaction_without_a_local_model_still_caps_history(project, monkeypatch, tmp_path):
    monkeypatch.setattr(memory.config, "CONFIG_DIR", tmp_path / "cfg")
    d = Dispatcher(_hw(), catalog=[], workspace=Workspace(project))
    conv = _long_conversation()
    assert memory.compact(conv, d)
    assert [m["content"] for m in conv.messages] == ["sys", "task 2", "did 2"]
    assert conv.memory == ""


def test_run_compacts_automatically_when_history_is_large(project, monkeypatch):
    conv = _long_conversation()
    conv.messages[1]["content"] = "x" * (memory.COMPACT_AT_CHARS + 1)
    with (
        patch.object(orch.memory, "compact") as compact,
        patch.object(orch, "completion", return_value=_final("ok")),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        orch.run("next", "gpt-5", hardware=_hw(), conversation=conv)
    compact.assert_called_once()


def test_memory_file_lives_in_the_project(project):
    path = memory.memory_file(project)
    assert path == project.resolve() / ".localforge" / "session.md"
    assert memory.memory_dir(project) == path.parent / "memory"


# --- permission prompt ---------------------------------------------------------------


@pytest.fixture
def fresh_session(monkeypatch):
    monkeypatch.setattr(cli_module, "_session", cli_module._SessionState())
    # the app's root callback always pushes a theme before any command runs
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    yield cli_module._session
    cli_module.console.pop_theme()


def test_prompt_declines_without_a_terminal(fresh_session, monkeypatch):
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: False, raising=False)
    assert cli_module._LiveActivity("m").approve("write", "Create a.py", "+x") is False


def test_prompt_yes_no_always(fresh_session, monkeypatch):
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_module, "_drain_buffered_input", lambda: None)
    answers = iter(["maybe", "n", "y", "a"])
    prompts = []

    def fake_input(prompt=""):
        prompts.append(cli_module.console.render_str(prompt).plain)
        return next(answers)

    monkeypatch.setattr(cli_module.console, "input", fake_input)
    act = cli_module._LiveActivity("m")
    assert act.approve("write", "Create a.py", "+x") is False  # "maybe" re-asks, then n
    assert act.approve("command", "Run command", "ls") is True
    assert act.approve("command", "Run command", "ls") is True  # "a": always from now on
    assert act.approve("command", "Run command", "pwd") is True  # no more answers needed
    assert "command" in fresh_session.always_allow and "write" not in fresh_session.always_allow
    assert prompts[0] == "  Allow? (y)es / (n)o / (a)lways allow file changes this session: "


def test_auto_and_yes_approve_everything(fresh_session, monkeypatch):
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: False, raising=False)
    CliRunner().invoke(cli_module.app, ["auto", "on"])
    assert cli_module._LiveActivity("m").approve("write", "t", "+x") is True
    CliRunner().invoke(cli_module.app, ["auto", "off"])
    assert cli_module._LiveActivity("m").approve("write", "t", "+x") is False


def test_clear_starts_a_fresh_conversation(fresh_session, project, monkeypatch, tmp_path):
    monkeypatch.setattr(memory.config, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.chdir(project)
    memory.save(project, "old memory")
    conv = fresh_session.conversation_for(project.resolve())
    assert conv.memory == "old memory"
    conv.messages.append({"role": "user", "content": "x"})

    CliRunner().invoke(cli_module.app, ["clear"])
    assert fresh_session.conversation.messages == []
    assert memory.load(project) == "old memory"  # kept unless --forget
    CliRunner().invoke(cli_module.app, ["clear", "--forget"])
    assert memory.load(project) == ""


def test_run_answer_is_rendered_as_markdown(fresh_session, monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module.config, "CONFIG_FILE", tmp_path / "config.env")
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)
    from localforge.orchestrator import RunResult, RunStats

    with patch.object(cli_module, "run_orchestrator", return_value=RunResult("## Done\n- **one**", RunStats())):
        out = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"]).output
    assert "Done" in out and "## Done" not in out and "**one**" not in out


# --- session start: pick the orchestrator; session end: save memory -------------------


CHOICES = [
    ("ollama/qwen2.5:7b", "ollama/qwen2.5:7b  (local)", {"LOCALFORGE_AUTH_METHOD": "local", "LOCALFORGE_FRONTIER_PROVIDER": "local"}),
    ("claude-opus-5", "claude-opus-5  (your `claude` login)", {"LOCALFORGE_AUTH_METHOD": "cli_login", "LOCALFORGE_FRONTIER_PROVIDER": "anthropic"}),
]


@pytest.fixture
def picker(fresh_session, monkeypatch):
    """Drive the session-start picker with scripted answers; returns the
    prompts that were shown."""
    monkeypatch.setattr(cli_module, "_model_choices", lambda: CHOICES)
    monkeypatch.setattr(cli_module, "_drain_buffered_input", lambda: None)
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda p: True)
    prompts = []

    def run(answers):
        answers = iter(answers)

        def fake_input(prompt=""):
            prompts.append(cli_module.console.render_str(prompt).plain)
            return next(answers)

        monkeypatch.setattr(cli_module.console, "input", fake_input)
        return cli_module._choose_orchestrator_at_start()

    run.prompts = prompts
    return run


def test_resuming_offers_to_keep_the_last_orchestrator(picker, monkeypatch, capsys):
    """Reported: resuming showed the whole model list instead of asking
    whether to keep the current orchestrator."""
    monkeypatch.setenv("LOCALFORGE_FRONTIER_MODEL", "claude-opus-5")
    monkeypatch.setenv("LOCALFORGE_AUTH_METHOD", "cli_login")
    monkeypatch.setenv("LOCALFORGE_FRONTIER_PROVIDER", "anthropic")
    assert picker(["", "1"]) is True  # Enter alone picks nothing
    assert picker.prompts == ["Choose (1-2): ", "Choose (1-2): "]  # never the long list
    out = capsys.readouterr().out
    assert "Orchestrator from last time: claude-opus-5" in out and "Keep using it" in out
    assert "ollama/qwen2.5:7b" not in out
    assert not cli_module.config.CONFIG_FILE.exists()  # nothing changed


def test_choosing_a_different_one_shows_the_list(picker, monkeypatch, capsys):
    monkeypatch.setenv("LOCALFORGE_FRONTIER_MODEL", "claude-opus-5")
    assert picker(["2", "9", "1"]) is True  # out-of-range re-asks
    out = capsys.readouterr().out
    assert "Which model should orchestrate this session?" in out and "(last used)" in out
    saved = cli_module.config.CONFIG_FILE.read_text()
    assert "LOCALFORGE_FRONTIER_MODEL=ollama/qwen2.5:7b" in saved and "LOCALFORGE_AUTH_METHOD=local" in saved


def test_a_deleted_local_orchestrator_goes_straight_to_the_list(picker, monkeypatch, capsys):
    monkeypatch.setenv("LOCALFORGE_FRONTIER_MODEL", "ollama/gemma3:4b")  # no longer downloaded
    assert picker(["2"]) is True
    out = capsys.readouterr().out
    assert "ollama/gemma3:4b from last time isn't available here any more" in out
    assert "Keep using it" not in out
    assert "LOCALFORGE_FRONTIER_MODEL=claude-opus-5" in cli_module.config.CONFIG_FILE.read_text()


def test_first_ever_session_shows_the_list(picker, monkeypatch, capsys):
    monkeypatch.delenv("LOCALFORGE_FRONTIER_MODEL", raising=False)
    assert picker(["1"]) is True
    out = capsys.readouterr().out
    assert "Keep using it" not in out and "Which model should orchestrate" in out


def test_a_custom_cloud_model_with_its_login_counts_as_usable(picker, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_FRONTIER_MODEL", "claude-some-future-model")
    assert cli_module._current_is_usable("claude-some-future-model", CHOICES) is True
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda p: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli_module._current_is_usable("claude-some-future-model", CHOICES) is False


def test_backing_out_of_the_start_picker_ends_the_session(fresh_session, monkeypatch):
    monkeypatch.setattr(cli_module, "_model_choices", lambda: [("m", "m", {})])
    monkeypatch.setattr(cli_module, "_drain_buffered_input", lambda: None)

    def ctrl_d(prompt=""):
        raise EOFError

    monkeypatch.setattr(cli_module.console, "input", ctrl_d)
    assert cli_module._choose_orchestrator_at_start() is False


def test_repl_runs_the_start_picker_and_saves_memory_on_exit(monkeypatch):
    from localforge import repl

    events = []
    console = MagicMock()
    console.input.side_effect = ["/exit"]
    monkeypatch.setattr(repl.banner, "render", lambda c: None)
    repl.run_repl(MagicMock(), console, on_start=lambda: events.append("start") or True, on_exit=lambda: events.append("exit"))
    assert events == ["start", "exit"]


def test_exit_folds_the_session_into_memory(fresh_session, project, monkeypatch):
    conv = fresh_session.conversation_for(project)
    conv.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "build x"}, {"role": "assistant", "content": "built x"}]
    fresh_session.root = project
    local = Local("## Goal\nbuild x")
    monkeypatch.setattr(cli_module, "_memory_dispatcher", lambda hooks=None: Dispatcher(_hw(), catalog=CATALOG, installed={"talker"}, hooks=hooks))
    with patch.dict("localforge.memory.BACKENDS", {"stub": local}):
        cli_module._save_memory_at_exit()
    assert memory.load(project) == "## Goal\nbuild x"
    assert "build x" in local.prompts[0]


def test_memory_command_shows_the_note_and_its_keeper(fresh_session, project, monkeypatch):
    monkeypatch.chdir(project)
    memory.save(project, "## Goal\nship it")
    monkeypatch.setattr(cli_module, "_memory_dispatcher", lambda hooks=None: Dispatcher(_hw(), catalog=CATALOG, installed={"talker"}))
    monkeypatch.setitem(memory.BACKENDS, "stub", Local())
    out = " ".join(CliRunner().invoke(cli_module.app, ["memory"]).output.split())
    assert "ship it" in out and "kept by talker (local)" in out
    CliRunner().invoke(cli_module.app, ["memory", "clear"])
    assert memory.load(project) == ""


def test_memory_never_downloads_a_model_to_keep_itself():
    d = Dispatcher(_hw(), catalog=CATALOG, installed=set())
    assert memory.keeper(d) is None
