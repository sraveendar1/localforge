"""A per-session scratchpad and a Claude-Code-style memory store.

Asked for: somewhere for temporary work (pseudo-code, drafts) that doesn't
pollute the project or git, and memory like Claude Code's -- a per-project
folder with a MEMORY.md index and one fact per file, kept current by a
local model.
"""

import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import memory, scratchpad
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.orchestrator import Conversation, RunResult, RunStats
from localforge.tools import Dispatcher, build_tool_schemas
from localforge.workspace import Workspace, WorkspaceError


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


CATALOG = [ModelEntry(name="keeper", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)]


class Local:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        return {"type": "text", "content": self.reply, "tokens": 4}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print(1)\n")
    return tmp_path


@pytest.fixture
def pad(project):
    sp = scratchpad.Scratchpad(project)
    sp.ensure()
    yield sp
    sp.remove()


def _never(*a):
    raise AssertionError("no approval should be asked for scratchpad work")


# --- scratchpad ---------------------------------------------------------------------------


def test_scratchpad_is_outside_the_project_private_and_per_session(project, pad):
    other = scratchpad.Scratchpad(project)
    other.ensure()
    try:
        assert project not in pad.root.parents
        assert pad.root != other.root  # a new session gets its own
        assert oct(os.stat(pad.session_dir).st_mode)[-3:] == "700"
    finally:
        other.remove()


def test_scratch_work_needs_no_approval_and_never_touches_the_project(project, pad):
    ws = Workspace(project, _never, scratch=pad.root)
    assert ws.write_file("scratchpad/plan.md", "1. do it\n").startswith("Created scratchpad/plan.md")
    ws.make_dir("scratchpad/tries")
    ws.edit_file("scratchpad/plan.md", "do it", "do it well")
    ws.move_path("scratchpad/plan.md", "scratchpad/tries/plan.md")
    assert "do it well" in ws.read_file("scratchpad/tries/plan.md")
    assert ws.list_files("scratchpad") == "scratchpad/tries/plan.md"
    ws.delete_path("scratchpad/tries", recursive=True)
    assert sorted(p.name for p in project.rglob("*")) == ["app.py", "src"]  # project untouched


def test_scratchpad_paths_cannot_escape(project, pad):
    ws = Workspace(project, _never, scratch=pad.root)
    with pytest.raises(WorkspaceError, match="outside the scratchpad"):
        ws.resolve("scratchpad/../../etc")
    with pytest.raises(WorkspaceError, match="not the scratchpad itself"):
        ws.delete_path("scratchpad")


def test_promoting_a_draft_goes_through_the_normal_diff_approval(project, pad):
    asked = []

    def approver(kind, title, detail):
        asked.append((kind, title, detail))
        return True

    ws = Workspace(project, approver, scratch=pad.root)
    ws.write_file("scratchpad/app.py", "print(2)\n")  # no prompt
    assert asked == []
    out = ws.move_path("scratchpad/app.py", "src/app.py")
    assert out.startswith("Updated src/app.py from scratchpad/app.py")
    assert asked[0][:2] == ("write", "Update src/app.py") and "+print(2)" in asked[0][2]
    assert (project / "src" / "app.py").read_text() == "print(2)\n"
    assert not (pad.root / "app.py").exists()  # draft consumed


def test_a_declined_promotion_keeps_the_draft(project, pad):
    ws = Workspace(project, lambda *a: False, scratch=pad.root)
    ws.write_file("scratchpad/new.py", "x\n")
    assert "declined" in ws.move_path("scratchpad/new.py", "new.py")
    assert (pad.root / "new.py").exists() and not (project / "new.py").exists()


def test_local_model_drafts_into_the_scratchpad_without_a_prompt(project, pad):
    coder = [ModelEntry(name="c", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)]
    d = Dispatcher(_hw(), catalog=coder, workspace=Workspace(project, _never, scratch=pad.root))
    with patch.dict("localforge.tools.BACKENDS", {"stub": Local("```python\ndef f(): pass\n```")}):
        out = d.dispatch("delegate_coding_task", {"instructions": "draft f", "path": "scratchpad/f.py"})
    assert out.startswith("Created scratchpad/f.py")
    assert (pad.root / "f.py").read_text() == "def f(): pass\n"


def test_the_orchestrator_is_told_where_the_scratchpad_is(project, pad):
    snap = Workspace(project, scratch=pad.root).snapshot()
    assert "Scratchpad: scratchpad/" in snap and str(pad.root) in snap
    assert "scratchpad/" in orch.SYSTEM_PROMPT and "move_path" in orch.SYSTEM_PROMPT


def test_session_scratchpad_is_removed_at_exit(project, monkeypatch):
    monkeypatch.chdir(project)
    pad = cli_module._session.scratchpad_for(project)
    (pad.root / "draft.txt").write_text("x")
    cli_module._save_memory_at_exit()  # nothing to remember, but the scratchpad still goes
    assert not pad.session_dir.exists() and cli_module._session.scratch is None


def test_a_one_off_run_removes_its_scratchpad(project, monkeypatch):
    from localforge import trust

    monkeypatch.chdir(project)
    trust.trust(project)
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)
    seen = {}

    def fake_run(task, frontier_model, workspace=None, **kw):
        seen["scratch"] = workspace.scratch
        return RunResult("ok", RunStats())

    with patch.object(cli_module, "run_orchestrator", side_effect=fake_run):
        CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"])
    assert seen["scratch"] is not None and not seen["scratch"].exists()


def test_scratch_command_lists_and_clears(project, monkeypatch):
    monkeypatch.chdir(project)
    pad = cli_module._session.scratchpad_for(project.resolve())
    (pad.root / "notes.md").write_text("hello")
    out = CliRunner().invoke(cli_module.app, ["scratch"]).output
    assert "scratchpad/notes.md" in out
    CliRunner().invoke(cli_module.app, ["scratch", "clear"])
    assert pad.files() == []


# --- memory: MEMORY.md index + one fact per file ---------------------------------------


def test_remember_writes_a_fact_file_with_frontmatter_and_indexes_it(project):
    name = memory.remember(project, "Prefers Pytest!", "Use pytest, not unittest.", "prefers pytest", "feedback")
    assert name == "prefers-pytest"
    fact = (memory.memory_dir(project) / "prefers-pytest.md").read_text()
    assert fact.startswith("---\nname: prefers-pytest\ndescription: prefers pytest\ntype: feedback\n---")
    assert "Use pytest, not unittest." in fact
    index = (memory.memory_dir(project) / "MEMORY.md").read_text()
    assert index == "- [prefers-pytest](prefers-pytest.md) — prefers pytest\n"


def test_remember_updates_by_name_and_forget_removes(project):
    memory.remember(project, "db", "Uses SQLite.", type="project")
    memory.remember(project, "db", "Moved to Postgres.", type="project")
    facts = memory.list_facts(project)
    assert len(facts) == 1 and facts[0]["content"] == "Moved to Postgres."
    assert memory.forget_fact(project, "db") and memory.list_facts(project) == []
    assert not (memory.memory_dir(project) / "MEMORY.md").exists()


def test_unknown_types_fall_back_and_empty_facts_are_refused(project):
    memory.remember(project, "x", "fact", type="bogus")
    assert memory.list_facts(project)[0]["type"] == "project"
    with pytest.raises(ValueError):
        memory.remember(project, "y", "   ")


def test_memory_is_stored_in_the_project_and_kept_out_of_git(project):
    memory.remember(project, "a", "b")
    memory.save(project, "summary")
    folder = project.resolve() / ".localforge"
    assert (folder / "session.md").is_file() and (folder / "memory" / "MEMORY.md").is_file()
    assert (folder / ".gitignore").read_text().rstrip().endswith("*")  # git ignores the folder by default
    assert not any(p.suffix == ".md" for p in project.rglob("*") if folder not in p.parents)  # nowhere else
    assert not (cli_module.config.CONFIG_DIR / "projects").exists()


def test_old_single_file_memory_is_migrated(project):
    old = cli_module.config.CONFIG_DIR / "memory" / (memory._central_dir(project).name + ".md")
    old.parent.mkdir(parents=True)
    old.write_text("## Goal\nold summary\n")
    assert memory.load(project) == "## Goal\nold summary"
    assert not old.exists() and memory.memory_file(project).exists()


def test_facts_reach_the_orchestrator_every_run(project):
    memory.remember(project, "style", "Keep functions under 30 lines.", type="feedback")
    sent = []

    def fake_completion(**kw):
        from unittest.mock import MagicMock

        sent.append(kw["messages"][0]["content"])
        msg = MagicMock(tool_calls=None, content="ok")
        msg.model_dump.return_value = {"role": "assistant", "content": "ok"}
        return MagicMock(choices=[MagicMock(message=msg)], usage=None)

    with (
        patch.object(orch, "completion", side_effect=fake_completion),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        orch.run("hi", "gpt-5", hardware=_hw(), conversation=Conversation(), workspace=Workspace(project))
    assert "Remembered for this project:" in sent[0] and "Keep functions under 30 lines." in sent[0]


def test_orchestrator_can_remember_and_forget(project):
    d = Dispatcher(_hw(), catalog=[], workspace=Workspace(project))
    assert d.dispatch("remember", {"name": "deploy", "content": "Deploys via fly.io.", "type": "reference"}) == "Remembered deploy."
    assert memory.list_facts(project)[0]["type"] == "reference"
    assert d.dispatch("forget", {"name": "deploy"}) == "Forgot deploy."
    assert "missing the required argument" in d.dispatch("remember", {"name": "x"})
    names = {s["function"]["name"] for s in build_tool_schemas(_hw(), [])}
    assert {"remember", "forget"} <= names


def test_exit_extracts_facts_and_saves_the_summary(project, monkeypatch):
    monkeypatch.chdir(project)
    conv = cli_module._session.conversation_for(project)
    conv.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "always use type hints"},
        {"role": "assistant", "content": "noted"},
    ]
    reply = '{"memories": [{"name": "type-hints", "type": "feedback", "description": "always type hints", "content": "User wants type hints everywhere."}], "forget": []}'
    local = Local(reply)
    monkeypatch.setattr(cli_module, "_memory_dispatcher", lambda hooks=None: Dispatcher(_hw(), catalog=CATALOG, installed={"keeper"}, hooks=hooks))
    with patch.dict("localforge.memory.BACKENDS", {"stub": local}):
        cli_module._save_memory_at_exit()
    facts = memory.list_facts(project)
    assert [f["name"] for f in facts] == ["type-hints"] and facts[0]["type"] == "feedback"
    assert memory.load(project) == reply  # the same stub also wrote the summary
    assert "always use type hints" in local.prompts[0]


def test_extraction_ignores_an_unusable_reply(project):
    conv = Conversation(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"keeper"})
    with patch.dict("localforge.memory.BACKENDS", {"stub": Local("I can't produce JSON, sorry")}):
        assert memory.extract_facts(conv, d, project) == 0
    assert memory.list_facts(project) == []


def test_memory_command_shows_facts_and_summary_and_can_forget(project, monkeypatch):
    monkeypatch.chdir(project)
    memory.remember(project, "tests", "Run make test.", "run make test", "project")
    memory.save(project, "## Current state\nhalf done")
    monkeypatch.setattr(cli_module, "_memory_dispatcher", lambda hooks=None: Dispatcher(_hw(), catalog=[], installed=set()))
    out = " ".join(CliRunner().invoke(cli_module.app, ["memory"]).output.split())
    assert "tests" in out and "run make test" in out and "half done" in out
    CliRunner().invoke(cli_module.app, ["memory", "forget", "tests"])
    assert memory.list_facts(project) == []
    CliRunner().invoke(cli_module.app, ["memory", "clear"])
    assert memory.load(project) == ""


def test_resuming_mentions_summary_and_facts(project, monkeypatch, capsys):
    memory.remember(project, "a", "fact a")
    memory.save(project, "where we left off")
    conv = cli_module._SessionState().conversation_for(project)
    assert conv.memory == "where we left off" and "fact a" in conv.facts


# --- memory moved into the project (asked for: "memory in the trusted folder makes a lot more sense") ---


def test_memory_from_the_old_central_place_is_moved_in(project):
    central = memory._central_dir(project)
    (central / "memory").mkdir(parents=True)
    (central / "session.md").write_text("## Goal\nfrom before\n")
    (central / "usage.json").write_text("{}")
    assert memory.load(project) == "## Goal\nfrom before"
    folder = project.resolve() / ".localforge"
    assert (folder / "usage.json").is_file() and (folder / ".gitignore").is_file()
    assert not central.exists()


def test_file_tools_cannot_touch_localforges_memory(project):
    from localforge.workspace import Workspace, WorkspaceError

    memory.remember(project, "prefers-tabs", "tabs")
    ws = Workspace(project, lambda *a: True)
    with pytest.raises(WorkspaceError, match="remember/forget"):
        ws.resolve(".localforge/memory/MEMORY.md")
    assert ".localforge" not in ws.list_files()  # and it isn't listed or searched


def test_a_moved_project_keeps_its_memory(project, tmp_path_factory):
    memory.remember(project, "prefers-tabs", "tabs, not spaces")
    moved = tmp_path_factory.mktemp("elsewhere") / "renamed-project"
    project.rename(moved)
    assert "tabs, not spaces" in memory.facts_for_prompt(moved)
