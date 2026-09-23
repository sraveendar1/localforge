"""`localforge init`: a project brief, like Claude Code's CLAUDE.md, written
and kept current by the local memory model.

Asked for: "a similar function like init in claude that captures the overall
objective of the project and keeps it updated using the memory selected open
weighted llm".
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import brief, memory, trust
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import Dispatcher
from localforge.workspace import Workspace

KEEPER = ModelEntry(name="qwen2.5:7b", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=4, quality_tier=1)
DRAFT = "# Project brief\n\n## What this project is\nA todo API for the team.\n"


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


class Local:
    def __init__(self, reply=DRAFT):
        self.reply, self.prompts = reply, []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        return {"type": "text", "content": self.reply, "tokens": 50}


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "api.py").write_text("def get_todos(): ...\n")
    (tmp_path / "README.md").write_text("# Todo API\nA small service.\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "todo"\ndependencies = ["fastapi"]\n')
    monkeypatch.chdir(tmp_path)
    trust.trust(tmp_path)
    cli_module._session.root = tmp_path.resolve()
    # no terminal under the test runner, so approvals can't be typed: approve
    # them here and test declining separately
    cli_module._session.auto_approve = True
    yield tmp_path.resolve()
    cli_module._session.auto_approve = False


@pytest.fixture
def local_keeper(monkeypatch):
    backend = Local()
    monkeypatch.setattr(cli_module, "_memory_dispatcher", lambda hooks=None: Dispatcher(_hw(), catalog=[KEEPER], installed={KEEPER.name}, hooks=hooks))
    monkeypatch.setitem(__import__("localforge.backends", fromlist=["BACKENDS"]).BACKENDS, "stub", backend)
    return backend


# --- what the local model is given ----------------------------------------------------


def test_the_brief_is_drafted_from_the_project_and_what_localforge_remembers(project, local_keeper):
    memory.remember(project, "deploy", "Deploys to fly.io on merge.", type="project")
    memory.save(project, "Last session: added the /todos endpoint.")
    CliRunner().invoke(cli_module.app, ["init"])

    prompt = local_keeper.prompts[0]
    assert "# Todo API" in prompt and "fastapi" in prompt  # README and manifest
    assert "src/api.py" in prompt  # the layout
    assert "Deploys to fly.io" in prompt and "added the /todos endpoint" in prompt  # memory
    assert "under 500 words" in prompt and "don't invent any" in prompt


def test_the_brief_is_written_to_the_project_after_approval(project, local_keeper):
    out = CliRunner().invoke(cli_module.app, ["init"], ).output
    assert (project / "LOCALFORGE.md").read_text() == DRAFT
    assert "Created LOCALFORGE.md" in out and "Every session in this folder now starts with" in out


def test_declining_writes_nothing(project, local_keeper):
    cli_module._session.auto_approve = False  # declined (no terminal to approve on)
    out = CliRunner().invoke(cli_module.app, ["init"]).output
    assert not (project / "LOCALFORGE.md").exists()
    assert "declined" in out


def test_a_refresh_rewrites_from_scratch_but_an_update_builds_on_what_is_there(project, local_keeper):
    (project / "LOCALFORGE.md").write_text("# Project brief\n\n## What this project is\nThe old description.\n")
    CliRunner().invoke(cli_module.app, ["init"])
    assert "The old description." in local_keeper.prompts[-1] and "keep what's still true" in local_keeper.prompts[-1]

    CliRunner().invoke(cli_module.app, ["init", "--refresh"])
    assert "The old description." not in local_keeper.prompts[-1]


def test_markdown_wrapped_in_a_code_fence_is_unwrapped(project, monkeypatch, local_keeper):
    local_keeper.reply = "```markdown\n# Project brief\n\nIt is a todo API.\n```"
    CliRunner().invoke(cli_module.app, ["init"])
    assert (project / "LOCALFORGE.md").read_text().startswith("# Project brief")
    assert "```" not in (project / "LOCALFORGE.md").read_text()


def test_without_a_local_model_it_says_so_instead_of_using_the_paid_one(project, monkeypatch):
    monkeypatch.setattr(cli_module, "_memory_dispatcher", lambda hooks=None: Dispatcher(_hw(), catalog=[], installed=set(), hooks=hooks))
    result = CliRunner().invoke(cli_module.app, ["init"])
    assert result.exit_code == 1 and "No local model is available" in result.output
    assert not (project / "LOCALFORGE.md").exists()


# --- every session starts with it ---------------------------------------------------


def test_the_brief_reaches_the_orchestrator(project):
    (project / "LOCALFORGE.md").write_text("# Project brief\n\nA todo API for the team.\n")
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
        orch.run("hi", "gpt-5", hardware=_hw(), conversation=orch.Conversation(), workspace=Workspace(project))
    assert "Project brief" in sent[0] and "A todo API for the team." in sent[0]
    assert "trust it over guesswork" in sent[0]


def test_another_tools_brief_is_read_rather_than_duplicated(project):
    (project / "CLAUDE.md").write_text("# Notes for Claude\n\nRun make test.\n")
    assert "Run make test." in brief.brief_for_prompt(project)
    (project / "LOCALFORGE.md").write_text("# Project brief\n\nOurs wins.\n")
    assert "Ours wins." in brief.brief_for_prompt(project)  # its own comes first


# --- keeping it up to date -----------------------------------------------------------


def test_after_a_session_that_changed_files_an_update_is_drafted_and_left_pending(project, local_keeper, monkeypatch):
    (project / "LOCALFORGE.md").write_text("# Project brief\n\nThe old description.\n")
    conv = cli_module._session.conversation_for(project)
    conv.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "add an endpoint"}, {"role": "assistant", "content": "done"}]
    cli_module._session.files_changed = 2
    local_keeper.reply = "# Project brief\n\nNow with two endpoints.\n"

    cli_module._save_memory_at_exit()
    assert brief.pending_path(project).is_file()
    assert (project / "LOCALFORGE.md").read_text() == "# Project brief\n\nThe old description.\n"  # untouched


def test_no_changes_means_no_pending_update(project, local_keeper):
    (project / "LOCALFORGE.md").write_text("# Project brief\n\nUnchanged.\n")
    conv = cli_module._session.conversation_for(project)
    conv.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "just a question"}, {"role": "assistant", "content": "an answer"}]
    cli_module._session.files_changed = 0
    cli_module._save_memory_at_exit()
    assert not brief.pending_path(project).exists()


def test_init_offers_the_pending_update_without_asking_the_model_again(project, local_keeper):
    (project / "LOCALFORGE.md").write_text("# Project brief\n\nOld.\n")
    brief.save_pending(project, "# Project brief\n\nDrafted at the end of the last session.\n")
    out = CliRunner().invoke(cli_module.app, ["init"], ).output
    assert "update drafted at the end of the last session" in out
    assert "Drafted at the end of the last session." in (project / "LOCALFORGE.md").read_text()
    assert local_keeper.prompts == []  # no second draft needed
    assert not brief.pending_path(project).exists()  # consumed


def test_approved_writes_count_as_changes(project):
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        cli_module._session.files_changed = 0
        cli_module._session.auto_approve = True
        activity = cli_module._LiveActivity("m")
        activity.approve("write", "Create a.py", "+x")
        activity.approve("command", "Run command", "ls")  # not a file change
        activity.approve("delete", "Delete b.py", "File b.py")
    finally:
        cli_module._session.auto_approve = False
        cli_module.console.pop_theme()
    assert cli_module._session.files_changed == 2
