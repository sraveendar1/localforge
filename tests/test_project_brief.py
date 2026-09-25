"""`localforge goals` (`/init` is a backward-compatible alias): a project
brief, like Claude Code's CLAUDE.md, written and kept current by the local
memory model.

Asked for: "a similar function like init in claude that captures the overall
objective of the project and keeps it updated using the memory selected open
weighted llm". Renamed to `goals`, since it's really capturing what the
project is *for*, not just a one-time init step -- and it's no longer only a
one-time step: the first task in a goal-less project offers to draft it
right then, and a mid-session prompt offers to refresh it once enough has
changed.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import brief, memory, trust
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.orchestrator import RunResult, RunStats
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
    CliRunner().invoke(cli_module.app, ["goals"])

    prompt = local_keeper.prompts[0]
    assert "# Todo API" in prompt and "fastapi" in prompt  # README and manifest
    assert "src/api.py" in prompt  # the layout
    assert "Deploys to fly.io" in prompt and "added the /todos endpoint" in prompt  # memory
    assert "under 500 words" in prompt and "don't invent any" in prompt


def test_the_brief_is_written_to_the_project_after_approval(project, local_keeper):
    out = CliRunner().invoke(cli_module.app, ["goals"], ).output
    assert (project / "AGENTS.md").read_text() == DRAFT
    assert "Created AGENTS.md" in out and "Every session in this folder now starts with" in out


def test_init_is_still_a_working_alias_for_goals(project, local_keeper):
    out = CliRunner().invoke(cli_module.app, ["init"]).output
    assert (project / "AGENTS.md").read_text() == DRAFT
    assert "Created AGENTS.md" in out


def test_declining_writes_nothing(project, local_keeper):
    cli_module._session.auto_approve = False  # declined (no terminal to approve on)
    out = CliRunner().invoke(cli_module.app, ["goals"]).output
    assert not (project / "AGENTS.md").exists()
    assert "declined" in out


def test_a_refresh_rewrites_from_scratch_but_an_update_builds_on_what_is_there(project, local_keeper):
    (project / "AGENTS.md").write_text("# Project brief\n\n## What this project is\nThe old description.\n")
    CliRunner().invoke(cli_module.app, ["goals"])
    assert "The old description." in local_keeper.prompts[-1] and "keep what's still true" in local_keeper.prompts[-1]

    CliRunner().invoke(cli_module.app, ["goals", "--refresh"])
    assert "The old description." not in local_keeper.prompts[-1]


def test_markdown_wrapped_in_a_code_fence_is_unwrapped(project, monkeypatch, local_keeper):
    local_keeper.reply = "```markdown\n# Project brief\n\nIt is a todo API.\n```"
    CliRunner().invoke(cli_module.app, ["goals"])
    assert (project / "AGENTS.md").read_text().startswith("# Project brief")
    assert "```" not in (project / "AGENTS.md").read_text()


def test_without_a_local_model_it_says_so_instead_of_using_the_paid_one(project, monkeypatch):
    monkeypatch.setattr(cli_module, "_memory_dispatcher", lambda hooks=None: Dispatcher(_hw(), catalog=[], installed=set(), hooks=hooks))
    result = CliRunner().invoke(cli_module.app, ["goals"])
    assert result.exit_code == 1 and "No local model is available" in result.output
    assert not (project / "AGENTS.md").exists()


# --- every session starts with it ---------------------------------------------------


def test_the_brief_reaches_the_orchestrator(project):
    (project / "AGENTS.md").write_text("# Project brief\n\nA todo API for the team.\n")
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
    (project / "AGENTS.md").write_text("# Project brief\n\nOurs wins.\n")
    assert "Ours wins." in brief.brief_for_prompt(project)  # its own comes first


def test_the_old_name_is_still_read_and_init_moves_it_to_agents_md(project, local_keeper):
    (project / "LOCALFORGE.md").write_text("# Project brief\n\nThe old description.\n")
    assert "The old description." in brief.brief_for_prompt(project)  # still read, before CLAUDE.md
    local_keeper.reply = "# Project brief\n\nThe updated description.\n"
    result = CliRunner().invoke(cli_module.app, ["goals"], input="y\ny\n")
    assert (project / "AGENTS.md").read_text().startswith("# Project brief")
    assert not (project / "LOCALFORGE.md").exists(), result.output  # moved, after its own approval



# --- keeping it up to date -----------------------------------------------------------


def test_after_a_session_that_changed_files_an_update_is_drafted_and_left_pending(project, local_keeper, monkeypatch):
    (project / "AGENTS.md").write_text("# Project brief\n\nThe old description.\n")
    conv = cli_module._session.conversation_for(project)
    conv.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "add an endpoint"}, {"role": "assistant", "content": "done"}]
    cli_module._session.files_changed = 2
    local_keeper.reply = "# Project brief\n\nNow with two endpoints.\n"

    cli_module._save_memory_at_exit()
    assert brief.pending_path(project).is_file()
    assert (project / "AGENTS.md").read_text() == "# Project brief\n\nThe old description.\n"  # untouched


def test_no_changes_means_no_pending_update(project, local_keeper):
    (project / "AGENTS.md").write_text("# Project brief\n\nUnchanged.\n")
    conv = cli_module._session.conversation_for(project)
    conv.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "just a question"}, {"role": "assistant", "content": "an answer"}]
    cli_module._session.files_changed = 0
    cli_module._save_memory_at_exit()
    assert not brief.pending_path(project).exists()


def test_init_offers_the_pending_update_without_asking_the_model_again(project, local_keeper):
    (project / "AGENTS.md").write_text("# Project brief\n\nOld.\n")
    brief.save_pending(project, "# Project brief\n\nDrafted at the end of the last session.\n")
    out = CliRunner().invoke(cli_module.app, ["goals"], ).output
    assert "update drafted at the end of the last session" in out
    assert "Drafted at the end of the last session." in (project / "AGENTS.md").read_text()
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


# --- a drafted update is offered, not left to drift ------------------------------------------


def test_a_drafted_update_is_offered_at_the_next_session_start(project, monkeypatch):
    brief.save_pending(project, "# Project brief\n\nNow with two endpoints.\n")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 1)
    ran = []
    monkeypatch.setattr(cli_module, "_write_goals", lambda folder, **kw: ran.append(folder) or True)
    cli_module._offer_brief_update_at_start(project)
    assert ran == [project]  # reviewed through /goals: the diff and approval as usual


def test_later_leaves_the_draft_for_goals(project, monkeypatch):
    brief.save_pending(project, "# Project brief\n\nDraft.\n")
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 2)
    monkeypatch.setattr(cli_module, "_write_goals", lambda *a, **kw: pytest.fail("ran /goals"))
    cli_module._offer_brief_update_at_start(project)
    assert brief.pending_path(project).is_file()


def test_no_draft_means_no_question(project, monkeypatch):
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda *a: pytest.fail("asked"))
    cli_module._offer_brief_update_at_start(project)


# --- offering to set goals from a project's first task ---------------------------------------


def _run_directly(task: str, **kw) -> None:
    """CliRunner's `invoke()` swaps out `sys.stdin` for its own isolated
    stream for the call, so patching `sys.stdin.isatty` beforehand has no
    effect on code reached through it -- call `run()` as a plain function
    instead, like the rest of the suite does for isatty-sensitive paths
    (see `_offer_upgrades_at_start`/`_offer_brief_update_at_start` tests)."""
    cli_module.run(task=task, frontier_model=kw.pop("frontier_model", "claude-opus-5"), show_usage=False, yes=kw.pop("yes", False))


def test_the_first_task_in_a_goal_less_project_offers_to_draft_goals(project, local_keeper, monkeypatch, capsys):
    """Asked for: the "what do you want to build" moment should itself be
    the moment goals are captured, seeded with that task -- not a separate
    step someone has to remember."""
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 1)  # yes, set them up
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        with patch.object(cli_module, "run_orchestrator", return_value=RunResult("done", RunStats())):
            _run_directly("build a todo API")
    finally:
        cli_module.console.pop_theme()
    out = capsys.readouterr().out

    assert (project / "AGENTS.md").read_text() == DRAFT
    assert "No AGENTS.md yet for this project" in out
    assert "build a todo API" in local_keeper.prompts[0]  # seeded with the task, not just a cold read


def test_declining_the_first_task_offer_is_not_asked_again_this_session(project, monkeypatch, capsys):
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    asked = []

    def fake_ask_number(prompt, count):
        asked.append(1)
        return 2  # no, not now

    monkeypatch.setattr(cli_module, "_ask_number", fake_ask_number)
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        with patch.object(cli_module, "run_orchestrator", return_value=RunResult("done", RunStats())):
            _run_directly("one")
            capsys.readouterr()  # discard the first task's output
            _run_directly("two")
    finally:
        cli_module.console.pop_theme()
    out = capsys.readouterr().out

    assert not (project / "AGENTS.md").exists()
    assert len(asked) == 1  # only offered once, on the first task
    assert "No AGENTS.md yet" not in out


def test_a_scripted_run_is_never_asked_about_goals(project, capsys):
    """Default test environment: no real tty, so this must never fire --
    a scripted/CI `localforge run` must not block on a prompt."""
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        with patch.object(cli_module, "run_orchestrator", return_value=RunResult("done", RunStats())):
            _run_directly("one", yes=True)
    finally:
        cli_module.console.pop_theme()
    out = capsys.readouterr().out
    assert not (project / "AGENTS.md").exists()
    assert "No AGENTS.md yet" not in out


# --- offering to refresh goals mid-session, not only at the next start -----------------------


def test_goals_refresh_is_offered_once_enough_has_changed_mid_session(project, local_keeper, monkeypatch, capsys):
    (project / "AGENTS.md").write_text("# Project brief\n\nOld.\n")
    cli_module._session.files_changed = cli_module.GOALS_REFRESH_THRESHOLD
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 2)  # not now
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        with patch.object(cli_module, "run_orchestrator", return_value=RunResult("done", RunStats())):
            _run_directly("one")
            out = capsys.readouterr().out
            assert "may be out of date" in out
            # The watermark moved up, so it isn't offered again immediately after.
            _run_directly("two")
            out2 = capsys.readouterr().out
    finally:
        cli_module.console.pop_theme()
    assert "may be out of date" not in out2


def test_goals_refresh_is_not_offered_below_the_threshold(project, local_keeper, monkeypatch, capsys):
    (project / "AGENTS.md").write_text("# Project brief\n\nOld.\n")
    cli_module._session.files_changed = cli_module.GOALS_REFRESH_THRESHOLD - 1
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda *a: pytest.fail("asked"))
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        with patch.object(cli_module, "run_orchestrator", return_value=RunResult("done", RunStats())):
            _run_directly("one")
    finally:
        cli_module.console.pop_theme()
    out = capsys.readouterr().out
    assert "may be out of date" not in out


def test_goals_offers_never_read_stdin_from_the_background_worker_thread(project, local_keeper, monkeypatch):
    """Reported: "not seeing the ability to queue... single threaded." Both
    goals offers call _ask_number, a blocking console.input() -- fine on the
    main thread, but on the background worker thread (background.py's
    TaskRunner) it would read stdin out from under prompt_toolkit's own
    PromptSession, which owns the terminal for the whole session. That
    silently broke background tasks (and therefore queueing behind one) for
    any project with no AGENTS.md yet, or one with 5+ changed files. The
    offer must be skipped -- not just answered a particular way -- whenever
    `run()` executes on the worker thread; _ask_number must never even be
    called there."""
    from localforge.background import TaskRunner

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda *a: pytest.fail("_ask_number was called from the worker thread"))

    def run_task(task: str) -> None:
        cli_module.run(task=task, frontier_model="claude-opus-5", show_usage=False, yes=False)

    runner = TaskRunner(run_task)
    cli_module._session.runner = runner
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        with patch.object(cli_module, "run_orchestrator", return_value=RunResult("done", RunStats())):
            runner.submit("build a todo API")
            runner.thread.join(timeout=5)
    finally:
        cli_module.console.pop_theme()
        cli_module._session.runner = None
    assert not runner.busy
    # Deferred, not lost: skipped on the worker thread rather than consumed,
    # so a later foreground opportunity can still offer it.
    assert not (project / "AGENTS.md").exists()
    assert cli_module._session.goals_offered is False


def test_the_deferred_goals_offer_still_fires_once_off_the_worker_thread(project, local_keeper, monkeypatch, capsys):
    """The flip side of the test above: run() on the main thread (no
    background runner in play) must still make the offer as before -- the
    fix is thread-aware, not a blanket disable."""
    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_module, "_ask_number", lambda prompt, count: 1)  # yes, set them up
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        with patch.object(cli_module, "run_orchestrator", return_value=RunResult("done", RunStats())):
            _run_directly("build a todo API")
    finally:
        cli_module.console.pop_theme()
    out = capsys.readouterr().out
    assert (project / "AGENTS.md").read_text() == DRAFT
    assert "No AGENTS.md yet for this project" in out
