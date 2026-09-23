"""Folder trust (like Claude Code) and full create/read/update/delete on a
trusted folder, each change still behind its own permission prompt."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge import trust
from localforge.hardware import HardwareProfile
from localforge.orchestrator import RunResult, RunStats
from localforge.tools import Dispatcher, build_tool_schemas
from localforge.workspace import Workspace, WorkspaceError


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def _yes(*a):
    return True


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "src" / "b.py").write_text("y = 2\n")
    (tmp_path / "notes.txt").write_text("hi\n")
    return tmp_path


# --- trust ---------------------------------------------------------------------------


@pytest.mark.untrusted
def test_trust_is_remembered_and_covers_subfolders(tmp_path):
    assert not trust.is_trusted(tmp_path)
    trust.trust(tmp_path)
    assert trust.is_trusted(tmp_path) and trust.is_trusted(tmp_path / "deep" / "er")
    assert not trust.is_trusted(tmp_path.parent)
    assert trust.untrust(tmp_path) and not trust.is_trusted(tmp_path)


@pytest.mark.untrusted
def test_trust_file_lives_in_localforge_config_not_the_project(tmp_path):
    trust.trust(tmp_path)
    assert trust._file().parent == cli_module.config.CONFIG_DIR
    assert not (tmp_path / "trusted_folders.json").exists()


@pytest.fixture
def ask(monkeypatch):
    monkeypatch.setattr(cli_module, "_drain_buffered_input", lambda: None)
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    yield lambda answers: monkeypatch.setattr(cli_module.console, "input", lambda prompt="": next(answers))
    cli_module.console.pop_theme()


@pytest.mark.untrusted
def test_trust_prompt_needs_an_explicit_yes(tmp_path, ask):
    ask(iter(["", "3", "1"]))  # Enter and junk re-ask; no default
    assert cli_module._ask_trust(tmp_path) is True
    assert trust.is_trusted(tmp_path)


@pytest.mark.untrusted
def test_saying_no_does_not_trust_and_ends_the_session(tmp_path, ask, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ask(iter(["2"]))
    picker = MagicMock()
    monkeypatch.setattr(cli_module, "_choose_orchestrator_at_start", picker)
    assert cli_module._start_session() is False
    assert not trust.is_trusted(tmp_path)
    picker.assert_not_called()  # never gets as far as choosing a model


@pytest.mark.untrusted
def test_a_trusted_folder_is_not_asked_again(tmp_path, ask):
    trust.trust(tmp_path)
    ask(iter([]))  # any prompt would raise StopIteration
    assert cli_module._ask_trust(tmp_path) is True


@pytest.mark.untrusted
def test_one_off_run_refuses_an_untrusted_folder_without_a_terminal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)
    with patch.object(cli_module, "run_orchestrator") as run:
        result = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"])
    assert result.exit_code == 1 and "isn't a trusted folder" in result.output
    run.assert_not_called()
    with patch.object(cli_module, "run_orchestrator", return_value=RunResult("ok", RunStats())) as run:
        result = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5", "--yes"])
    run.assert_called_once()  # --yes is explicit consent for this run


# --- create / read / update / delete ------------------------------------------------------


def test_crud_tools_are_offered():
    names = {s["function"]["name"] for s in build_tool_schemas(_hw(), [])}
    assert {"read_file", "list_files", "search", "edit_file", "make_dir", "move_path", "delete_path"} <= names


def test_make_dir_and_move(project):
    ws = Workspace(project, _yes)
    assert ws.make_dir("docs/api") == "Created folder docs/api/."
    assert (project / "docs" / "api").is_dir()
    assert ws.move_path("notes.txt", "docs/notes.md") == "Moved notes.txt to docs/notes.md."
    assert (project / "docs" / "notes.md").read_text() == "hi\n" and not (project / "notes.txt").exists()
    with pytest.raises(WorkspaceError, match="already exists"):
        ws.move_path("src/a.py", "src/b.py")


def test_delete_a_file_shows_exactly_what_goes(project):
    seen = {}

    def approver(kind, title, detail):
        seen.update(kind=kind, title=title, detail=detail)
        return True

    assert Workspace(project, approver).delete_path("notes.txt") == "Deleted notes.txt."
    assert seen == {"kind": "delete", "title": "Delete notes.txt", "detail": "File notes.txt (3 bytes)"}
    assert not (project / "notes.txt").exists()


def test_deleting_a_non_empty_folder_needs_recursive_and_lists_its_files(project):
    seen = {}

    def approver(kind, title, detail):
        seen["detail"] = detail
        return True

    ws = Workspace(project, approver)
    with pytest.raises(WorkspaceError, match="pass recursive=true"):
        ws.delete_path("src")
    assert ws.delete_path("src", recursive=True) == "Deleted src."
    assert "src/a.py" in seen["detail"] and "src/b.py" in seen["detail"] and "2 file(s)" in seen["detail"]
    assert not (project / "src").exists()


def test_declined_delete_removes_nothing(project):
    out = Workspace(project, lambda *a: False).delete_path("notes.txt")
    assert "declined" in out and (project / "notes.txt").exists()


@pytest.mark.parametrize("path", [".", "", "../x", "/etc/hosts"])
def test_delete_cannot_touch_the_root_or_escape(project, path):
    with pytest.raises(WorkspaceError):
        Workspace(project, _yes).delete_path(path)
    assert (project / "src" / "a.py").exists()


def test_deleting_a_symlink_removes_the_link_not_the_target(project, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "keep.txt").write_text("keep")
    (project / "link.txt").symlink_to(outside / "keep.txt")
    # resolving follows the link outside the project, so it's refused outright
    with pytest.raises(WorkspaceError, match="outside the project"):
        Workspace(project, _yes).delete_path("link.txt")
    assert (outside / "keep.txt").read_text() == "keep"


def test_always_allowing_file_changes_does_not_cover_deletes(monkeypatch, project):
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli_module, "_drain_buffered_input", lambda: None)
        answers = iter(["a", "n"])
        monkeypatch.setattr(cli_module.console, "input", lambda prompt="": next(answers))
        act = cli_module._LiveActivity("m")
        assert act.approve("write", "Create x", "+x") is True  # "always" for file changes
        assert act.approve("write", "Create y", "+y") is True  # no prompt now
        assert act.approve("delete", "Delete y", "File y") is False  # deletes still ask -> "n"
    finally:
        cli_module.console.pop_theme()


def test_dispatcher_routes_crud_calls(project):
    d = Dispatcher(_hw(), catalog=[], workspace=Workspace(project, _yes))
    assert d.dispatch("make_dir", {"path": "out"}) == "Created folder out/."
    assert d.dispatch("move_path", {"source": "notes.txt", "destination": "out/n.txt"}) == "Moved notes.txt to out/n.txt."
    assert d.dispatch("delete_path", {"path": "out", "recursive": True}) == "Deleted out."
    assert "missing the required argument" in d.dispatch("move_path", {"source": "src"})
