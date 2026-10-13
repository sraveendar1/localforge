"""The desktop app starts the backend in the home folder and passes the project in
LOCALFORGE_PROJECT_DIR (macOS refuses getcwd() inside protected folders the app wasn't granted)."""

import os

import pytest
import typer

from localforge import cli


def test_enters_the_project_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_PROJECT_DIR", str(tmp_path))
    monkeypatch.chdir(os.path.expanduser("~") if os.path.isdir(os.path.expanduser("~")) else tmp_path)
    cli._enter_project_dir()
    assert os.getcwd() == str(tmp_path.resolve()) or os.path.samefile(os.getcwd(), tmp_path)


def test_no_variable_means_no_change(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALFORGE_PROJECT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    cli._enter_project_dir()
    assert os.path.samefile(os.getcwd(), tmp_path)


def test_getcwd_is_answered_when_the_system_refuses_to_report_it(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_PROJECT_DIR", str(tmp_path))
    real = os.getcwd
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(1, "Operation not permitted")
        return real()

    monkeypatch.setattr(os, "getcwd", flaky)
    cli._enter_project_dir()
    assert os.getcwd() == os.path.abspath(tmp_path)


def test_an_unenterable_folder_gives_a_readable_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCALFORGE_PROJECT_DIR", str(tmp_path / "missing"))
    with pytest.raises(typer.Exit):
        cli._enter_project_dir()
    out = capsys.readouterr().out
    assert '"fatal": true' in out and "can't open" in out and "Files and Folders" in out
