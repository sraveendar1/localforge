"""`/desktop`: hand off a terminal session to the desktop app -- launch it
open to the current folder and end this session so the GUI (which loads
memory.load(root) on start) picks up where the terminal left off.
"""
import platform

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module


@pytest.fixture(autouse=True)
def _clear_desktop_bin_override(monkeypatch):
    monkeypatch.delenv("LOCALFORGE_DESKTOP_BIN", raising=False)


def test_desktop_command_fails_cleanly_with_no_app_installed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_find_desktop_app_binary", lambda: None)
    result = CliRunner().invoke(cli_module.app, ["desktop"])
    assert result.exit_code == 1
    assert "Couldn't find an installed desktop app" in result.output


def test_desktop_command_launches_the_found_binary_and_exits_zero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli_module, "_find_desktop_app_binary", lambda: tmp_path / "fake-desktop")
    monkeypatch.setattr(cli_module.subprocess, "Popen", lambda args: calls.append(args))

    result = CliRunner().invoke(cli_module.app, ["desktop"])

    assert result.exit_code == 0
    assert "Launched the desktop app" in result.output
    assert len(calls) == 1
    assert calls[0] == [str(tmp_path / "fake-desktop"), str(tmp_path.resolve())]


def test_desktop_command_uses_macos_open_for_an_app_bundle(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    app_bundle = tmp_path / "LocalForge Desktop.app"
    monkeypatch.setattr(cli_module, "_find_desktop_app_binary", lambda: app_bundle)
    monkeypatch.setattr(cli_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli_module.subprocess, "Popen", lambda args: calls.append(args))

    result = CliRunner().invoke(cli_module.app, ["desktop"])

    assert result.exit_code == 0
    assert calls == [["open", "-a", str(app_bundle), "--args", str(tmp_path.resolve())]]


def test_env_override_takes_priority_and_must_exist(monkeypatch, tmp_path):
    real_bin = tmp_path / "my-localforge-desktop"
    real_bin.write_text("#!/bin/sh\n")
    monkeypatch.setenv("LOCALFORGE_DESKTOP_BIN", str(real_bin))
    assert cli_module._find_desktop_app_binary() == real_bin

    monkeypatch.setenv("LOCALFORGE_DESKTOP_BIN", str(tmp_path / "does-not-exist"))
    assert cli_module._find_desktop_app_binary() is None


def test_linux_lookup_uses_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    fake = tmp_path / "localforge-desktop"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: str(fake) if name == "localforge-desktop" else None)
    assert cli_module._find_desktop_app_binary() == fake


def test_no_desktop_app_found_on_linux_without_a_match(monkeypatch):
    monkeypatch.setattr(cli_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: None)
    assert cli_module._find_desktop_app_binary() is None
