"""A double-clicked .app gets macOS's minimal launchd PATH, not the user's
shell PATH -- so `claude` (or any other CLI tool installed via a shell rc
file: Homebrew, nvm, the tool's own installer) resolves fine from a
terminal (`npm run tauri dev`) but not from `/Applications/LocalForge
Desktop.app`. Reported live: the installed app said "The 'claude' CLI is
required" right after a terminal-launched dev build had just used it
successfully. `_augment_path_for_gui_launch()` merges in whatever a login
shell resolves before `serve --stdio` does any `shutil.which()` lookups.
"""
import os

import localforge.cli as cli_module


def test_merges_in_whatever_the_login_shell_resolves(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("SHELL", "/bin/zsh")

    def fake_run(cmd, **kwargs):
        assert cmd == ["/bin/zsh", "-ilc", "echo -n $PATH"]
        return type("R", (), {"stdout": "/opt/homebrew/bin:/usr/bin:/bin:/Users/me/.claude/local"})()

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    monkeypatch.setattr(cli_module.platform, "system", lambda: "Darwin")

    cli_module._augment_path_for_gui_launch()

    parts = os.environ["PATH"].split(os.pathsep)
    assert "/opt/homebrew/bin" in parts
    assert "/Users/me/.claude/local" in parts
    # No duplicates from the overlapping /usr/bin:/bin already present.
    assert parts.count("/usr/bin") == 1 and parts.count("/bin") == 1


def test_a_failed_login_shell_leaves_path_untouched(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(cli_module.platform, "system", lambda: "Darwin")

    def raises(cmd, **kwargs):
        raise OSError("no such shell")

    monkeypatch.setattr(cli_module.subprocess, "run", raises)

    cli_module._augment_path_for_gui_launch()

    assert os.environ["PATH"] == "/usr/bin:/bin"


def test_windows_is_a_no_op(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(cli_module.platform, "system", lambda: "Windows")

    def boom(cmd, **kwargs):
        raise AssertionError("should never shell out on Windows")

    monkeypatch.setattr(cli_module.subprocess, "run", boom)

    cli_module._augment_path_for_gui_launch()

    assert os.environ["PATH"] == "/usr/bin:/bin"


def test_serve_command_augments_path_before_doing_anything_else(monkeypatch, tmp_path):
    """Regression: the fix has to run before serve_stdio() (and therefore
    before any trust/model resolution) does its first shutil.which() --
    once the desktop app's trust protocol was added, that first check
    could happen very early in the session."""
    calls = []
    monkeypatch.setattr(cli_module, "_augment_path_for_gui_launch", lambda: calls.append("augment"))
    monkeypatch.setattr(cli_module, "serve_stdio", lambda *a, **k: calls.append("serve_stdio"), raising=False)

    from localforge import serve as serve_module
    monkeypatch.setattr(serve_module, "serve_stdio", lambda *a, **k: calls.append("serve_stdio"))

    from typer.testing import CliRunner
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_module.app, ["serve", "--stdio"])

    assert result.exit_code == 0
    assert calls == ["augment", "serve_stdio"]
