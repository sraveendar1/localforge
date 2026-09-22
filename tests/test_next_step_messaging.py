"""Every "what now?" message must point to the interactive session.

Reported: after setup the user was told `Try: localforge run "..."`, when
the natural next step since the interactive session was added is simply
typing `localforge`. These guard the setup/doctor/panel/wizard wording and
the installer ending.
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "localforge"


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    # CONFIG_FILE is resolved at import, so the env var alone would still let
    # config.load() read the developer's real ~/.config/localforge/config.env.
    monkeypatch.setenv("LOCALFORGE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(cli_module.config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli_module.config, "CONFIG_FILE", tmp_path / "config.env")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_configured_panel_leads_with_the_session(monkeypatch):
    monkeypatch.setenv("LOCALFORGE_FRONTIER_MODEL", "claude-opus-5")
    out = _flat(CliRunner().invoke(cli_module.app, []).output)

    assert "Start a session" in out
    assert "/help" in out
    # `localforge run` is still offered, but only as the secondary one-off
    assert out.index("Start a session") < out.index("localforge run")
    assert "One-off without a session" in out


def test_unconfigured_panel_says_to_type_localforge_after_setup(monkeypatch):
    monkeypatch.delenv("LOCALFORGE_FRONTIER_MODEL", raising=False)
    out = _flat(CliRunner().invoke(cli_module.app, []).output)

    assert "localforge setup" in out
    assert "type localforge to start a session" in out


def test_doctor_ready_message_points_to_the_session(monkeypatch):
    monkeypatch.setenv("LOCALFORGE_FRONTIER_MODEL", "ollama/qwen2.5:72b")
    from localforge.hardware import HardwareProfile

    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=64, free_disk_gb=500, gpus=[])
    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module.OllamaBackend, "is_running", return_value=True),
        patch.object(cli_module, "detect_hardware", return_value=hw),
        patch.object(cli_module, "_installed_model_names", return_value={"qwen2.5:72b"}),
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        out = _flat(CliRunner().invoke(cli_module.app, ["doctor"]).output)

    assert "Ready to go. Type localforge to start a session." in out


def test_no_code_still_tells_users_to_try_localforge_run():
    """The old `Try: localforge run "..."` next-step line must not creep back
    into any user-facing message.
    """
    offenders = []
    for path in SRC.glob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"Try:.*localforge run", line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)


def test_wizard_done_screen_points_to_the_session():
    text = (SRC / "tui.py").read_text()
    done = text[text.index("class DoneScreen") :]
    assert "localforge[/bold] in your terminal to start a session" in done


def test_installer_ends_without_opening_the_session():
    """A bare `localforge` opens the session whenever stdin is a terminal,
    which would leave `./install.sh` sitting in a prompt instead of
    finishing. The final call must take stdin from /dev/null.
    """
    last_call = [
        line for line in (ROOT / "install.sh").read_text().splitlines() if line.strip().startswith('"$HOME/.local/bin/localforge"')
    ][-1]
    assert last_call.strip() == '"$HOME/.local/bin/localforge" < /dev/null'
