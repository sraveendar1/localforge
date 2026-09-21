"""Setup must never auto-select a provider or model.

Regression tests for a reported bug: setup "didn't ask for a selection and
went directly to Anthropic". Cause was a pre-filled `default="anthropic"` on
the provider prompt, so a single stray/buffered Enter (e.g. the one pressed
at install.sh's "Press Enter to continue") was consumed by it and silently
accepted the default without the user ever choosing.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge.hardware import HardwareProfile


class StubOllama:
    def __init__(self, *a, **k):
        self.pulled: list[str] = []

    def is_running(self):
        return True

    def ensure_available(self, model_name, on_progress=None):
        self.pulled.append(model_name)


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_CONFIG_DIR", str(tmp_path / "config"))
    cli_module.config.CONFIG_DIR = tmp_path / "config"
    cli_module.config.CONFIG_FILE = cli_module.config.CONFIG_DIR / "config.env"
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    stub = StubOllama()
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])
    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module, "OllamaBackend", lambda *a, **k: stub),
        patch.object(cli_module, "detect_hardware", return_value=hw),
        patch.object(cli_module, "recommend_models", return_value={}),
        patch.object(cli_module, "webbrowser"),
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        yield stub


def _saved(tmp_path_config_file) -> str:
    return tmp_path_config_file.read_text() if tmp_path_config_file.exists() else ""


def test_stray_enter_does_not_auto_select_anthropic(setup_env):
    """THE reported bug: a blank line must re-ask, never pick a provider."""
    # two blank Enters, then deliberately choose 4 (local), then model 2
    result = CliRunner().invoke(cli_module.app, ["setup"], input="\n\n4\n2\n")

    assert result.exit_code == 0
    # the provider question was asked three times: twice re-asked, once answered
    assert result.output.count("Choose a provider") == 3
    # and the blank answers did NOT select Anthropic
    saved = _saved(cli_module.config.CONFIG_FILE)
    assert "ANTHROPIC_API_KEY" not in saved
    assert "LOCALFORGE_FRONTIER_MODEL=ollama/qwen2.5:72b" in saved


def test_provider_menu_is_numbered_with_every_option(setup_env):
    result = CliRunner().invoke(cli_module.app, ["setup"], input="4\n1\n")
    assert result.exit_code == 0
    for n, name in enumerate(["Anthropic", "OpenAI", "Gemini", "Local"], start=1):
        assert f"{n}) {name}" in result.output


def test_provider_selectable_by_number(setup_env):
    """Picking '4' must select local, not fall through to a default."""
    result = CliRunner().invoke(cli_module.app, ["setup"], input="4\n1\n")
    assert result.exit_code == 0
    assert "LOCALFORGE_FRONTIER_MODEL=ollama/llama3.1:70b" in _saved(cli_module.config.CONFIG_FILE)


def test_invalid_provider_reasks_instead_of_defaulting(setup_env):
    result = CliRunner().invoke(cli_module.app, ["setup"], input="banana\n9\n4\n1\n")
    assert result.exit_code == 0
    assert "isn't one of the options" in result.output
    assert "LOCALFORGE_FRONTIER_MODEL=ollama/llama3.1:70b" in _saved(cli_module.config.CONFIG_FILE)


def test_model_choice_has_no_default_either(setup_env):
    """A blank answer at the model menu must re-ask, not pick choice 1."""
    # provider 4 (local), blank Enter at model menu, then choose 2
    result = CliRunner().invoke(cli_module.app, ["setup"], input="4\n\n2\n")
    assert result.exit_code == 0
    assert result.output.count("Choose a model") == 2  # re-asked once
    assert "LOCALFORGE_FRONTIER_MODEL=ollama/qwen2.5:72b" in _saved(cli_module.config.CONFIG_FILE)


def test_model_menu_is_numbered(setup_env):
    result = CliRunner().invoke(cli_module.app, ["setup"], input="1\n1\n2\nsk-fake\n")  # anthropic, api key auth, model 2, key
    assert result.exit_code == 0
    assert "1) claude-opus-5" in result.output
    assert "2) claude-sonnet-5" in result.output
    # chose 2 -> Sonnet, definitely not the first entry by default
    assert "LOCALFORGE_FRONTIER_MODEL=claude-sonnet-5" in _saved(cli_module.config.CONFIG_FILE)
