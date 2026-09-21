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
def isolated_setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_CONFIG_DIR", str(tmp_path / "config"))
    cli_module.config.CONFIG_DIR = tmp_path / "config"
    cli_module.config.CONFIG_FILE = cli_module.config.CONFIG_DIR / "config.env"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    stub = StubOllama()
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])

    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module, "OllamaBackend", lambda *a, **k: stub),
        patch.object(cli_module, "detect_hardware", return_value=hw),
        patch.object(cli_module, "recommend_models", return_value={}),
        patch.object(cli_module, "webbrowser") as mock_webbrowser,
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        yield mock_webbrowser


@pytest.mark.parametrize(
    "provider,expected_url",
    [
        ("anthropic", "https://console.anthropic.com/settings/keys"),
        ("openai", "https://platform.openai.com/api-keys"),
        ("gemini", "https://aistudio.google.com/app/apikey"),
    ],
)
def test_setup_opens_the_right_console_url_when_no_key_is_present(isolated_setup_env, provider, expected_url):
    result = CliRunner().invoke(
        cli_module.app,
        ["setup"],
        input=f"{provider}\n1\n1\nsk-fake-key\n",  # provider, auth=api key, model 1, key
    )
    assert result.exit_code == 0
    isolated_setup_env.open.assert_called_once_with(expected_url)
    assert expected_url in result.output


def test_setup_does_not_open_browser_when_key_already_present(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALFORGE_CONFIG_DIR", str(tmp_path / "config"))
    cli_module.config.CONFIG_DIR = tmp_path / "config"
    cli_module.config.CONFIG_FILE = cli_module.config.CONFIG_DIR / "config.env"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-already-set")

    stub = StubOllama()
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])

    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module, "OllamaBackend", lambda *a, **k: stub),
        patch.object(cli_module, "detect_hardware", return_value=hw),
        patch.object(cli_module, "recommend_models", return_value={}),
        patch.object(cli_module, "webbrowser") as mock_webbrowser,
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        result = CliRunner().invoke(cli_module.app, ["setup"], input="anthropic\n1\n1\n")  # provider, auth=api key, model 1

    assert result.exit_code == 0
    mock_webbrowser.open.assert_not_called()
