from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile

CATALOG_PICK = {
    "coding": ModelEntry(
        name="qwen2.5-coder:14b", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=2, quality_tier=2
    ),
    "docs": None,
    "general": None,
    "image": None,
    "video": None,
}


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

    stub = StubOllama()
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])

    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module, "OllamaBackend", lambda *a, **k: stub),
        patch.object(cli_module, "detect_hardware", return_value=hw),
        patch.object(cli_module, "recommend_models", return_value=CATALOG_PICK),
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        yield stub


def test_setup_with_local_provider_needs_no_api_key_and_pulls_orchestrator_model(isolated_setup_env):
    result = CliRunner().invoke(
        cli_module.app,
        ["setup"],
        input="local\n2\n",  # provider=local, pick the 2nd curated choice (ollama/qwen2.5:72b)
    )
    assert result.exit_code == 0
    normalized_output = " ".join(result.output.split())
    assert "no API key needed" in normalized_output

    # the orchestrator model itself gets pulled, with the "ollama/" prefix stripped
    assert isolated_setup_env.pulled[0] == "qwen2.5:72b"

    saved = cli_module.config.CONFIG_FILE.read_text()
    assert "LOCALFORGE_FRONTIER_MODEL=ollama/qwen2.5:72b" in saved
    assert "ANTHROPIC_API_KEY" not in saved
    assert "OPENAI_API_KEY" not in saved


def test_doctor_reports_configured_for_local_frontier_without_any_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LOCALFORGE_FRONTIER_MODEL", "ollama/qwen2.5:72b")

    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])
    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module.OllamaBackend, "is_running", return_value=True),
        patch.object(cli_module, "detect_hardware", return_value=hw),
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        result = CliRunner().invoke(cli_module.app, ["doctor"])

    assert "Frontier model configured: ollama/qwen2.5:72b" in result.output
    assert "self-hosted, no API key needed" in result.output
