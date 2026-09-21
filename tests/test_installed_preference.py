"""Setup must reuse suitable already-installed models instead of downloading.

Regression tests for a reported bug: re-running setup downloaded
`llama3.1:8b` (4.9 GB, a tier-1 docs model) while `mistral-nemo:12b` (a
tier-2 docs model) was already installed. The recommender only ever looked
at the static catalog and never checked what was on disk, so the frontier
model advisor -- with no idea mistral-nemo was installed -- picked the
smaller model for "headroom" and triggered a pointless download.
"""

from unittest.mock import MagicMock, patch

from localforge.advisor import recommend_models
from localforge.catalog import ModelEntry, best_match, recommendations
from localforge.hardware import HardwareProfile


def _entry(name, modality, tier, ram=8, vram=0, disk=2):
    return ModelEntry(
        name=name, modality=modality, runtime="ollama", min_vram_gb=vram, min_ram_gb=ram, disk_gb=disk, quality_tier=tier
    )


# Mirrors the user's real machine and catalog at the time of the report.
CATALOG = [
    _entry("qwen2.5-coder:14b", "coding", 2, ram=16, vram=10, disk=9),
    _entry("qwen2.5-coder:7b", "coding", 1, ram=8, vram=6, disk=4.5),
    _entry("qwen2.5-coder:32b", "coding", 3, ram=32, vram=20, disk=20),
    _entry("llama3.1:8b", "docs", 1, ram=8, vram=6, disk=4.7),
    _entry("mistral-nemo:12b", "docs", 2, ram=12, vram=8, disk=7),
    _entry("qwen2.5:3b", "general", 1, ram=4, disk=1.9),
]
HW = HardwareProfile(os="Darwin", arch="arm64", cpu_cores=10, ram_gb=16, free_disk_gb=100, gpus=[])
HW_WITH_GPU = HardwareProfile(
    os="Darwin",
    arch="arm64",
    cpu_cores=10,
    ram_gb=16,
    free_disk_gb=100,
    gpus=[{"name": "Apple M4", "vram_gb": 12, "backend": "metal"}],
)


# --- deterministic path (catalog.best_match / recommendations) ---


def test_prefers_installed_model_over_higher_tier_download():
    installed = {"qwen2.5-coder:7b"}
    pick = best_match("coding", HW_WITH_GPU, CATALOG, installed=installed)
    assert pick.name == "qwen2.5-coder:7b"  # reused, not the 9 GB 14b


def test_without_installed_info_behaviour_is_unchanged():
    """Callers that don't pass `installed` get exactly the old result."""
    assert best_match("coding", HW_WITH_GPU, CATALOG).name == "qwen2.5-coder:14b"


def test_falls_back_to_best_tier_when_nothing_installed_fits():
    installed = {"some-unrelated-model:latest"}
    assert best_match("coding", HW_WITH_GPU, CATALOG, installed=installed).name == "qwen2.5-coder:14b"


def test_installed_model_that_does_not_fit_hardware_is_not_preferred():
    """Being on disk isn't enough -- it still has to run on this machine.
    32b needs 32 GB RAM; this machine has 16.
    """
    installed = {"qwen2.5-coder:32b"}
    pick = best_match("coding", HW_WITH_GPU, CATALOG, installed=installed)
    assert pick.name != "qwen2.5-coder:32b"
    assert pick.name == "qwen2.5-coder:14b"


def test_among_several_installed_picks_the_highest_tier():
    """The exact reported scenario: both docs models installed, tier-2 wins."""
    installed = {"llama3.1:8b", "mistral-nemo:12b"}
    assert best_match("docs", HW_WITH_GPU, CATALOG, installed=installed).name == "mistral-nemo:12b"


def test_recommendations_reuses_everything_on_the_reporters_machine():
    installed = {"llama3.1:8b", "mistral-nemo:12b", "qwen2.5-coder:7b", "qwen2.5:3b"}
    recs = recommendations(HW_WITH_GPU, CATALOG, installed=installed)
    assert {m: e.name for m, e in recs.items()} == {
        "coding": "qwen2.5-coder:7b",
        "docs": "mistral-nemo:12b",
        "general": "qwen2.5:3b",
    }
    assert all(e.name in installed for e in recs.values())  # zero downloads


# --- frontier-model advisor path ---


def test_advisor_tells_the_frontier_model_which_candidates_are_installed():
    """The advisor was the actual source of the reported download: it had no
    way to know mistral-nemo was installed. It must now be told.
    """
    captured = {}

    def fake_completion(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        raise RuntimeError("stop after capturing the prompt")

    installed = {"mistral-nemo:12b"}
    with patch("localforge.advisor.completion", side_effect=fake_completion):
        recommend_models(HW_WITH_GPU, "claude-opus-5", catalog=CATALOG, installed=installed)

    prompt = captured["prompt"]
    assert '"installed": true' in prompt
    assert "no download" in prompt.lower()
    # the installed flag is attached to the right model
    nemo_segment = prompt[prompt.index('"mistral-nemo:12b"') :][:200]
    assert '"installed": true' in nemo_segment


def test_advisor_fallback_prefers_installed_models():
    """If the frontier call fails, the deterministic fallback must also reuse
    rather than download.
    """
    installed = {"qwen2.5-coder:7b", "mistral-nemo:12b", "qwen2.5:3b"}
    with patch("localforge.advisor.completion", side_effect=RuntimeError("no network")):
        recs = recommend_models(HW_WITH_GPU, "claude-opus-5", catalog=CATALOG, installed=installed)
    assert recs["coding"].name == "qwen2.5-coder:7b"
    assert recs["docs"].name == "mistral-nemo:12b"


def test_advisor_can_still_choose_an_upgrade_when_it_judges_it_worthwhile():
    """Installed is a strong preference, not a lock -- if the frontier model
    deliberately picks a better non-installed model, that choice stands.
    """
    response = MagicMock()
    response.choices[0].message.tool_calls[0].function.arguments = '{"coding": "qwen2.5-coder:14b"}'
    with patch("localforge.advisor.completion", return_value=response):
        recs = recommend_models(HW_WITH_GPU, "claude-opus-5", catalog=CATALOG, installed={"qwen2.5-coder:7b"})
    assert recs["coding"].name == "qwen2.5-coder:14b"


# --- setup's actual download decision ---


def test_setup_does_not_pull_models_that_are_already_installed(tmp_path, monkeypatch):
    """End to end through `localforge setup`: nothing already on disk may be
    passed to ensure_available (i.e. re-downloaded/re-checked for pulling).
    """
    from typer.testing import CliRunner

    import localforge.cli as cli_module

    monkeypatch.setenv("LOCALFORGE_CONFIG_DIR", str(tmp_path / "config"))
    cli_module.config.CONFIG_DIR = tmp_path / "config"
    cli_module.config.CONFIG_FILE = cli_module.config.CONFIG_DIR / "config.env"

    installed = [{"name": n} for n in ("qwen2.5-coder:7b", "mistral-nemo:12b", "qwen2.5:3b")]

    class StubOllama:
        pulled: list[str] = []

        def __init__(self, *a, **k):
            pass

        def is_running(self):
            return True

        def list_installed(self):
            return installed

        def ensure_available(self, name, on_progress=None):
            StubOllama.pulled.append(name)

    StubOllama.pulled = []
    recs = {
        "coding": CATALOG[1],  # qwen2.5-coder:7b -- installed
        "docs": CATALOG[4],  # mistral-nemo:12b -- installed
        "general": CATALOG[5],  # qwen2.5:3b -- installed
    }
    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module, "OllamaBackend", StubOllama),
        patch.object(cli_module, "detect_hardware", return_value=HW_WITH_GPU),
        patch.object(cli_module, "recommend_models", return_value=recs),
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        # provider 4 (local) -> model 1; local orchestrator pull is separate
        result = CliRunner().invoke(cli_module.app, ["setup"], input="4\n1\n")

    assert result.exit_code == 0
    delegate_pulls = [n for n in StubOllama.pulled if n in {e.name for e in recs.values()}]
    assert delegate_pulls == [], f"re-downloaded installed models: {delegate_pulls}"
    normalized = " ".join(result.output.split())
    assert "Nothing to download" in normalized


def test_an_installed_model_is_not_rejected_for_lack_of_free_disk():
    """Reported machine after a surprise 9 GB pull: 16 GB RAM, 12 GB VRAM,
    5.8 GB free. The installed 14b needs no more disk, so it must still fit;
    a model that would have to be downloaded still needs the space.
    """
    from localforge.catalog import best_match, candidates, load_catalog
    from localforge.hardware import GPU, HardwareProfile

    hw = HardwareProfile(
        os="Darwin", arch="arm64", cpu_cores=10, ram_gb=16, free_disk_gb=5.8,
        gpus=[GPU(name="Apple M4", vram_gb=12, backend="metal")],
    )
    catalog = load_catalog()
    installed = {"qwen2.5-coder:14b", "qwen2.5-coder:7b"}

    assert best_match("coding", hw, catalog, installed=installed).name == "qwen2.5-coder:14b"
    # without the installed info the 9 GB download genuinely doesn't fit
    assert "qwen2.5-coder:14b" not in {m.name for m in candidates("coding", hw, catalog)}
