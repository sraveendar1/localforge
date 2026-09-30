"""`localforge setup` asks, right after the orchestrator is chosen, whether
the local/cloud models that coding/docs/general work is delegated to should
be picked automatically or per task type ("Advanced"). Also: /models and
/advanced-model name the concrete model that "auto" currently resolves to,
instead of just the word "auto" (reported: "this is not intuitive").
"""
import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge import delegate_target as dt
from localforge.catalog import ModelEntry, load_catalog
from localforge.hardware import HardwareProfile

runner = CliRunner()


def _recs():
    catalog = load_catalog()
    return {
        m: next(e for e in catalog if e.modality == m)
        for m in dt.MODALITIES
    }


def _scripted(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(cli_module.console, "input", lambda prompt="": next(it))


# --- _walk_through_advanced_model_setup ---------------------------------------


def test_keeping_every_recommendation_changes_nothing(monkeypatch):
    recs = _recs()
    before = dict(recs)
    _scripted(monkeypatch, ["", "1", ""])  # blank = default "Keep this" for all three
    cli_module._walk_through_advanced_model_setup(recs, installed=set())
    assert recs == before
    assert dt.get_all() == {m: dt.AUTO for m in dt.MODALITIES}


def test_picking_a_different_local_model_replaces_the_download_entry(monkeypatch):
    recs = _recs()
    catalog = load_catalog()
    ranked = sorted((m for m in catalog if m.modality == "coding"), key=lambda m: -m.quality_tier)
    # coding: option 2 (pick local), then the 1st listed entry; docs/general: keep.
    _scripted(monkeypatch, ["2", "1", "", ""])
    cli_module._walk_through_advanced_model_setup(recs, installed=set())
    assert recs["coding"].name == ranked[0].name  # so setup downloads THIS one, not the old pick


def test_picking_a_cloud_model_clears_the_download_entry_and_pins_it(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    recs = _recs()
    # coding: 3 (cloud), (a)pi, provider, model; docs/general: keep.
    _scripted(monkeypatch, ["3", "a", "anthropic", "claude-haiku-4-5", "", ""])
    cli_module._walk_through_advanced_model_setup(recs, installed=set())
    assert recs["coding"] is None  # nothing to download for a cloud target
    assert dt.get("coding") == dt.DelegateTarget(kind="api", provider="anthropic", model="claude-haiku-4-5")


def test_a_cloud_pick_uses_the_providers_default_model_when_none_is_typed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from localforge import config

    recs = _recs()
    _scripted(monkeypatch, ["3", "a", "anthropic", "", "", ""])  # blank model -> default
    cli_module._walk_through_advanced_model_setup(recs, installed=set())
    assert dt.get("coding").model == config.FRONTIER_MODEL_CHOICES["anthropic"][0]


def test_an_unusable_cloud_pick_stays_automatic_instead_of_saving_it(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    recs = _recs()
    before = recs["coding"]
    _scripted(monkeypatch, ["3", "a", "anthropic", "claude-haiku-4-5", "", ""])
    cli_module._walk_through_advanced_model_setup(recs, installed=set())
    assert recs["coding"] is before  # untouched: the local recommendation still applies
    assert dt.get("coding") == dt.AUTO


def test_a_cli_login_cloud_pick_needs_the_cli(monkeypatch):
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda provider: True)
    recs = _recs()
    _scripted(monkeypatch, ["3", "c", "anthropic", "claude-haiku-4-5", "", ""])
    cli_module._walk_through_advanced_model_setup(recs, installed=set())
    assert dt.get("coding") == dt.DelegateTarget(kind="cli", provider="anthropic", model="claude-haiku-4-5")


def test_each_task_type_is_decided_independently(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    recs = _recs()
    docs_before = recs["docs"]
    # coding -> cloud; docs -> keep; general -> keep
    _scripted(monkeypatch, ["3", "a", "anthropic", "claude-haiku-4-5", "", ""])
    cli_module._walk_through_advanced_model_setup(recs, installed=set())
    assert recs["coding"] is None
    assert recs["docs"] is docs_before
    assert dt.get("docs") == dt.AUTO and dt.get("general") == dt.AUTO


# --- naming the concrete model instead of "auto" --------------------------------


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=64, free_disk_gb=200, gpus=[])


def _entry(name="qwen2.5-coder:7b"):
    return ModelEntry(name=name, modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=4.7, quality_tier=2)


def test_auto_is_described_with_the_concrete_model_and_its_install_state():
    recs = {"coding": _entry()}
    assert cli_module._describe_active_target("coding", recs, {"qwen2.5-coder:7b"}) == (
        "qwen2.5-coder:7b (auto, local, installed)"
    )
    assert "not installed" in cli_module._describe_active_target("coding", recs, set())


def test_auto_with_nothing_fitting_says_so():
    assert "no local model fits" in cli_module._describe_active_target("coding", {"coding": None}, set())


def test_a_pin_is_described_by_the_pin_not_the_recommendation(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    dt.apply("coding", "api:anthropic:claude-haiku-4-5")
    text = cli_module._describe_active_target("coding", {"coding": _entry()}, set())
    assert "claude-haiku-4-5" in text and "qwen2.5-coder" not in text


def test_bare_advanced_model_names_the_actual_model(monkeypatch):
    monkeypatch.setattr(cli_module, "detect_hardware", _hw)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda ollama: set())
    monkeypatch.setattr(cli_module, "recommendations", lambda hw, installed=None: {m: _entry() for m in dt.MODALITIES})
    result = runner.invoke(cli_module.app, ["advanced-model"])
    assert result.exit_code == 0
    assert "qwen2.5-coder:7b" in result.output
    assert "auto, local" in result.output


# --- /models shows each task type's actual role ---------------------------------


def _fake_models_env(monkeypatch, installed=()):
    monkeypatch.setattr(cli_module, "detect_hardware", _hw)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda ollama: set(installed))
    monkeypatch.setattr(cli_module, "recommendations", lambda hw, installed=None: {m: _entry() for m in dt.MODALITIES})


def test_models_shows_auto_role_by_default(monkeypatch):
    _fake_models_env(monkeypatch)
    result = runner.invoke(cli_module.app, ["models"])
    assert result.exit_code == 0
    assert "auto (local)" in result.output
    assert "pinned" not in result.output


def test_models_reflects_a_cloud_pin_instead_of_the_stale_recommendation(monkeypatch):
    _fake_models_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    dt.apply("coding", "api:anthropic:claude-haiku-4-5")
    result = runner.invoke(cli_module.app, ["models"])
    assert "claude-haiku-4-5" in result.output
    assert "pinned" in result.output and "cloud" in result.output


def test_models_reflects_a_local_pin(monkeypatch):
    _fake_models_env(monkeypatch, installed={"qwen2.5-coder:14b"})
    dt.apply("coding", "qwen2.5-coder:14b")
    result = runner.invoke(cli_module.app, ["models"])
    assert "qwen2.5-coder:14b" in result.output
    assert "pinned" in result.output and "local" in result.output


# --- the setup prompt itself ------------------------------------------------------


def test_setup_source_asks_auto_or_advanced_after_the_orchestrator_step():
    """Guards the ordering the user asked for: the question comes after the
    orchestrator is chosen (step 2) and before the hardware scan (step 3)."""
    import inspect

    src = inspect.getsource(cli_module.setup)
    ask = src.index("Automatic (recommended)")
    assert src.index("_prompt_for_model(provider)") < ask < src.index("detect_hardware()")
    assert "_walk_through_advanced_model_setup(recs, installed)" in src
