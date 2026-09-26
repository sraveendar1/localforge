"""`localforge local-model` (and its /local-model REPL twin, dispatched
through the same Typer app -- see repl.py): show or override which model
actually does coding/docs/general work, independently of the orchestrator.
"""
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge import delegate_target as dt

runner = CliRunner()


def test_bare_shows_all_three_as_automatic():
    result = runner.invoke(cli_module.app, ["local-model"])
    assert result.exit_code == 0
    for modality in dt.MODALITIES:
        assert f"{modality}: " in result.output
    assert "auto" in result.output


def test_bare_with_a_modality_shows_just_that_one():
    result = runner.invoke(cli_module.app, ["local-model", "coding"])
    assert result.exit_code == 0
    assert "coding:" in result.output
    assert "docs:" not in result.output


def test_unknown_modality_is_rejected():
    result = runner.invoke(cli_module.app, ["local-model", "nonsense", "x"])
    assert result.exit_code == 1
    assert "Unknown task type" in result.output


def test_pins_a_real_catalog_model():
    from localforge.catalog import load_catalog

    a_coding_model = next(m.name for m in load_catalog() if m.modality == "coding")
    result = runner.invoke(cli_module.app, ["local-model", "coding", a_coding_model])
    assert result.exit_code == 0
    assert dt.get("coding") == dt.DelegateTarget(kind="ollama", model=a_coding_model)
    assert a_coding_model in result.output


def test_rejects_a_model_not_in_the_catalog_for_that_modality():
    result = runner.invoke(cli_module.app, ["local-model", "coding", "not-a-real-model"])
    assert result.exit_code == 1
    assert "isn't a coding model in the catalog" in result.output
    assert dt.get("coding") == dt.AUTO  # nothing was changed


def test_setting_auto_clears_a_previous_pin():
    from localforge.catalog import load_catalog

    a_coding_model = next(m.name for m in load_catalog() if m.modality == "coding")
    runner.invoke(cli_module.app, ["local-model", "coding", a_coding_model])
    result = runner.invoke(cli_module.app, ["local-model", "coding", "auto"])
    assert result.exit_code == 0
    assert dt.get("coding") == dt.AUTO


def test_a_cloud_api_target_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(cli_module.app, ["local-model", "coding", "api:anthropic:claude-haiku-4-5"])
    assert result.exit_code == 1
    assert "No API key set for anthropic" in result.output
    assert dt.get("coding") == dt.AUTO


def test_a_cloud_api_target_succeeds_with_a_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = runner.invoke(cli_module.app, ["local-model", "coding", "api:anthropic:claude-haiku-4-5"])
    assert result.exit_code == 0
    assert dt.get("coding") == dt.DelegateTarget(kind="api", provider="anthropic", model="claude-haiku-4-5")
    assert "costs money" in result.output


def test_a_cloud_cli_target_needs_the_cli_available(monkeypatch):
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda provider: False)
    monkeypatch.setattr(cli_module.cli_transport, "requirements_message", lambda provider: "install it first")
    result = runner.invoke(cli_module.app, ["local-model", "docs", "cli:anthropic:claude-haiku-4-5"])
    assert result.exit_code == 1
    assert "install it first" in result.output
    assert dt.get("docs") == dt.AUTO


def test_a_cloud_cli_target_succeeds_when_the_cli_is_available(monkeypatch):
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda provider: True)
    result = runner.invoke(cli_module.app, ["local-model", "docs", "cli:anthropic:claude-haiku-4-5"])
    assert result.exit_code == 0
    assert dt.get("docs") == dt.DelegateTarget(kind="cli", provider="anthropic", model="claude-haiku-4-5")


def test_an_unparseable_cloud_target_is_rejected():
    result = runner.invoke(cli_module.app, ["local-model", "coding", "api:missing-model-part"])
    assert result.exit_code == 1
    assert "Couldn't parse" in result.output
    assert dt.get("coding") == dt.AUTO


def test_local_model_is_registered_as_a_slash_command():
    from localforge.repl import slash_commands

    assert "/local-model" in dict(slash_commands())
