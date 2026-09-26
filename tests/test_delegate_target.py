import pytest

from localforge import delegate_target as dt


def test_unset_and_explicit_auto_both_parse_to_auto():
    assert dt.parse(None) == dt.AUTO
    assert dt.parse("") == dt.AUTO
    assert dt.parse("auto") == dt.AUTO


def test_ollama_target_round_trips():
    target = dt.parse("ollama:qwen2.5-coder:14b")
    assert target == dt.DelegateTarget(kind="ollama", model="qwen2.5-coder:14b")
    assert target.is_cloud is False
    assert dt.render(target) == "ollama:qwen2.5-coder:14b"


def test_api_and_cli_cloud_targets_round_trip():
    api = dt.parse("api:anthropic:claude-haiku-4-5-20251001")
    assert api == dt.DelegateTarget(kind="api", provider="anthropic", model="claude-haiku-4-5-20251001")
    assert api.is_cloud is True
    assert dt.render(api) == "api:anthropic:claude-haiku-4-5-20251001"

    cli = dt.parse("cli:anthropic:claude-haiku-4-5-20251001")
    assert cli == dt.DelegateTarget(kind="cli", provider="anthropic", model="claude-haiku-4-5-20251001")
    assert cli.is_cloud is True


def test_garbage_degrades_to_auto_instead_of_raising():
    assert dt.parse("ollama:") == dt.AUTO  # no model name
    assert dt.parse("api:anthropic") == dt.AUTO  # missing model
    assert dt.parse("api::model") == dt.AUTO  # missing provider
    assert dt.parse("nonsense") == dt.AUTO
    assert dt.parse("somekey:with:extra:colons") == dt.AUTO  # unknown kind


def test_set_and_get_round_trip_through_config(monkeypatch, tmp_path):
    monkeypatch.setattr("localforge.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("localforge.config.CONFIG_FILE", tmp_path / "config.env")
    monkeypatch.delenv(dt.TARGET_ENV_VARS["coding"], raising=False)

    assert dt.get("coding") == dt.AUTO

    target = dt.DelegateTarget(kind="api", provider="gemini", model="gemini/gemini-2.5-flash")
    dt.set_target("coding", target)
    assert dt.get("coding") == target

    dt.clear("coding")
    assert dt.get("coding") == dt.AUTO


def test_get_all_covers_every_modality(monkeypatch, tmp_path):
    monkeypatch.setattr("localforge.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("localforge.config.CONFIG_FILE", tmp_path / "config.env")
    for modality in dt.MODALITIES:
        monkeypatch.delenv(dt.TARGET_ENV_VARS[modality], raising=False)

    assert dt.get_all() == {m: dt.AUTO for m in dt.MODALITIES}


def test_describe_reads_naturally():
    assert "auto" in dt.describe(dt.AUTO)
    assert dt.describe(dt.parse("ollama:qwen2.5-coder:14b")) == "qwen2.5-coder:14b (local, via Ollama)"
    assert dt.describe(dt.parse("api:anthropic:claude-haiku-4-5")) == "claude-haiku-4-5 (cloud, anthropic, via your API key)"
    assert dt.describe(dt.parse("cli:anthropic:claude-haiku-4-5")) == "claude-haiku-4-5 (cloud, anthropic, via your CLI login)"


def test_apply_rejects_unknown_modality():
    with pytest.raises(dt.InvalidTarget, match="Unknown task type"):
        dt.apply("nonsense", "auto")


def test_apply_auto_clears_a_pin(monkeypatch, tmp_path):
    monkeypatch.setattr("localforge.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("localforge.config.CONFIG_FILE", tmp_path / "config.env")
    dt.set_target("coding", dt.DelegateTarget(kind="ollama", model="x"))
    result = dt.apply("coding", "auto")
    assert result == dt.AUTO
    assert dt.get("coding") == dt.AUTO


def test_apply_rejects_a_local_model_not_in_the_catalog():
    with pytest.raises(dt.InvalidTarget, match="isn't a coding model in the catalog"):
        dt.apply("coding", "not-a-real-model")


def test_apply_pins_a_real_catalog_model(monkeypatch, tmp_path):
    monkeypatch.setattr("localforge.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("localforge.config.CONFIG_FILE", tmp_path / "config.env")
    from localforge.catalog import load_catalog

    a_coding_model = next(m.name for m in load_catalog() if m.modality == "coding")
    result = dt.apply("coding", a_coding_model)
    assert result == dt.DelegateTarget(kind="ollama", model=a_coding_model)
    assert dt.get("coding") == result


def test_apply_rejects_an_unparseable_cloud_target():
    with pytest.raises(dt.InvalidTarget, match="Couldn't parse"):
        dt.apply("coding", "api:missing-model-part")


def test_apply_rejects_api_target_with_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(dt.InvalidTarget, match="No API key set for anthropic"):
        dt.apply("coding", "api:anthropic:claude-haiku-4-5")


def test_apply_accepts_api_target_with_a_key(monkeypatch, tmp_path):
    monkeypatch.setattr("localforge.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("localforge.config.CONFIG_FILE", tmp_path / "config.env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = dt.apply("coding", "api:anthropic:claude-haiku-4-5")
    assert result == dt.DelegateTarget(kind="api", provider="anthropic", model="claude-haiku-4-5")


def test_apply_rejects_cli_target_when_unavailable(monkeypatch):
    from localforge import cli_transport

    monkeypatch.setattr(cli_transport, "available", lambda provider: False)
    monkeypatch.setattr(cli_transport, "requirements_message", lambda provider: "install it")
    with pytest.raises(dt.InvalidTarget, match="install it"):
        dt.apply("docs", "cli:anthropic:claude-haiku-4-5")


def test_apply_accepts_cli_target_when_available(monkeypatch, tmp_path):
    from localforge import cli_transport

    monkeypatch.setattr("localforge.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("localforge.config.CONFIG_FILE", tmp_path / "config.env")
    monkeypatch.setattr(cli_transport, "available", lambda provider: True)
    result = dt.apply("docs", "cli:anthropic:claude-haiku-4-5")
    assert result == dt.DelegateTarget(kind="cli", provider="anthropic", model="claude-haiku-4-5")
