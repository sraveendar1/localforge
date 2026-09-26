"""Dispatcher.resolve()/_run() wiring for an "advanced" per-modality
delegate target (delegate_target.py): keep automatic (today's behavior),
pin a specific local model, or route to a paid cloud model instead --
either via API key or a CLI subscription login.
"""
from unittest.mock import MagicMock

import pytest

from localforge import delegate_target as dt
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import Dispatcher, DownloadDeclined

CATALOG = [
    ModelEntry(name="small-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=2, quality_tier=1),
    ModelEntry(name="big-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=20, quality_tier=3),
]


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=64, free_disk_gb=200, gpus=[])


@pytest.fixture(autouse=True)
def _clear_targets(monkeypatch):
    for var in dt.TARGET_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)


def test_auto_is_unaffected_by_this_feature():
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"big-coder"})
    assert d.resolve("coding").name == "big-coder"  # best_match's usual pick


def test_a_pinned_local_model_wins_even_if_a_higher_tier_would_fit(monkeypatch):
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "ollama:small-coder")
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"small-coder", "big-coder"})
    entry = d.resolve("coding")
    assert entry.name == "small-coder"
    assert entry.runtime == "ollama"


def test_a_stale_local_pin_falls_back_to_automatic(monkeypatch):
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "ollama:no-such-model")
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"big-coder"})
    assert d.resolve("coding").name == "big-coder"


def test_an_api_cloud_target_resolves_to_a_synthetic_entry(monkeypatch):
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "api:anthropic:claude-haiku-4-5-20251001")
    d = Dispatcher(_hw(), catalog=CATALOG, installed=set())
    entry = d.resolve("coding")
    assert entry.name == "claude-haiku-4-5-20251001"
    assert entry.runtime == "api"
    assert entry.provider == "anthropic"
    assert entry.modality == "coding"


def test_a_cli_cloud_target_resolves_to_a_synthetic_entry(monkeypatch):
    monkeypatch.setenv(dt.TARGET_ENV_VARS["docs"], "cli:anthropic:claude-haiku-4-5-20251001")
    d = Dispatcher(_hw(), catalog=CATALOG, installed=set())
    entry = d.resolve("docs")
    assert entry.runtime == "cli"
    assert entry.provider == "anthropic"


def test_resolve_is_still_cached_per_modality_per_run(monkeypatch):
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "ollama:small-coder")
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"small-coder"})
    first = d.resolve("coding")
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "ollama:big-coder")  # changed mid-run: shouldn't matter
    assert d.resolve("coding") is first


class _FakeCloudBackend:
    """Raises on any unexpected kwarg -- proves the dispatcher only ever
    passes `provider` to a backend that actually wants it."""

    def __init__(self, content="cloud output", tokens=99, cost_usd=0.05, notional_cost_usd=0.0):
        self.content = content
        self.tokens = tokens
        self.cost_usd = cost_usd
        self.notional_cost_usd = notional_cost_usd
        self.ensure_available_calls = []
        self.generate_calls = []

    def ensure_available(self, model_name, on_progress=None, provider=None):
        self.ensure_available_calls.append((model_name, provider))

    def generate(self, model_name, prompt, on_token=None, provider=None, **kwargs):
        self.generate_calls.append((model_name, prompt, provider, kwargs))
        return {
            "type": "text", "content": self.content, "tokens": self.tokens,
            "cost_usd": self.cost_usd, "notional_cost_usd": self.notional_cost_usd,
        }


class _StrictOllamaBackend:
    """Raises on a `provider` kwarg -- proves a local entry never gets one."""

    def ensure_available(self, model_name, on_progress=None):
        pass

    def generate(self, model_name, prompt, on_token=None, **kwargs):
        assert "provider" not in kwargs
        return {"type": "text", "content": "local output", "tokens": 10}


def test_run_accounts_cloud_cost_separately_from_local_tokens(monkeypatch):
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "api:anthropic:claude-haiku-4-5")
    d = Dispatcher(_hw(), catalog=CATALOG, installed=set())
    fake_backend = _FakeCloudBackend()
    monkeypatch.setitem(__import__("localforge.tools", fromlist=["BACKENDS"]).BACKENDS, "api", fake_backend)

    entry = d.resolve("coding")
    result = d._run("coding", entry, "write a function", on_delegate=None)

    assert result["content"] == "cloud output"
    assert d.local_tokens_generated == 0
    assert d.delegate_tokens_generated == 99
    assert d.delegate_cost_usd == 0.05
    assert d.delegate_notional_cost_usd == 0.0
    # provider reached both ensure_available and generate
    assert fake_backend.ensure_available_calls == [("claude-haiku-4-5", "anthropic")]
    assert fake_backend.generate_calls[0][2] == "anthropic"


def test_run_never_passes_provider_to_a_local_backend(monkeypatch):
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"big-coder"})
    monkeypatch.setitem(__import__("localforge.tools", fromlist=["BACKENDS"]).BACKENDS, "ollama", _StrictOllamaBackend())

    entry = d.resolve("coding")
    result = d._run("coding", entry, "write a function", on_delegate=None)

    assert result["content"] == "local output"
    assert d.local_tokens_generated == 10
    assert d.delegate_tokens_generated == 0
    assert d.delegate_cost_usd == 0.0


def test_a_cloud_target_never_triggers_the_download_approval_flow(monkeypatch):
    """A cloud entry's runtime is never "ollama", so _run()'s download-
    approval gate (entry.runtime == "ollama") must never fire for it --
    unlike an uninstalled *local* pin, which still should."""
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "api:anthropic:claude-haiku-4-5")
    d = Dispatcher(_hw(), catalog=CATALOG, installed=set())  # nothing installed
    fake_backend = _FakeCloudBackend()
    monkeypatch.setitem(__import__("localforge.tools", fromlist=["BACKENDS"]).BACKENDS, "api", fake_backend)

    approver = MagicMock(return_value=False)  # would decline any approval it's asked for
    d.workspace = MagicMock(approver=approver)

    entry = d.resolve("coding")
    d._run("coding", entry, "write a function", on_delegate=None)  # must not raise DownloadDeclined

    approver.assert_not_called()


def test_an_uninstalled_local_pin_still_asks_for_download_approval(monkeypatch):
    monkeypatch.setenv(dt.TARGET_ENV_VARS["coding"], "ollama:big-coder")
    d = Dispatcher(_hw(), catalog=CATALOG, installed=set())  # big-coder not installed
    monkeypatch.setitem(__import__("localforge.tools", fromlist=["BACKENDS"]).BACKENDS, "ollama", _StrictOllamaBackend())

    approver = MagicMock(return_value=False)
    d.workspace = MagicMock(approver=approver)

    entry = d.resolve("coding")
    with pytest.raises(DownloadDeclined):
        d._run("coding", entry, "write a function", on_delegate=None)

    approver.assert_called_once()
