"""Paid image generation is billed per image, and a subscription (say Google AI
Pro) doesn't cover API use by itself. So: a monthly limit that stops generation
before anything is billed, a confirmation that says what it costs, and a
running total the user can see.
"""
import io
import json
import os
from datetime import datetime

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.tools as tools
from localforge import config, spend, trust
from localforge import delegate_target as dt
from localforge.backends import cloud
from localforge.hardware import HardwareProfile
from localforge.serve import StdioServer
from localforge.tools import Dispatcher
from localforge.workspace import Workspace

runner = CliRunner()
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 30


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


class FakeImages:
    def __init__(self, cost=0.0):
        self.cost, self.calls = cost, 0

    def generate(self, model, prompt, provider=None):
        self.calls += 1
        return cloud.GeneratedImage(data=PNG, mime_type="image/png", cost_usd=self.cost)


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "k" * 35)
    dt.apply("image", "api:gemini:gemini/gemini-2.5-flash-image")  # $0.039 an image in LiteLLM's price list
    fake = FakeImages(cost=0.0)  # LiteLLM reports no cost: the known price must count
    monkeypatch.setattr(tools, "_image_backend", fake)
    return fake


def _dispatcher(tmp_path, approve=True):
    asked = []
    ws = Workspace(tmp_path, approver=lambda kind, title, detail: asked.append((kind, title, detail)) or (approve(kind) if callable(approve) else approve))
    return Dispatcher(_hw(), installed=set(), workspace=ws), asked


def _gen(d, name="a.png"):
    return d.dispatch("generate_image", {"instructions": "a cat", "path": name})


# --- the ledger -------------------------------------------------------------------


def test_the_ledger_counts_by_provider_and_month():
    spend.record("gemini", 0.039, "m")
    spend.record("gemini", 0.045, "m")
    spend.record("openai", 0.1, "m")
    assert spend.spent("gemini") == pytest.approx(0.084) and spend.images("gemini") == 2 and spend.spent("openai") == pytest.approx(0.1)
    assert spend.spent("gemini", "1999-01") == 0.0  # a different month starts from zero


def test_a_new_month_starts_clean(monkeypatch):
    spend.record("gemini", 5.0)
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2099, 1, 5)
    monkeypatch.setattr(spend, "datetime", Later)
    assert spend.spent("gemini") == 0.0 and spend.month_key() == "2099-01"


def test_a_corrupt_ledger_is_read_as_empty_not_fatal():
    (config.CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    (config.CONFIG_DIR / "spend.json").write_text("{nope")
    assert spend.spent("gemini") == 0.0
    spend.record("gemini", 1.0)  # and it can be written over
    assert spend.spent("gemini") == 1.0


def test_no_limit_means_nothing_is_refused():
    spend.record("gemini", 1000.0)
    assert spend.check("gemini", 0.5) is None


def test_the_limit_stops_the_call_that_would_pass_it():
    spend.set_budget("gemini", 0.10)
    spend.record("gemini", 0.07)
    assert "would go over" in spend.check("gemini", 0.045)
    assert spend.check("gemini", 0.02) is None  # this one still fits


def test_a_reached_limit_stops_even_when_the_price_is_unknown():
    spend.set_budget("openai", 1.0)
    spend.record("openai", 1.0)
    assert "limit is reached" in spend.check("openai", None)


def test_the_limit_can_be_removed_and_is_saved_with_the_settings():
    spend.set_budget("gemini", 10)
    assert "LOCALFORGE_BUDGET_USD_GEMINI=10" in config.CONFIG_FILE.read_text() and spend.budget("gemini") == 10.0
    spend.set_budget("gemini", None)
    assert spend.budget("gemini") is None


@pytest.mark.parametrize("raw", ["", "abc", "-5", "0"])
def test_a_nonsense_limit_means_no_limit(monkeypatch, raw):
    monkeypatch.setenv("LOCALFORGE_BUDGET_USD_GEMINI", raw)
    assert spend.budget("gemini") is None


# --- the guard in front of a paid image -------------------------------------------


def test_it_asks_first_saying_what_it_costs_and_the_month_so_far(tmp_path, gemini):
    spend.set_budget("gemini", 10)
    spend.record("gemini", 2.10)
    d, asked = _dispatcher(tmp_path)
    _gen(d)
    kind, title, detail = asked[0]
    assert kind == "spend" and "gemini-2.5-flash-image" in title
    assert "about $0.039" in detail and "charged to your gemini API key" in detail
    assert "subscription doesn't cover API use" in detail
    assert "$2.10 of your $10.00 limit" in detail


def test_declining_the_cost_generates_nothing_and_bills_nothing(tmp_path, gemini):
    d, asked = _dispatcher(tmp_path, approve=lambda kind: kind != "spend")
    out = _gen(d)
    assert "declined generating an image" in out and "nothing was generated or billed" in out
    assert gemini.calls == 0 and spend.spent("gemini") == 0.0 and not (tmp_path / "a.png").exists()


def test_a_generated_image_is_counted_at_the_known_price_even_when_litellm_reports_none(tmp_path, gemini):
    d, _ = _dispatcher(tmp_path)
    out = _gen(d)
    assert gemini.calls == 1 and spend.spent("gemini") == pytest.approx(0.039) and spend.images("gemini") == 1
    assert d.delegate_cost_usd == pytest.approx(0.039) and "Cost: $0.039" in out


def test_litellms_own_figure_wins_when_it_is_higher(tmp_path, monkeypatch, gemini):
    monkeypatch.setattr(tools, "_image_backend", FakeImages(cost=0.2))
    d, _ = _dispatcher(tmp_path)
    _gen(d)
    assert spend.spent("gemini") == pytest.approx(0.2)


def test_past_the_limit_nothing_is_asked_generated_or_billed(tmp_path, gemini):
    spend.set_budget("gemini", 0.05)
    spend.record("gemini", 0.03)  # 0.03 + 0.039 > 0.05
    d, asked = _dispatcher(tmp_path)
    out = _gen(d)
    assert "would go over this month's gemini limit" in out and "Nothing was generated or billed" in out and "localforge budget" in out
    assert asked == [] and gemini.calls == 0 and spend.spent("gemini") == pytest.approx(0.03)


def test_it_stops_exactly_where_the_limit_says(tmp_path, gemini):
    spend.set_budget("gemini", 0.10)  # room for two images at $0.039, not three
    d, _ = _dispatcher(tmp_path)
    results = [_gen(d, f"{i}.png") for i in range(3)]
    assert gemini.calls == 2 and "Created" in results[0] and "Created" in results[1] and "would go over" in results[2]
    assert spend.spent("gemini") == pytest.approx(0.078)


def test_a_declined_save_is_still_billed_and_still_counted(tmp_path, gemini):
    """The image was generated and paid for before the user was asked to save it."""
    d, _ = _dispatcher(tmp_path, approve=lambda kind: kind == "spend")
    out = _gen(d)
    assert "declined" in out and "billed" in out and spend.spent("gemini") == pytest.approx(0.039)


def test_a_login_based_image_is_not_billed_or_asked_about(tmp_path, monkeypatch):
    monkeypatch.setattr("localforge.cli_transport.available", lambda p: p == "openai")
    dt.apply("image", "cli:openai:codex-image")

    class Cli:
        def generate(self, model, prompt, provider=None):
            return cloud.GeneratedImage(data=PNG, mime_type="image/png", cost_usd=0.0)

    monkeypatch.setattr(tools, "_image_cli_backend", Cli())
    d, asked = _dispatcher(tmp_path)
    _gen(d)
    assert [k for k, _t, _d in asked] == ["write"] and spend.spent("openai") == 0.0


def test_a_failed_provider_call_is_not_counted(tmp_path, monkeypatch, gemini):
    class Boom:
        def generate(self, *a, **k):
            raise RuntimeError("content policy")

    monkeypatch.setattr(tools, "_image_backend", Boom())
    d, _ = _dispatcher(tmp_path)
    with pytest.raises(RuntimeError):
        _gen(d)
    assert spend.spent("gemini") == 0.0


# --- the terminal -----------------------------------------------------------------


def test_budget_shows_spending_and_the_credit_note():
    spend.record("gemini", 0.5)
    out = runner.invoke(cli_module.app, ["budget"]).output
    assert "gemini: $0.50 (no limit set) this month, 1 image" in out and "openai: $0.00" in out
    assert "$10 a month in Google Cloud credits" in out and "no free tier" in " ".join(out.split())


def test_budget_sets_and_clears_a_limit():
    ok = runner.invoke(cli_module.app, ["budget", "gemini", "10"])
    assert ok.exit_code == 0 and "limited to $10 a month" in ok.output and spend.budget("gemini") == 10.0
    assert "of $10.00" in runner.invoke(cli_module.app, ["budget", "gemini"]).output
    off = runner.invoke(cli_module.app, ["budget", "gemini", "off"])
    assert off.exit_code == 0 and spend.budget("gemini") is None


def test_budget_refuses_nonsense():
    assert runner.invoke(cli_module.app, ["budget", "gemini", "lots"]).exit_code == 1
    assert runner.invoke(cli_module.app, ["budget", "gemini", "--", "-3"]).exit_code == 1
    assert runner.invoke(cli_module.app, ["budget", "claude", "5"]).exit_code == 1
    assert spend.budget("gemini") is None


def test_the_slash_help_lists_budget():
    from localforge import repl

    assert "/budget" in [name for name, _description in repl.slash_commands()]


# --- the desktop ------------------------------------------------------------------


def _server(tmp_path):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    return StdioServer(tmp_path, "m", out=out, run_fn=lambda *a, **k: None, scratch_root=tmp_path / "scratch", conversation=object()), out


def _last(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind][-1]


def test_desktop_reports_spending_and_limits(tmp_path):
    spend.set_budget("gemini", 10)
    spend.record("gemini", 1.25)
    server, out = _server(tmp_path)
    server.handle({"type": "budget_request"})
    rows = {r["provider"]: r for r in _last(out, "budget")["providers"]}
    assert rows["gemini"]["spent"] == 1.25 and rows["gemini"]["limit"] == 10.0 and rows["gemini"]["images"] == 1
    assert rows["openai"]["limit"] is None


def test_desktop_sets_and_clears_a_limit(tmp_path):
    server, out = _server(tmp_path)
    server.handle({"type": "set_budget", "provider": "gemini", "usd": 10})
    assert spend.budget("gemini") == 10.0 and {r["provider"]: r for r in _last(out, "budget")["providers"]}["gemini"]["limit"] == 10.0
    server.handle({"type": "set_budget", "provider": "gemini", "usd": None})
    assert spend.budget("gemini") is None


@pytest.mark.parametrize("bad", [-1, 0, "lots", 10**9, True])
def test_desktop_refuses_a_nonsense_limit(tmp_path, bad):
    server, out = _server(tmp_path)
    server.handle({"type": "set_budget", "provider": "gemini", "usd": bad})
    assert spend.budget("gemini") is None and "isn't an amount" in _last(out, "error")["message"]


def test_desktop_refuses_an_unknown_provider(tmp_path):
    server, out = _server(tmp_path)
    server.handle({"type": "set_budget", "provider": "anthropic", "usd": 5})
    assert _last(out, "error")["message"].startswith("Unknown provider")


def test_desktop_chat_budget_command_shows_and_sets(tmp_path):
    spend.record("gemini", 0.5)
    server, out = _server(tmp_path)
    server.handle({"type": "user_message", "text": "/budget"})
    assert "gemini: $0.50 (no limit set) this month, 1 image" in _last(out, "system_text")["text"]
    server.handle({"type": "user_message", "text": "/budget gemini 10"})
    assert spend.budget("gemini") == 10.0 and "of $10.00" in _last(out, "system_text")["text"]
    server.handle({"type": "user_message", "text": "/budget gemini off"})
    assert spend.budget("gemini") is None
    server.handle({"type": "user_message", "text": "/budget gemini lots"})
    assert "isn't an amount" in _last(out, "error")["message"]
