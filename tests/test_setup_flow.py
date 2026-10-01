"""First-run setup inside the desktop app: what's set up and what isn't, saving
an API key (checked with the provider first, never echoed back), using a
provider's CLI login, choosing the orchestrator -- so a new user never has to
open a terminal for `localforge setup`. Also the friendlier failure when a task
is sent with nothing set up, and Gemini's model choices.
"""
import io
import json
import os
import time

import httpx
import pytest

import localforge.cli as cli_module
import localforge.serve as serve_module
from localforge import config, project_models, provider_check, trust
from localforge import delegate_target as dt
from localforge.serve import StdioServer

KEY = "sk-ant-api03-" + "x" * 40
GKEY = "AIza" + "y" * 35


@pytest.fixture(autouse=True)
def bare_machine(monkeypatch):
    """A machine with nothing set up: no keys, no CLIs, no Ollama."""
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(serve_module.cli_transport, "available", lambda p: False)
    monkeypatch.setattr("localforge.cli_transport.available", lambda p: False)
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: False)
    monkeypatch.setattr(serve_module.OllamaBackend, "list_installed", lambda self: [])
    monkeypatch.setattr(serve_module.shutil, "which", lambda name: None)


def _server(tmp_path, model="claude-opus-5", cli_provider=None, run_fn=None):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    server = StdioServer(tmp_path, model, cli_provider, out=out, run_fn=run_fn or (lambda *a, **k: None),
                         scratch_root=tmp_path / "scratch", conversation=__import__("localforge.orchestrator", fromlist=["x"]).Conversation())
    return server, out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


def _last(out, kind):
    got = _events(out, kind)
    assert got, f"no {kind!r} event"
    return got[-1]


def _accepts(monkeypatch, status=provider_check.OK, detail="ok"):
    monkeypatch.setattr(provider_check, "check_key", lambda provider, key: (status, detail))


# --- what's set up ----------------------------------------------------------------


def test_a_bare_machine_needs_setup_and_says_why(tmp_path):
    server, out = _server(tmp_path)
    server._emit_setup_status()
    st = _last(out, "setup_status")
    assert st["needs_setup"] is True
    assert st["orchestrator"]["ready"] is False and "API key or login" in st["orchestrator"]["reason"]
    assert [p["id"] for p in st["providers"]] == ["anthropic", "openai", "gemini"]
    assert all(p["key_set"] is False for p in st["providers"])
    assert st["ollama"] == {"installed": False, "running": False, "models": [], "suggested": []}


def test_a_key_makes_the_orchestrator_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    server, out = _server(tmp_path)
    server._emit_setup_status()
    st = _last(out, "setup_status")
    assert st["needs_setup"] is False
    assert next(p for p in st["providers"] if p["id"] == "anthropic")["key_set"] is True


def test_an_installed_cli_alone_is_not_enough_until_its_login_is_chosen(tmp_path, monkeypatch):
    """`claude` on the PATH but the run going to the API: that's the case that used to fail."""
    monkeypatch.setattr(serve_module.cli_transport, "available", lambda p: p == "anthropic")
    server, out = _server(tmp_path)  # cli_provider None: the route is the API, and there's no key
    server._emit_setup_status()
    st = _last(out, "setup_status")
    assert st["needs_setup"] is True
    claude = next(p for p in st["providers"] if p["id"] == "anthropic")
    assert claude["cli"]["installed"] is True and claude["cli"]["command"] == "claude"


def test_a_chosen_login_that_is_installed_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_module.cli_transport, "available", lambda p: p == "anthropic")
    server, out = _server(tmp_path, cli_provider="anthropic")
    server._emit_setup_status()
    assert _last(out, "setup_status")["needs_setup"] is False


def test_a_local_orchestrator_that_isnt_downloaded_needs_setup(tmp_path):
    server, out = _server(tmp_path, model="ollama/gemma3:4b")
    server._emit_setup_status()
    st = _last(out, "setup_status")
    assert st["needs_setup"] is True and "gemma3:4b" in st["orchestrator"]["reason"]


def test_a_downloaded_local_orchestrator_is_ready_and_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: True)
    monkeypatch.setattr(serve_module.OllamaBackend, "list_installed", lambda self: [{"name": "gemma3:4b", "size": 3_300_000_000}])
    monkeypatch.setattr(serve_module.model_fit, "installed_sizes", lambda: {"gemma3:4b": 3.3})  # (the suite's fixture hides it)
    server, out = _server(tmp_path, model="ollama/gemma3:4b")
    server._emit_setup_status()
    st = _last(out, "setup_status")
    assert st["needs_setup"] is False
    assert st["ollama"]["running"] is True and [m["name"] for m in st["ollama"]["models"]] == ["gemma3:4b"]


def test_a_model_from_another_provider_we_cant_judge_doesnt_get_in_the_way(tmp_path):
    server, out = _server(tmp_path, model="mistral/mistral-large")
    server._emit_setup_status()
    assert _last(out, "setup_status")["needs_setup"] is False


# --- saving an API key ------------------------------------------------------------


def test_a_good_key_is_saved_privately_and_chosen_as_the_orchestrator(tmp_path, monkeypatch):
    _accepts(monkeypatch)
    server, out = _server(tmp_path)
    server._save_api_key("anthropic", f"  {KEY}\n")
    res = _last(out, "setup_result")
    assert res["ok"] is True and "Saved your Anthropic (Claude) key" in res["message"]
    saved = config.CONFIG_FILE.read_text()
    assert f"ANTHROPIC_API_KEY={KEY}" in saved and "LOCALFORGE_AUTH_METHOD=api_key" in saved and "LOCALFORGE_FRONTIER_PROVIDER=anthropic" in saved
    assert oct(config.CONFIG_FILE.stat().st_mode & 0o777) == "0o600"
    assert server.frontier_model == config.FRONTIER_MODEL_CHOICES["anthropic"][0] and server.cli_provider is None
    assert _last(out, "setup_status")["needs_setup"] is False
    assert _last(out, "settings")["model"] == server.frontier_model


def test_the_key_never_appears_in_anything_sent_to_the_app(tmp_path, monkeypatch):
    _accepts(monkeypatch)
    server, out = _server(tmp_path)
    server._save_api_key("anthropic", KEY)
    assert KEY not in out.getvalue()


def test_a_key_the_provider_rejects_is_not_saved(tmp_path, monkeypatch):
    _accepts(monkeypatch, provider_check.REJECTED, "anthropic rejected the key (HTTP 401)")
    server, out = _server(tmp_path)
    server._save_api_key("anthropic", KEY)
    res = _last(out, "setup_result")
    assert res["ok"] is False and "rejected that key" in res["message"] and KEY not in out.getvalue()
    assert not config.CONFIG_FILE.exists()
    assert _last(out, "setup_status")["needs_setup"] is True


@pytest.mark.parametrize("bad", ["", "short", "has a space in the middle of it and is long enough to pass"])
def test_obvious_mistakes_are_refused_before_anything_is_sent(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(provider_check, "check_key", lambda *a: pytest.fail("shouldn't ask the provider about this"))
    server, out = _server(tmp_path)
    server._save_api_key("anthropic", bad)
    assert _last(out, "setup_result")["ok"] is False and not config.CONFIG_FILE.exists()


def test_when_the_provider_cant_be_reached_the_key_is_kept_with_a_note(tmp_path, monkeypatch):
    _accepts(monkeypatch, provider_check.UNREACHABLE, "couldn't reach anthropic to check it (ConnectError)")
    server, out = _server(tmp_path)
    server._save_api_key("anthropic", KEY)
    res = _last(out, "setup_result")
    assert res["ok"] is True and "couldn't check it right now" in res["message"]
    assert f"ANTHROPIC_API_KEY={KEY}" in config.CONFIG_FILE.read_text()


def test_a_second_key_doesnt_take_over_a_working_orchestrator(tmp_path, monkeypatch):
    _accepts(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    server, out = _server(tmp_path, model="claude-sonnet-5")
    server._save_api_key("openai", "sk-" + "z" * 40)
    assert server.frontier_model == "claude-sonnet-5"
    assert "OPENAI_API_KEY=sk-" in config.CONFIG_FILE.read_text()
    assert "chose" not in _last(out, "setup_result")["message"]


def test_a_gemini_key_chooses_a_gemini_model(tmp_path, monkeypatch):
    _accepts(monkeypatch)
    monkeypatch.setattr(cli_module, "_gemini_models", lambda key: ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"))
    server, out = _server(tmp_path)
    server._save_api_key("gemini", GKEY)
    assert server.frontier_model == "gemini/gemini-2.5-pro"
    assert f"GEMINI_API_KEY={GKEY}" in config.CONFIG_FILE.read_text()


def test_an_unknown_provider_is_refused(tmp_path):
    server, out = _server(tmp_path)
    server._save_api_key("local", KEY)
    assert _last(out, "setup_result")["ok"] is False


def test_the_message_handler_runs_it_off_the_reader_thread(tmp_path, monkeypatch):
    _accepts(monkeypatch)
    server, out = _server(tmp_path)
    server.handle({"type": "save_api_key", "provider": "anthropic", "key": KEY})
    for _ in range(100):
        if _events(out, "setup_result"):
            break
        time.sleep(0.02)
    assert _last(out, "setup_result")["ok"] is True


def test_a_project_that_saved_its_own_models_gets_the_new_orchestrator_too(tmp_path, monkeypatch):
    """Otherwise its models file would put the old, unusable one straight back."""
    _accepts(monkeypatch)
    (tmp_path / ".localforge").mkdir()
    (tmp_path / ".localforge" / "models.json").write_text(json.dumps({"orchestrator": {"model": "claude-haiku-4-5", "auth_method": "api_key", "provider": "anthropic"}}))
    server, out = _server(tmp_path)
    assert server.frontier_model == "claude-haiku-4-5"
    monkeypatch.setattr(serve_module, "_orchestrator_readiness", lambda m, c: (False, "x") if m == "claude-haiku-4-5" else (True, ""))
    server._save_api_key("anthropic", KEY)
    assert project_models.load(tmp_path)["orchestrator"]["model"] == server.frontier_model != "claude-haiku-4-5"


# --- using a login ----------------------------------------------------------------


def test_a_login_that_answers_is_chosen(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_module.cli_transport, "available", lambda p: True)
    monkeypatch.setattr(serve_module.cli_transport, "probe", lambda p: (True, "ok"))
    server, out = _server(tmp_path)
    server._use_login("anthropic")
    assert _last(out, "setup_result")["ok"] is True
    assert server.cli_provider == "anthropic" and server.frontier_model == config.FRONTIER_MODEL_CHOICES["anthropic"][0]
    saved = config.CONFIG_FILE.read_text()
    assert "LOCALFORGE_AUTH_METHOD=cli_login" in saved and "LOCALFORGE_FRONTIER_PROVIDER=anthropic" in saved
    assert _last(out, "setup_status")["needs_setup"] is False


def test_a_login_whose_cli_isnt_installed_says_how_to_get_it(tmp_path):
    server, out = _server(tmp_path)
    server._use_login("openai")
    res = _last(out, "setup_result")
    assert res["ok"] is False and "codex" in res["message"] and not config.CONFIG_FILE.exists()


def test_a_signed_out_login_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_module.cli_transport, "available", lambda p: True)
    monkeypatch.setattr(serve_module.cli_transport, "probe", lambda p: (False, "Not logged in · Please run /login"))
    server, out = _server(tmp_path)
    server._use_login("anthropic")
    res = _last(out, "setup_result")
    assert res["ok"] is False and "claude login" in res["message"]
    assert not config.CONFIG_FILE.exists() and server.cli_provider is None


# --- choosing from what's available -----------------------------------------------


def test_choosing_a_downloaded_local_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: {"gemma3:4b"})
    server, out = _server(tmp_path)
    server._choose_orchestrator("ollama/gemma3:4b")
    assert _last(out, "setup_result")["ok"] is True and server.frontier_model == "ollama/gemma3:4b" and server.cli_provider is None
    assert "LOCALFORGE_AUTH_METHOD=local" in config.CONFIG_FILE.read_text()


def test_choosing_something_that_isnt_available_is_refused(tmp_path):
    server, out = _server(tmp_path)
    server._choose_orchestrator("claude-opus-5")
    assert _last(out, "setup_result")["ok"] is False and server.frontier_model == "claude-opus-5"
    assert not config.CONFIG_FILE.exists()


# --- the first task with nothing set up -------------------------------------------


def test_a_first_task_with_no_key_says_what_to_do_not_the_libraries_error(tmp_path):
    import litellm

    def run(text, model, **kw):
        raise litellm.AuthenticationError("Missing Anthropic API Key - A call is being made to anthropic but...", "anthropic", model)

    server, out = _server(tmp_path, run_fn=run)
    server._run_turn("hello")
    err = _last(out, "error")
    assert "needs a Anthropic (Claude) API key or login" in err["message"] and "Set up" in err["message"]
    assert "Missing Anthropic API Key" not in err["message"] and err["setup_needed"] is True
    assert "AuthenticationError" in err["detail"]  # the original is kept for anyone who wants it
    assert _last(out, "task_summary")["summary"]["error"] == err["message"]
    assert _last(out, "setup_status")["needs_setup"] is True  # so the app can show the setup screen


def test_other_failures_are_reported_as_before(tmp_path):
    def run(*a, **k):
        raise RuntimeError("model exploded")

    server, out = _server(tmp_path, run_fn=run)
    server._run_turn("hello")
    err = _last(out, "error")
    assert err["message"] == "RuntimeError: model exploded" and err["setup_needed"] is False


def test_an_auth_error_with_a_key_present_isnt_blamed_on_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)

    def run(*a, **k):
        raise RuntimeError("invalid x-api-key")  # the key exists but is wrong: not a "not set up" problem

    server, out = _server(tmp_path, run_fn=run)
    server._run_turn("hello")
    assert _last(out, "error")["setup_needed"] is False


# --- doctor -----------------------------------------------------------------------


def test_doctor_checks_each_key_with_its_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GKEY)
    monkeypatch.setattr(provider_check, "check_key", lambda p, k: (provider_check.REJECTED, "anthropic rejected the key (HTTP 401)") if p == "anthropic" else (provider_check.OK, "ok"))
    monkeypatch.setattr(provider_check, "gemini_image_models", lambda k: ["gemini/gemini-2.5-flash-image"])
    checks = {c["name"]: c for c in serve_module._doctor("claude-opus-5", None)}
    assert checks["Anthropic (Claude) API key"]["ok"] is False and "rejected" in checks["Anthropic (Claude) API key"]["detail"]
    assert checks["Google (Gemini) API key"]["ok"] is True
    assert checks["Gemini image models"]["ok"] is True and "gemini-2.5-flash-image" in checks["Gemini image models"]["detail"]
    assert checks["Orchestrator"]["ok"] is True and checks["Ollama installed"]["ok"] is False


def test_doctor_says_when_gemini_has_no_image_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GKEY)
    monkeypatch.setattr(provider_check, "check_key", lambda p, k: (provider_check.OK, "ok"))
    monkeypatch.setattr(provider_check, "gemini_image_models", lambda k: [])
    checks = {c["name"]: c for c in serve_module._doctor("gemini/gemini-2.5-pro", None)}
    assert checks["Gemini image models"]["ok"] is False


def test_terminal_doctor_reports_a_rejected_key(monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(provider_check, "check_key", lambda p, k: (provider_check.REJECTED, "anthropic rejected the key (HTTP 401)"))
    out = CliRunner().invoke(cli_module.app, ["doctor"]).output
    assert "anthropic rejected the key" in out and "ANTHROPIC_API_KEY" in out


# --- Gemini: the choices ----------------------------------------------------------


def test_the_image_picker_lists_gemini_models_from_the_key_with_prices(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GKEY)
    monkeypatch.setattr(provider_check, "gemini_image_models", lambda k: ["gemini/gemini-3.1-flash-image-preview", "gemini/imagen-3.0-generate-001"])
    server, out = _server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "image"})
    cloud = _last(out, "delegate_options")["cloud"]
    assert [(c["model"], c["price_usd"]) for c in cloud] == [("gemini/gemini-3.1-flash-image-preview", 0.045), ("gemini/imagen-3.0-generate-001", 0.04)]


def test_when_googles_list_cant_be_read_the_curated_gemini_image_models_are_used(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GKEY)
    monkeypatch.setattr(provider_check, "gemini_image_models", lambda k: [])
    server, out = _server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "image"})
    assert [c["model"] for c in _last(out, "delegate_options")["cloud"]] == config.IMAGE_MODEL_CHOICES["gemini"]


def test_a_task_type_can_choose_among_several_models_of_a_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GKEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(cli_module, "_gemini_models", lambda key: ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite"))
    server, out = _server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "docs"})
    cloud = _last(out, "delegate_options")["cloud"]
    gemini = [c["model"] for c in cloud if c["provider"] == "gemini"]
    anthropic = [c["model"] for c in cloud if c["provider"] == "anthropic"]
    assert gemini == ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite"]
    assert anthropic == config.FRONTIER_MODEL_CHOICES["anthropic"]  # not just the first
    assert all(c["kind"] == "api" for c in cloud)


def test_a_picked_flash_model_can_really_be_pinned(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GKEY)
    dt.apply("docs", "api:gemini:gemini/gemini-2.5-flash")
    assert dt.get("docs").model == "gemini/gemini-2.5-flash"


# --- provider_check ---------------------------------------------------------------


def _http(monkeypatch, status=200, json_body=None, raises=None):
    def get(url, **kw):
        if raises:
            raise raises
        return httpx.Response(status, json=json_body or {}, request=httpx.Request("GET", url))

    monkeypatch.setattr(provider_check.httpx, "get", get)


@pytest.mark.parametrize("status, expected", [(200, "ok"), (401, "rejected"), (403, "rejected"), (400, "rejected"), (500, "unreachable"), (429, "unreachable")])
def test_check_key_reads_the_providers_answer(monkeypatch, status, expected):
    _http(monkeypatch, status)
    assert provider_check.check_key("openai", KEY)[0] == expected


def test_check_key_treats_a_network_failure_as_unreachable_not_wrong(monkeypatch):
    _http(monkeypatch, raises=httpx.ConnectError("blocked"))
    status, detail = provider_check.check_key("anthropic", KEY)
    assert status == "unreachable" and "ConnectError" in detail and KEY not in detail


def test_the_key_is_sent_in_a_header_never_the_url(monkeypatch):
    seen = {}

    def get(url, headers=None, **kw):
        seen.update(url=url, headers=headers, kw=kw)
        return httpx.Response(200, json={}, request=httpx.Request("GET", url))

    monkeypatch.setattr(provider_check.httpx, "get", get)
    provider_check.check_key("gemini", GKEY)
    assert GKEY not in seen["url"] and seen["headers"]["x-goog-api-key"] == GKEY and "params" not in seen["kw"]


def test_gemini_image_models_are_only_those_the_key_lists_and_litellm_can_call(monkeypatch):
    body = {"models": [
        {"name": "models/gemini-3.1-flash-image-preview"}, {"name": "models/gemini-2.5-flash-image"},
        {"name": "models/imagen-3.0-generate-001"}, {"name": "models/gemini-2.5-pro"},          # text: not an image model
        {"name": "models/imagen-99-made-up"},                                                   # image-ish, but LiteLLM can't call it
    ]}
    _http(monkeypatch, 200, body)
    found = provider_check.gemini_image_models(GKEY)
    assert found[0] == "gemini/gemini-3.1-flash-image-preview" and "gemini/imagen-3.0-generate-001" in found
    assert "gemini/gemini-2.5-pro" not in found and "gemini/imagen-99-made-up" not in found


def test_gemini_image_models_is_empty_when_the_list_cant_be_read(monkeypatch):
    _http(monkeypatch, raises=httpx.ConnectError("x"))
    assert provider_check.gemini_image_models(GKEY) == []
    assert provider_check.image_models_for("gemini", GKEY) == config.IMAGE_MODEL_CHOICES["gemini"]
    assert provider_check.image_models_for("openai", None) == config.IMAGE_MODEL_CHOICES["openai"]


def test_prices_come_from_litellm_where_known():
    assert provider_check.image_price("gemini/gemini-2.5-flash-image") == 0.039
    assert provider_check.image_price("gpt-image-1") is None and provider_check.image_price("nonsense") is None


def test_every_curated_gemini_image_model_is_one_litellm_knows():
    import litellm

    known = {k for k, v in litellm.model_cost.items() if v.get("mode") == "image_generation"}
    assert set(config.IMAGE_MODEL_CHOICES["gemini"]) <= known
