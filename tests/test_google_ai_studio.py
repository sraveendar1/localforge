"""Google AI Studio as an orchestrator (reported: "we can't use Google AI
Studio"). The saved id was a bare `gemini-2.5-pro`, which LiteLLM routes to
Vertex AI (Google Cloud credentials), so an AI Studio key never worked."""

import contextlib
import os
from unittest.mock import patch

import httpx
import litellm

import localforge.cli as cli_module
from localforge import cli_transport, config


def test_offered_gemini_ids_route_to_ai_studio_not_vertex():
    for model in config.FRONTIER_MODEL_CHOICES["gemini"]:
        assert litellm.get_llm_provider(model)[1] == "gemini", model


def test_a_bare_saved_gemini_id_is_migrated_on_load(monkeypatch):
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "gemini-2.5-pro")
    config.load()
    assert os.environ[config.FRONTIER_MODEL_ENV_VAR] == "gemini/gemini-2.5-pro"
    assert config.litellm_model_id("claude-opus-5") == "claude-opus-5"
    assert config.litellm_model_id("ollama/gemma3:4b") == "ollama/gemma3:4b"


def test_google_api_key_counts_as_an_ai_studio_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    config.load()
    assert os.environ["GEMINI_API_KEY"] == "g-key"


def test_the_gemini_cli_gets_the_bare_model_id():
    spec = config.FRONTIER_CLI_AUTH["gemini"]
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        raise RuntimeError("stop here")

    with patch.object(cli_transport.subprocess, "run", fake_run), patch.object(cli_transport, "available", return_value=True):
        with contextlib.suppress(Exception):  # only the command line matters here
            cli_transport.complete("gemini", [{"role": "user", "content": "hi"}], [], model="gemini/gemini-2.5-pro")
    assert spec["model_flag"] in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index(spec["model_flag"]) + 1] == "gemini-2.5-pro"


def _google_list(names):
    return {"models": [{"name": f"models/{n}", "supportedGenerationMethods": methods} for n, methods in names]}


def test_model_menu_lists_what_the_key_can_use(monkeypatch):
    cli_module._gemini_cache.clear()
    listed = _google_list(
        [
            ("gemini-2.5-flash", ["generateContent"]),
            ("gemini-3.1-pro-preview", ["generateContent"]),
            ("gemini-2.5-pro", ["generateContent"]),
            ("gemini-2.5-flash-preview-tts", ["generateContent"]),  # audio, not an orchestrator
            ("text-embedding-004", ["embedContent"]),
        ]
    )
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["url"], seen["headers"] = url, headers
        return httpx.Response(200, json=listed, request=httpx.Request("GET", url))

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with patch.object(cli_module.httpx, "get", fake_get):
        models = cli_module._provider_models("gemini")
    assert models == ["gemini/gemini-3.1-pro-preview", "gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"]
    assert "k" not in seen["url"] and seen["headers"] == {"x-goog-api-key": "k"}


def test_model_menu_falls_back_to_the_curated_list_offline(monkeypatch):
    cli_module._gemini_cache.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with patch.object(cli_module.httpx, "get", side_effect=httpx.ConnectError("offline")):
        assert cli_module._provider_models("gemini") == config.FRONTIER_MODEL_CHOICES["gemini"]
    assert "k" not in cli_module._gemini_cache  # tried again next time


def test_model_command_accepts_a_bare_gemini_id(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(cli_module, "_provider_models", lambda p: list(config.FRONTIER_MODEL_CHOICES.get(p, [])))
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda _b: set())
    cli_module.model_command("gemini-2.5-pro")
    assert os.environ[config.FRONTIER_MODEL_ENV_VAR] == "gemini/gemini-2.5-pro"
    assert os.environ[config.FRONTIER_PROVIDER_ENV_VAR] == "gemini"
