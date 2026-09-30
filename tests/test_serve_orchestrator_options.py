"""The desktop header dropdown lists every orchestrator usable on this
machine (models already in Ollama + cloud providers with a key or CLI login),
not a fixed list, and picking one switches auth the way `/model` does in the
terminal (reported: "the drop down at the top does not contain all the models").
"""
import io
import json
import os
import time

import localforge.cli as cli_module
from localforge import config, trust
from localforge.serve import StdioServer


def make_server(tmp_path, model="test-model", cli_provider=None):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    trust.trust(tmp_path)
    server = StdioServer(tmp_path, model, cli_provider, out=out, run_fn=lambda *a, **k: None, scratch_root=scratch, conversation=object())
    return server, out


def events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


def wait_for(out, kind):
    for _ in range(100):
        if events(out, kind):
            return events(out, kind)[-1]
        time.sleep(0.02)
    raise AssertionError(f"no {kind} event")


def fake_env(monkeypatch, installed=("qwen2.5:14b", "llama3.1:8b")):
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda ollama: set(installed))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda provider: False)


def test_options_list_installed_local_models_and_keyed_providers(tmp_path, monkeypatch):
    fake_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    server, out = make_server(tmp_path)
    server.handle({"type": "orchestrator_options_request"})
    ev = wait_for(out, "orchestrator_options")
    ids = [o["id"] for o in ev["options"]]
    assert "ollama/qwen2.5:14b" in ids and "ollama/llama3.1:8b" in ids
    assert set(config.FRONTIER_MODEL_CHOICES["anthropic"]) <= set(ids)
    assert ev["current"] == "test-model"
    groups = {o["id"]: o["group"] for o in ev["options"]}
    assert groups["ollama/qwen2.5:14b"].startswith("Local")
    assert "API key" in groups["claude-opus-5"]


def test_providers_without_a_key_or_login_are_not_offered(tmp_path, monkeypatch):
    fake_env(monkeypatch, installed=())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    server, out = make_server(tmp_path)
    server.handle({"type": "orchestrator_options_request"})
    assert wait_for(out, "orchestrator_options")["options"] == []


def test_a_cli_login_provider_is_offered_and_labelled_as_a_login(tmp_path, monkeypatch):
    fake_env(monkeypatch, installed=())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda provider: provider == "anthropic")
    server, out = make_server(tmp_path)
    server.handle({"type": "orchestrator_options_request"})
    ev = wait_for(out, "orchestrator_options")
    assert ev["options"] and all("login" in o["group"] for o in ev["options"])


def test_picking_a_local_model_saves_local_auth_and_drops_the_cli_provider(tmp_path, monkeypatch):
    fake_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    server, out = make_server(tmp_path, model="claude-opus-5", cli_provider="anthropic")
    server.handle({"type": "set_model", "model": "ollama/qwen2.5:14b"})
    assert server.frontier_model == "ollama/qwen2.5:14b"
    assert server.cli_provider is None  # a stale claude login must not route a local model
    assert os.environ[config.AUTH_METHOD_ENV_VAR] == config.AUTH_LOCAL
    assert os.environ[config.FRONTIER_PROVIDER_ENV_VAR] == "local"
    assert events(out, "settings")[-1]["model"] == "ollama/qwen2.5:14b"


def test_picking_a_cloud_model_from_a_login_uses_that_providers_cli(tmp_path, monkeypatch):
    fake_env(monkeypatch, installed=())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda provider: provider == "anthropic")
    server, out = make_server(tmp_path, model="ollama/qwen2.5:14b")
    server.handle({"type": "set_model", "model": "claude-sonnet-5"})
    assert server.frontier_model == "claude-sonnet-5"
    assert server.cli_provider == "anthropic"
    assert os.environ[config.AUTH_METHOD_ENV_VAR] == config.AUTH_CLI_LOGIN


def test_a_typed_model_id_still_works_without_touching_saved_auth(tmp_path, monkeypatch):
    fake_env(monkeypatch, installed=())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    server, out = make_server(tmp_path)
    server.handle({"type": "set_model", "model": "some-custom-model"})
    assert server.frontier_model == "some-custom-model"
    assert config.AUTH_METHOD_ENV_VAR not in os.environ


def test_slash_model_switches_the_same_way(tmp_path, monkeypatch):
    fake_env(monkeypatch)
    server, out = make_server(tmp_path, model="claude-opus-5", cli_provider="anthropic")
    server.handle({"type": "user_message", "text": "/model ollama/llama3.1:8b"})
    assert server.cli_provider is None
    assert events(out, "model")[-1]["model"] == "ollama/llama3.1:8b"
