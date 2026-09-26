"""backends/cloud.py: the two delegate backends an "advanced" per-modality
override can route to instead of a local Ollama model -- an API-key call
(billed) and a CLI-login call (subscription, notional cost only). Both are
one-shot: no tool calling, just instructions in and content out, same
contract every local backend already fulfills.
"""
import types

import pytest

from localforge.backends import cloud as cloud_module
from localforge.backends.cloud import CloudApiBackend, CloudCliBackend, CloudProviderNotConfigured
from localforge import cli_transport


def _fake_completion_response(text="print('hi')", completion_tokens=42):
    usage = types.SimpleNamespace(completion_tokens=completion_tokens, prompt_tokens=10)
    message = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class TestCloudApiBackend:
    def test_ensure_available_raises_clearly_with_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        backend = CloudApiBackend()
        with pytest.raises(CloudProviderNotConfigured, match="ANTHROPIC_API_KEY"):
            backend.ensure_available("claude-haiku-4-5", provider="anthropic")

    def test_ensure_available_passes_with_a_key_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        CloudApiBackend().ensure_available("claude-haiku-4-5", provider="anthropic")  # no raise

    def test_generate_calls_litellm_and_returns_a_backend_result(self, monkeypatch):
        captured = {}

        def fake_completion(model, messages):
            captured["model"] = model
            captured["messages"] = messages
            return _fake_completion_response("the file contents")

        monkeypatch.setattr(cloud_module.litellm, "completion", fake_completion)
        monkeypatch.setattr(cloud_module.litellm, "completion_cost", lambda completion_response: 0.0123)

        result = CloudApiBackend().generate("claude-haiku-4-5-20251001", "write me a function", provider="anthropic")

        assert captured["model"] == "claude-haiku-4-5-20251001"
        assert captured["messages"] == [{"role": "user", "content": "write me a function"}]
        assert result["type"] == "text"
        assert result["content"] == "the file contents"
        assert result["tokens"] == 42
        assert result["cost_usd"] == 0.0123
        assert result["notional_cost_usd"] == 0.0

    def test_gemini_ids_get_the_gemini_prefix(self, monkeypatch):
        captured = {}

        def fake_completion(model, messages):
            captured["model"] = model
            return _fake_completion_response()

        monkeypatch.setattr(cloud_module.litellm, "completion", fake_completion)
        monkeypatch.setattr(cloud_module.litellm, "completion_cost", lambda completion_response: 0.0)

        CloudApiBackend().generate("gemini-2.5-flash", "hi", provider="gemini")

        assert captured["model"] == "gemini/gemini-2.5-flash"

    def test_an_unpriced_model_does_not_fail_the_delegation(self, monkeypatch):
        monkeypatch.setattr(cloud_module.litellm, "completion", lambda model, messages: _fake_completion_response())

        def raises_cost(completion_response):
            raise Exception("no pricing known for this model")

        monkeypatch.setattr(cloud_module.litellm, "completion_cost", raises_cost)

        result = CloudApiBackend().generate("some-model", "hi", provider="anthropic")
        assert result["cost_usd"] == 0.0

    def test_on_token_is_called_with_the_full_reply(self, monkeypatch):
        monkeypatch.setattr(cloud_module.litellm, "completion", lambda model, messages: _fake_completion_response("full text"))
        monkeypatch.setattr(cloud_module.litellm, "completion_cost", lambda completion_response: 0.0)

        chunks = []
        CloudApiBackend().generate("m", "p", on_token=chunks.append, provider="anthropic")
        assert chunks == ["full text"]


class TestCloudCliBackend:
    def test_ensure_available_raises_when_the_cli_is_not_available(self, monkeypatch):
        monkeypatch.setattr(cli_transport, "available", lambda provider: False)
        monkeypatch.setattr(cli_transport, "requirements_message", lambda provider: f"install {provider}'s CLI")
        with pytest.raises(CloudProviderNotConfigured, match="install anthropic's CLI"):
            CloudCliBackend().ensure_available("claude-haiku-4-5", provider="anthropic")

    def test_ensure_available_passes_when_the_cli_is_available(self, monkeypatch):
        monkeypatch.setattr(cli_transport, "available", lambda provider: True)
        CloudCliBackend().ensure_available("claude-haiku-4-5", provider="anthropic")  # no raise

    def test_generate_reuses_cli_transport_complete_with_no_tools(self, monkeypatch):
        captured = {}

        def fake_complete(provider, messages, tools, timeout=600.0, model=None, on_text=None):
            captured.update(provider=provider, messages=messages, tools=tools, model=model)
            usage = cli_transport._Usage(completion_tokens=17)
            msg = cli_transport._Message(content="delegated content", tool_calls=None)
            choice = types.SimpleNamespace(message=msg)
            return cli_transport.CLIResponse(choices=[choice], usage=usage, notional_cost_usd=0.004, local=False)

        monkeypatch.setattr(cli_transport, "complete", fake_complete)

        result = CloudCliBackend().generate("claude-haiku-4-5-20251001", "write this file", provider="anthropic")

        assert captured["provider"] == "anthropic"
        assert captured["tools"] == []
        assert captured["model"] == "claude-haiku-4-5-20251001"
        assert captured["messages"][0] == {"role": "system", "content": "write this file"}
        assert result["content"] == "delegated content"
        assert result["tokens"] == 17
        assert result["cost_usd"] == 0.0
        assert result["notional_cost_usd"] == 0.004
