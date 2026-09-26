"""Image upload/attach support across all three orchestrator transports:
API-key (LiteLLM, native multimodal content blocks), CLI-login (Claude
only, via `--input-format stream-json`), and local (a vision-capable
Ollama model, via its `images` field). See content_blocks.py and
orchestrator._check_image_support for the shared design.
"""
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from localforge import cli_transport, content_blocks, local_transport
from localforge.hardware import HardwareProfile
from localforge.orchestrator import ImageNotSupportedError, RunStats
import localforge.orchestrator as orch_module

IMAGE = {"mime_type": "image/png", "data": "QUJD"}  # "ABC" base64


def _hw() -> HardwareProfile:
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


# --- content_blocks.py: pure data-shape helpers ---


def test_user_content_is_plain_text_without_an_image():
    assert content_blocks.user_content("hello", None) == "hello"


def test_user_content_is_a_multimodal_block_list_with_an_image():
    content = content_blocks.user_content("describe this", IMAGE)
    assert content == [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]


def test_text_of_extracts_just_the_text_from_a_block_list():
    content = content_blocks.user_content("describe this", IMAGE)
    assert content_blocks.text_of(content) == "describe this"


def test_text_of_passes_through_a_plain_string():
    assert content_blocks.text_of("hello") == "hello"
    assert content_blocks.text_of(None) == ""


def test_extract_image_finds_the_latest_user_messages_image():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first turn, no image"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": content_blocks.user_content("second turn", IMAGE)},
    ]
    assert content_blocks.extract_image(messages) == IMAGE


def test_extract_image_is_none_when_no_message_has_one():
    assert content_blocks.extract_image([{"role": "user", "content": "plain"}]) is None


# --- orchestrator.py: pre-flight capability check ---


def test_check_image_support_allows_a_plain_api_key_call():
    orch_module._check_image_support("claude-opus-5", None)  # must not raise


def test_check_image_support_allows_claude_cli_login():
    orch_module._check_image_support("claude-opus-5", "anthropic")  # must not raise


def test_check_image_support_rejects_other_cli_logins():
    with pytest.raises(ImageNotSupportedError, match="codex"):
        orch_module._check_image_support("gpt-5", "codex")


def test_check_image_support_rejects_a_non_vision_local_model():
    with pytest.raises(ImageNotSupportedError, match="vision-capable"):
        orch_module._check_image_support("ollama/qwen2.5:7b", None)


def test_check_image_support_allows_a_vision_capable_local_model():
    orch_module._check_image_support("ollama/llava:7b", None)  # must not raise


def test_run_raises_up_front_before_any_call_is_made():
    """The whole point of checking before run() does any work: no call to
    completion()/a local model/a CLI should happen at all."""
    with patch.object(orch_module, "completion") as completion_mock:
        with pytest.raises(ImageNotSupportedError):
            orch_module.run("describe this", "gpt-5", hardware=_hw(), cli_provider="codex", image=IMAGE)
    completion_mock.assert_not_called()


def test_run_attaches_the_image_to_the_users_message_for_an_api_key_call():
    """litellm.completion() natively understands this content-block shape
    across Claude/GPT/Gemini vision models -- orchestrator.run() just needs
    to build it, no transport-specific code required for this path."""
    seen_messages = {}

    def fake_completion(model, messages, tools):
        seen_messages["messages"] = [dict(m) for m in messages]
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(tool_calls=None, content="done", model_dump=lambda: {"role": "assistant", "content": "done"}))]
        usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        resp.usage = usage
        return resp

    with (
        patch.object(orch_module, "completion", side_effect=fake_completion),
        patch.object(orch_module.litellm, "completion_cost", return_value=0.0),
    ):
        orch_module.run("describe this", "gpt-5", hardware=_hw(), image=IMAGE)

    user_message = next(m for m in seen_messages["messages"] if m["role"] == "user")
    assert user_message["content"] == content_blocks.user_content("describe this", IMAGE)


# --- local_transport.py: a vision-capable Ollama model ---


@pytest.mark.parametrize("model", ["llava", "llava:13b", "moondream", "bakllava:7b"])
def test_supports_vision_matches_known_vision_families(model):
    assert local_transport.supports_vision(f"ollama/{model}") is True


@pytest.mark.parametrize("model", ["qwen2.5:7b", "gemma3:4b", "llama3.1:8b"])
def test_supports_vision_is_false_for_text_only_models(model):
    assert local_transport.supports_vision(f"ollama/{model}") is False


def test_local_transport_passes_the_image_to_ollamas_images_field():
    seen = {}

    def handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": '{"final_answer": "done"}'}, "prompt_eval_count": 10, "eval_count": 5})
        return httpx.Response(200, json={"models": [{"name": "llava:7b"}], "version": "x"})

    real = httpx.Client
    messages = [{"role": "user", "content": content_blocks.user_content("what is this", IMAGE)}]
    with (
        patch.object(local_transport.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)),
        patch.object(local_transport.OllamaBackend, "is_running", return_value=True),
        patch.object(local_transport.OllamaBackend, "ensure_available"),
    ):
        local_transport.complete("ollama/llava:7b", messages, [])

    assert seen["messages"][0]["images"] == [IMAGE["data"]]


def test_local_transport_omits_images_field_without_an_attachment():
    seen = {}

    def handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": '{"final_answer": "done"}'}, "prompt_eval_count": 10, "eval_count": 5})
        return httpx.Response(200, json={"models": [{"name": "llava:7b"}], "version": "x"})

    real = httpx.Client
    with (
        patch.object(local_transport.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)),
        patch.object(local_transport.OllamaBackend, "is_running", return_value=True),
        patch.object(local_transport.OllamaBackend, "ensure_available"),
    ):
        local_transport.complete("ollama/llava:7b", [{"role": "user", "content": "plain text"}], [])

    assert "images" not in seen["messages"][0]


# --- cli_transport.py: Claude-only, via --input-format stream-json ---


def _proc(stdout: str, returncode: int = 0):
    return MagicMock(stdout=stdout, stderr="", returncode=returncode)


def test_claude_cli_gets_input_format_stream_json_and_an_image_content_block():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": content_blocks.user_content("what is this", IMAGE)},
    ]
    run_mock = MagicMock(return_value=_proc(json.dumps({"result": '{"final_answer": "a cat"}'})))
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", run_mock):
        cli_transport.complete("anthropic", messages, [])

    args, kwargs = run_mock.call_args
    cmd = args[0]
    assert "--input-format" in cmd and cmd[cmd.index("--input-format") + 1] == "stream-json"
    envelope = json.loads(kwargs["input"])
    assert envelope["type"] == "user"
    blocks = envelope["message"]["content"]
    assert blocks[0] == {"type": "text", "text": cli_transport._render_split(messages, [])[1]}
    assert blocks[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}


def test_claude_cli_without_an_image_is_unchanged():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    run_mock = MagicMock(return_value=_proc(json.dumps({"result": '{"final_answer": "hi"}'})))
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", run_mock):
        cli_transport.complete("anthropic", messages, [])

    cmd = run_mock.call_args.args[0]
    assert "--input-format" not in cmd
    assert isinstance(run_mock.call_args.kwargs["input"], str)
    assert not run_mock.call_args.kwargs["input"].startswith("{")  # plain flattened prompt, not a JSON envelope


def test_other_cli_providers_never_build_an_image_envelope():
    """extract_image() is only even consulted for provider == "anthropic" --
    orchestrator._check_image_support() already refuses other CLI providers
    before this is reached, but this pins the transport-level guard too."""
    messages = [{"role": "user", "content": content_blocks.user_content("what is this", IMAGE)}]
    run_mock = MagicMock(return_value=_proc(json.dumps({"result": '{"final_answer": "ok"}'})))
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", run_mock):
        cli_transport.complete("openai", messages, [])

    cmd = run_mock.call_args.args[0]
    assert "--input-format" not in cmd
