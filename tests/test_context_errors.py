"""Context-window failures reach the orchestrator (reported: "if the context
window is large for the lower-weighted models, it errors and doesn't tell the
frontier model or the orchestrator").

Two causes, both verified against Ollama 0.34:
* Ollama silently caps num_ctx at the length a model was trained for (asked
  for 262k, it loaded 32k) and cuts whatever prompt doesn't fit. No error.
* When a model can't be loaded at the window asked for, the reason is in the
  response body; a bare raise_for_status() dropped it ("500 Server Error").
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from localforge import local_transport, tools
from localforge.backends import ollama
from localforge.backends.ollama import (
    LocalModelOutOfMemory,
    LocalPromptTooLarge,
    OllamaBackend,
)
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import Dispatcher
from localforge.workspace import Workspace

OOM = "model requires more system memory (12.4 GiB) than is available (7.9 GiB)"


def _ollama(trained=32768, generate=None):
    """A fake Ollama: /api/show reports `trained`; /api/generate and /api/chat
    are answered by `generate(body)` -> httpx.Response."""
    sent = []

    def handler(request):
        body = json.loads(request.content or b"{}")
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"qwen2.context_length": trained}})
        sent.append(body)
        if generate is not None:
            return generate(body)
        return httpx.Response(200, content=json.dumps({"response": "ok", "done": True, "eval_count": 1}).encode())

    client = lambda *a: httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    return sent, client


# --- the backend never lets Ollama cut a prompt silently ---------------------------------


@pytest.mark.parametrize("streaming", [True, False])
def test_a_prompt_longer_than_the_trained_window_is_refused_not_cut(streaming):
    sent, client = _ollama(trained=8192)
    backend = OllamaBackend()
    kwargs = {"on_token": lambda _: None} if streaming else {}
    with patch.object(backend, "_client", client), pytest.raises(LocalPromptTooLarge, match="about 8,192 tokens"):
        backend.generate("old-llama:8b", "x" * 40_000, **kwargs)  # ~13k tokens
    assert sent == []  # nothing was sent


def test_the_window_never_exceeds_what_the_model_was_trained_for():
    sent, client = _ollama(trained=8192)
    backend = OllamaBackend()
    with patch.object(backend, "_client", client):
        backend.generate("old-llama:8b", "x" * 9000, on_token=lambda _: None, context_limit=32768)
    assert sent[0]["options"]["num_ctx"] == 8192
    assert sent[0]["options"]["num_predict"] <= 8192 - ollama.estimate_tokens("x" * 9000)


@pytest.mark.parametrize("streaming", [True, False])
def test_ollamas_own_reason_is_kept(streaming):
    _, client = _ollama(generate=lambda body: httpx.Response(500, json={"error": OOM}))
    backend = OllamaBackend()
    kwargs = {"on_token": lambda _: None} if streaming else {}
    with patch.object(backend, "_client", client), pytest.raises(LocalModelOutOfMemory, match="12.4 GiB"):
        backend.generate("qwen2.5-coder:7b", "hi", **kwargs)


def test_other_errors_keep_their_reason_too():
    _, client = _ollama(generate=lambda body: httpx.Response(400, json={"error": "invalid option num_ctx"}))
    backend = OllamaBackend()
    with patch.object(backend, "_client", client), pytest.raises(RuntimeError, match="invalid option num_ctx"):
        backend.generate("qwen2.5-coder:7b", "hi", on_token=lambda _: None)


def test_a_mid_stream_memory_error_is_recognised():
    lines = [{"response": "def", "done": False}, {"error": "CUDA error: out of memory"}]
    _, client = _ollama(generate=lambda body: httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode()))
    backend = OllamaBackend()
    with patch.object(backend, "_client", client), pytest.raises(LocalModelOutOfMemory):
        backend.generate("qwen2.5-coder:7b", "hi", on_token=lambda _: None)


# --- the dispatcher sizes to it, recovers, and tells the orchestrator --------------------

CODER = [ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1, kv_gb_per_8k=0.1)]


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=64, free_disk_gb=100, gpus=[], memory_bandwidth_gbps=400)


class Stub:
    def __init__(self, trained=None, fail_above=None, always_fail=False):
        self.trained, self.fail_above, self.always_fail, self.calls = trained, fail_above, always_fail, []

    def trained_context(self, name):
        return self.trained

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.calls.append((len(prompt), kw.get("context_limit")))
        if self.always_fail or (self.fail_above and (kw.get("context_limit") or 32768) > self.fail_above):
            raise LocalModelOutOfMemory(f"{name} couldn't be loaded: {OOM}")
        return {"type": "text", "content": "a reasonable answer from the model", "tokens": 5}


@pytest.fixture(autouse=True)
def _no_shrunk_windows():
    tools._shrunk_windows.clear()
    yield
    tools._shrunk_windows.clear()


def test_prompts_are_sized_to_the_trained_window(tmp_path):
    (tmp_path / "big.py").write_text("y = 1\n" * 10_000)  # 60k chars
    stub = Stub(trained=8192)
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(tmp_path))
    with patch.dict("localforge.tools.BACKENDS", {"stub": stub}):
        out = d.dispatch("delegate_coding_task", {"instructions": "summarize", "context_files": ["big.py"]})
        assert d.context_limit(CODER[0]) == 8192
    assert stub.calls[0][0] <= ollama.prompt_budget(8192)  # fits: nothing for Ollama to cut
    assert "big.py was cut" in out  # and the orchestrator is told what was left out


def test_out_of_memory_retries_with_half_the_window_and_says_so(tmp_path):
    stub = Stub(fail_above=16384)
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(tmp_path))
    with patch.dict("localforge.tools.BACKENDS", {"stub": stub}):
        out = d.dispatch("delegate_coding_task", {"instructions": "write it"})
    assert [limit for _, limit in stub.calls] == [32768, 16384]
    assert "ran with a 16,384-token context window because memory was short" in out and "12.4 GiB" in out
    assert d.context_limit(CODER[0]) == 16384  # remembered for the next steps...


def test_the_smaller_window_is_forgotten_after_a_while(tmp_path, monkeypatch):
    tools._shrunk_windows["coder"] = (16384, 0.0)
    monkeypatch.setattr(tools.time, "monotonic", lambda: tools.SHRUNK_WINDOW_SECONDS + 1)
    d = Dispatcher(_hw(), catalog=CODER)
    with patch.dict("localforge.tools.BACKENDS", {"stub": Stub()}):
        assert d.context_limit(CODER[0]) != 16384  # ...but not forever: memory may have freed up


def test_when_memory_stays_short_the_orchestrator_gets_ollamas_reason(tmp_path):
    """End to end: the orchestrator's next message carries the failure."""
    import localforge.orchestrator as orch

    stub = Stub(always_fail=True)
    seen = []

    def completion(**kw):
        seen.append(kw["messages"])
        msg = MagicMock()
        if len(seen) == 1:
            call = MagicMock(id="c1")
            call.function.name, call.function.arguments = "delegate_coding_task", '{"instructions": "write it"}'
            msg.tool_calls, msg.content = [call], None
            msg.model_dump.return_value = {"role": "assistant", "content": None}
        else:
            msg.tool_calls, msg.content = None, "told the user"
            msg.model_dump.return_value = {"role": "assistant", "content": "told the user"}
        resp = MagicMock()
        resp.choices, resp.usage = [MagicMock(message=msg)], None
        return resp

    with (
        patch.object(orch, "completion", side_effect=completion),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
        patch("localforge.tools.load_catalog", return_value=CODER),
        patch.dict("localforge.tools.BACKENDS", {"stub": stub}),
    ):
        orch.run("task", "claude-opus-5", hardware=_hw(), workspace=Workspace(tmp_path))
    result = next(m for m in seen[1] if m.get("role") == "tool")["content"]
    assert result.startswith("delegate_coding_task failed:")
    assert "12.4 GiB" in result and "memory is short" in result  # Ollama's reason and what it means
    assert "This step failed" in result  # counted as a failure, so a sane retry is allowed


# --- a local orchestrator too ------------------------------------------------------------


def test_a_local_orchestrator_keeps_ollamas_reason(monkeypatch):
    real_client = httpx.Client

    def fake_client(*a, **kw):
        def handler(request):
            return httpx.Response(500, json={"error": OOM})

        return real_client(base_url="http://x", transport=httpx.MockTransport(handler))

    backend = MagicMock(is_running=lambda: True, ensure_available=lambda name: None, trained_context=lambda name: 32768)
    monkeypatch.setattr(local_transport, "OllamaBackend", lambda: backend)
    monkeypatch.setattr(local_transport.httpx, "Client", fake_client)
    with pytest.raises(local_transport.LocalOrchestratorError, match="12.4 GiB"):
        local_transport.complete("ollama/qwen2.5-coder:7b", [{"role": "user", "content": "hi"}], [])


def test_a_local_orchestrators_window_respects_the_trained_length(monkeypatch):
    backend = MagicMock(trained_context=lambda name: 8192)
    monkeypatch.setattr(local_transport, "OllamaBackend", lambda: backend)
    assert local_transport.context_window("some-old-model:8b") == 8192
