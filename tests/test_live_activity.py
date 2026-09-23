"""Delegated work must be visible while it happens, and a run must not
silently download a model when a fitting one is already installed.

Reported: with `localforge`, all you saw during delegation was
"→ delegating coding to qwen2.5-coder:14b" and a spinner. Two causes: local
output wasn't streamed (Ollama was called with stream=False), and the
dispatcher ignored installed models, so it silently pulled the 9 GB 14b
mid-run although setup had picked the installed 7b.
"""

import json
import re
from unittest.mock import patch

import httpx
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.orchestrator import RunResult, RunStats
from localforge.tools import TASK_MODALITIES, ActivityHooks, Dispatcher

CODING = TASK_MODALITIES["coding"]["tool_name"]

CATALOG = [
    ModelEntry(name="coder:7b", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=8, disk_gb=5, quality_tier=2),
    ModelEntry(name="coder:14b", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=8, disk_gb=9, quality_tier=3),
    ModelEntry(name="coder:1b", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=8, disk_gb=1, quality_tier=1),
]


def _hw() -> HardwareProfile:
    return HardwareProfile(os="Darwin", arch="arm64", cpu_cores=10, ram_gb=32, free_disk_gb=200, gpus=[], memory_bandwidth_gbps=400)


class StreamingStub:
    def __init__(self, reply="def f():\n    return 1\n", tokens=12):
        self.reply, self.tokens = reply, tokens
        self.pulled: list[str] = []
        self.generated: list[str] = []

    def ensure_available(self, model_name, on_progress=None):
        self.pulled.append(model_name)
        if on_progress:
            on_progress({"status": "downloading", "total": 100, "completed": 50})

    def generate(self, model_name, prompt, on_token=None, **kwargs):
        self.generated.append(model_name)
        if on_token:
            for piece in self.reply.split(" "):
                on_token(piece + " ")
        return {"type": "text", "content": self.reply, "tokens": self.tokens}


# --- installed models win at run time -----------------------------------


def test_dispatcher_prefers_the_installed_model_over_a_bigger_download():
    stub = StreamingStub()
    dispatcher = Dispatcher(_hw(), catalog=CATALOG, installed={"coder:7b"})
    with patch.dict("localforge.tools.BACKENDS", {"stub": stub}):
        dispatcher.dispatch(CODING, "write f")
    assert stub.generated == ["coder:7b"]


def test_dispatcher_without_installed_info_keeps_best_fit():
    stub = StreamingStub()
    dispatcher = Dispatcher(_hw(), catalog=CATALOG)
    with patch.dict("localforge.tools.BACKENDS", {"stub": stub}):
        dispatcher.dispatch(CODING, "write f")
    assert stub.generated == ["coder:14b"]


def test_retry_never_switches_to_a_model_that_would_need_downloading():
    class EmptyFirst(StreamingStub):
        def generate(self, model_name, prompt, on_token=None, **kwargs):
            self.generated.append(model_name)
            content = "" if len(self.generated) == 1 else "x" * 500
            return {"type": "text", "content": content, "tokens": 1}

    stub = EmptyFirst()
    dispatcher = Dispatcher(_hw(), catalog=CATALOG, installed={"coder:7b"})
    with patch.dict("localforge.tools.BACKENDS", {"stub": stub}):
        dispatcher.dispatch(CODING, "y" * 100)
    # retried the same installed model, not coder:14b / coder:1b
    assert stub.generated == ["coder:7b", "coder:7b"]


def test_run_passes_installed_models_to_the_dispatcher():
    import localforge.orchestrator as orch

    seen = {}

    class Capture:
        def __init__(self, hardware, catalog=None, installed=None, hooks=None, **kwargs):
            seen["installed"] = installed
            raise RuntimeError("stop here")

    with (
        patch.object(orch, "Dispatcher", Capture),
        patch.object(orch.OllamaBackend, "list_installed", return_value=[{"name": "coder:7b"}]),
    ):
        try:
            orch.run("t", "gpt-5", hardware=_hw())
        except RuntimeError:
            pass
    assert seen["installed"] == {"coder:7b"}


def test_run_tolerates_ollama_being_unreachable_for_the_installed_list():
    import localforge.orchestrator as orch

    with patch.object(orch.OllamaBackend, "list_installed", side_effect=httpx.ConnectError("down")):
        assert orch._installed_models() is None


# --- hooks fire in order --------------------------------------------------


def test_hooks_see_delegate_tokens_pull_and_done():
    events = []
    hooks = ActivityHooks(
        on_delegate=lambda m, e: events.append(("delegate", e.name)),
        on_token=lambda chunk: events.append(("token", chunk)),
        on_done=lambda m, e, tokens, secs: events.append(("done", e.name, tokens)),
        on_pull=lambda name, ev: events.append(("pull", name, ev["completed"])),
    )
    dispatcher = Dispatcher(_hw(), catalog=CATALOG, installed={"coder:7b"}, hooks=hooks)
    with patch.dict("localforge.tools.BACKENDS", {"stub": StreamingStub(reply="a b", tokens=2)}):
        dispatcher.dispatch(CODING, "write f")

    assert events[0] == ("delegate", "coder:7b")
    assert events[1] == ("pull", "coder:7b", 50)
    assert [e[1] for e in events if e[0] == "token"] == ["a ", "b "]
    assert events[-1] == ("done", "coder:7b", 2)


def test_orchestrator_announces_each_frontier_round():
    import localforge.orchestrator as orch
    from tests.test_orchestrator import StubDispatcher, _final_response, _tool_call_response

    rounds = []
    responses = iter([_tool_call_response("c1"), _final_response()])
    with (
        patch.object(orch, "completion", side_effect=lambda **kw: next(responses)),
        patch.object(orch, "Dispatcher", StubDispatcher),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
        patch.object(orch, "_installed_models", return_value=None),
    ):
        orch.run("t", "gpt-5", hardware=_hw(), hooks=ActivityHooks(on_frontier=rounds.append))
    assert rounds == [1, 2]


# --- Ollama really streams -------------------------------------------------


def test_ollama_generate_streams_chunks_and_reads_final_token_count():
    lines = [
        {"response": "def ", "done": False},
        {"response": "f():", "done": False},
        {"response": "", "done": True, "eval_count": 7},
    ]
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(line) for line in lines).encode())

    backend = OllamaBackend()
    chunks = []
    with patch.object(backend, "_client", lambda *a: httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))):
        result = backend.generate("coder:7b", "prompt", on_token=chunks.append)

    assert requests[0]["stream"] is True
    assert chunks == ["def ", "f():"]
    assert result == {"type": "text", "content": "def f():", "tokens": 7, "truncated": False}


def test_ollama_streaming_surfaces_a_mid_stream_error():
    def handler(request):
        return httpx.Response(200, content=b'{"error": "model ran out of memory"}')

    backend = OllamaBackend()
    with patch.object(backend, "_client", lambda *a: httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))):
        try:
            backend.generate("coder:7b", "p", on_token=lambda c: None)
        except RuntimeError as exc:
            assert "out of memory" in str(exc)
        else:
            raise AssertionError("expected the Ollama error to be raised")


# --- what the user actually sees ------------------------------------------


def test_run_command_streams_local_output_indented_then_a_done_line(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module.config, "CONFIG_FILE", tmp_path / "config.env")
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)
    entry = CATALOG[0]

    def fake_run(task, frontier_model, cli_provider=None, hooks=None, **kwargs):
        hooks.on_frontier(1)
        hooks.on_delegate("coding", entry)
        for chunk in ["def f():\n", "    return 1", "\n"]:
            hooks.on_token(chunk)
        hooks.on_done("coding", entry, 9, 1.5)
        hooks.on_frontier(2)
        return RunResult("final answer", RunStats())

    with patch.object(cli_module, "run_orchestrator", side_effect=fake_run):
        out = CliRunner().invoke(cli_module.app, ["run", "task", "-m", "gpt-5"]).output

    lines = out.splitlines()
    assert "→ delegating coding to coder:7b (local, via stub)" in out
    assert "    def f():" in lines
    assert "        return 1" in lines
    # rate is measured from the first token, so only its presence is stable here
    assert re.search(r"✓ coder:7b finished coding: 9 tokens in 1\.5s, \d+ tok/s", out)
    assert out.index("def f()") < out.index("finished coding") < out.index("final answer")
