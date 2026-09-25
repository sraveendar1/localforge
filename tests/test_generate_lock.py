"""A side action -- a side question asked mid-task (background.TaskRunner.
ask_side_question fires it on its own thread by design, deliberately
concurrent with the running task), memory compaction, or a local
orchestrator's own turn -- can make its own call into Ollama at the same
time as the running task's own generate() call.

Reported bug: "if I compact or ask a question, the [running background]
task stops." On the kind of memory-constrained machine this project targets
(catalog.py's balanced-selection notes: often just one model fits at all),
Ollama can't hold two models' weights loaded at once, so a second concurrent
request makes it evict whichever model is already loaded -- which yanks the
model out from under the first call's in-flight generate(), surfacing as a
hard failure rather than a graceful queue. GENERATE_LOCK (backends/ollama.py)
serializes every generate() call in this process so a second call waits its
turn instead of racing the first one.
"""

from __future__ import annotations

import threading

import httpx

from localforge import local_transport
from localforge.backends import ollama
from localforge.backends.ollama import OllamaBackend


def _blocking_handler(entered: threading.Event, release: threading.Event):
    """A fake Ollama whose /api/generate or /api/chat blocks until
    `release` is set, after signalling `entered` -- so a test can prove
    nothing else runs while this request is in flight."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {}})
        entered.set()
        release.wait(timeout=5)
        return httpx.Response(
            200,
            json={"response": "ok", "message": {"content": "ok"}, "done": True, "eval_count": 1},
        )

    return handler


def test_generate_holds_the_lock_for_the_whole_call(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    handler = _blocking_handler(entered, release)
    backend = OllamaBackend()
    monkeypatch.setattr(
        backend, "_client", lambda *a, **k: httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    )

    thread = threading.Thread(target=backend.generate, args=("m", "prompt"))
    thread.start()
    try:
        assert entered.wait(timeout=2), "the fake request never started"
        # The call is inside its HTTP request right now, so the lock must
        # still be held -- a non-blocking acquire must fail.
        assert not ollama.GENERATE_LOCK.acquire(blocking=False), "generate() didn't hold the lock while its request was in flight"
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    # Released once the call finished.
    assert ollama.GENERATE_LOCK.acquire(blocking=False)
    ollama.GENERATE_LOCK.release()


def test_a_local_orchestrators_own_turn_shares_the_same_lock(monkeypatch):
    """local_transport.complete() (a local model as the orchestrator itself)
    is just as exposed to this race as a delegated generate() call, so it
    must hold the very same lock, not a lock of its own."""
    entered, release = threading.Event(), threading.Event()
    handler = _blocking_handler(entered, release)
    backend = OllamaBackend()
    monkeypatch.setattr(backend, "is_running", lambda: True)
    monkeypatch.setattr(backend, "ensure_available", lambda name, on_progress=None: None)
    monkeypatch.setattr(local_transport, "OllamaBackend", lambda: backend)
    monkeypatch.setattr(local_transport, "context_window", lambda name: 8192)
    real_client = httpx.Client  # captured before patching -- httpx.Client is the same object local_transport sees
    monkeypatch.setattr(
        local_transport.httpx, "Client", lambda *a, **k: real_client(base_url="http://x", transport=httpx.MockTransport(handler))
    )

    thread = threading.Thread(target=local_transport.complete, args=("ollama/m", [{"role": "user", "content": "hi"}], []))
    thread.start()
    try:
        assert entered.wait(timeout=2), "the fake request never started"
        assert not ollama.GENERATE_LOCK.acquire(blocking=False), "local_transport.complete() didn't hold GENERATE_LOCK"
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert ollama.GENERATE_LOCK.acquire(blocking=False)
    ollama.GENERATE_LOCK.release()


def test_a_second_generate_call_waits_instead_of_racing():
    """The concrete failure mode: two concurrent calls must run one at a
    time, never overlapping mid-request."""
    order: list[str] = []
    guard = threading.Lock()
    release_first = threading.Event()
    first_entered = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {}})
        with guard:
            order.append("start")
        if not first_entered.is_set():
            first_entered.set()
            release_first.wait(timeout=5)
        with guard:
            order.append("end")
        return httpx.Response(200, json={"response": "ok", "done": True, "eval_count": 1})

    backend_a, backend_b = OllamaBackend(), OllamaBackend()
    client = lambda *a, **k: httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    backend_a._client = client
    backend_b._client = client

    first = threading.Thread(target=backend_a.generate, args=("m", "prompt"))
    first.start()
    assert first_entered.wait(timeout=2)

    second = threading.Thread(target=backend_b.generate, args=("m", "prompt"))
    second.start()
    # The second call must be blocked on the lock, not off making its own
    # request while the first is still mid-flight.
    second.join(timeout=0.3)
    assert second.is_alive(), "a second generate() call ran concurrently with the first"

    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert order == ["start", "end", "start", "end"], f"the two calls overlapped: {order}"
