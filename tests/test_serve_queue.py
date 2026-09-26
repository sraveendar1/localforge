import io
import json
import types
from pathlib import Path
from localforge import trust
from localforge.serve import StdioServer


def make_server(tmp_path, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", object())
    trust.trust(tmp_path)  # these tests are about the protocol, not the trust gate itself
    server = StdioServer(tmp_path, "test-model", out=out, run_fn=lambda *args: None, scratch_root=scratch, **kwargs)
    return server, out


def events(out):
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def of_type(out, event_type):
    return [e for e in events(out) if e["type"] == event_type]


def test_queue_empty_on_fresh_session(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "get_state"})
    queue_event = of_type(out, "queue")[0]
    assert queue_event["items"] == []


def test_queue_list_empty_on_fresh_session(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "queue_list"})
    queue_event = of_type(out, "queue")[0]
    assert queue_event["items"] == []


def test_queue_clear_emits_empty_queue_event(tmp_path):
    server, out = make_server(tmp_path)
    server._queue.append("test message")
    server.handle({"type": "queue_clear"})
    queue_event = of_type(out, "queue")[0]
    assert queue_event["items"] == []


def test_new_session_emits_empty_queue_event(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "new_session"})
    queue_event = of_type(out, "queue")[0]
    assert queue_event["items"] == []
