import io
import json
import types

from localforge.serve import StdioServer, Scratchpad


def make_server(tmp_path, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", object())
    server = StdioServer(tmp_path, "test-model", out=out, run_fn=lambda *args: None, scratch_root=scratch, **kwargs)
    return server, out


def events(out):
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def of_type(out, event_type):
    return [e for e in events(out) if e["type"] == event_type]


def test_get_state_empty_todos(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "get_state"})
    assert "todos" in of_type(out, "todos_updated")[0]
    assert not of_type(out, "todos_updated")[0]["todos"]


def test_on_todos_replays_todos(tmp_path):
    server, out = make_server(tmp_path)
    server._on_todos([{"content": "Task 1", "status": "pending"}])
    server.handle({"type": "get_state"})
    assert "todos" in of_type(out, "todos_updated")[0]
    assert of_type(out, "todos_updated")[0]["todos"][0]["content"] == "Task 1"
    assert of_type(out, "todos_updated")[0]["todos"][0]["status"] == "pending"


def test_new_session_resets_todos(tmp_path):
    server, out = make_server(tmp_path)
    server._on_todos([{"content": "Task 1", "status": "pending"}])
    server.handle({"type": "new_session"})
    assert of_type(out, "session_reset")
    assert "todos" in of_type(out, "todos_updated")[-1]
    assert not of_type(out, "todos_updated")[-1]["todos"]
