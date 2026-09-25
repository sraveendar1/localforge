"""The desktop app drives localforge through `localforge serve --stdio`; these pin
the JSON-lines protocol, the approval bridge and cancel."""

import dataclasses
import io
import json
import threading
import time

from localforge.orchestrator import OrchestrationError
from localforge.serve import StdioServer


@dataclasses.dataclass
class FakeStats:
    frontier_tokens: int = 7


@dataclasses.dataclass
class FakeResult:
    answer: str
    stats: FakeStats


def _done(*args, **kwargs):
    return FakeResult("done", FakeStats())


def make_server(tmp_path, run_fn=None, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", object())
    server = StdioServer(tmp_path, "test-model", out=out, run_fn=run_fn or _done, scratch_root=scratch, **kwargs)
    return server, out


def events(out):
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def of_type(out, event_type):
    return [e for e in events(out) if e["type"] == event_type]


def wait_for(out, event_type, timeout=5.0, count=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = of_type(out, event_type)
        if len(found) >= count:
            return found[count - 1]
        time.sleep(0.01)
    raise AssertionError(f"no {event_type} x{count}; saw {[e['type'] for e in events(out)]}")


def approve_in_thread(server, kind, title="t", detail="d"):
    result = []
    thread = threading.Thread(target=lambda: result.append(server.approve(kind, title, detail)), daemon=True)
    thread.start()
    return thread, result


def respond(server, req, decision):
    return server.handle({"type": "approval_response", "id": req["id"], "decision": decision})


def test_turn_streams_events_and_finishes(tmp_path):
    seen = {}
    conv = object()
    def run_fn(text, model, **kw):
        seen.update(kw)
        hooks = kw["hooks"]
        hooks.on_frontier(1)
        hooks.on_tool("read_file", "a.py")
        hooks.on_tool_result("read_file", "ok")
        hooks.on_todos([{"content": "x", "status": "pending"}])
        hooks.on_answer_text("hel")
        hooks.on_answer_text("lo")
        return FakeResult("hello", FakeStats())

    server, out = make_server(tmp_path, run_fn, conversation=conv)
    try:
        assert server.handle({"type": "user_message", "text": "hi"}) is True
        done = wait_for(out, "run_finished")
        assert done["answer"] == "hello"
        assert done["stats"] == {"frontier_tokens": 7}
        assert [e["type"] for e in events(out)] == ["run_started", "frontier_round", "tool_call_started", "tool_call_finished", "todos_updated", "text_delta", "text_delta", "run_finished"]
        assert "".join(e["text"] for e in of_type(out, "text_delta")) == "hello"
        assert seen["conversation"] is conv
        assert seen["workspace"] is not None
    finally:
        server.close()


def test_approve_and_decline(tmp_path):
    server, out = make_server(tmp_path)
    thread, result = approve_in_thread(server, "write", "a.py", "+x")
    req = wait_for(out, "approval_request")
    assert req["kind"] == "write" and req["title"] == "a.py" and req["detail"] == "+x"
    respond(server, req, "approve")
    thread.join(5)
    assert result == [True]

    thread, result = approve_in_thread(server, "command")
    req = wait_for(out, "approval_request", count=2)
    respond(server, req, "decline")
    thread.join(5)
    assert result == [False]


def test_always_skips_later_prompts_but_not_for_delete(tmp_path):
    server, out = make_server(tmp_path)
    thread, result = approve_in_thread(server, "write")
    req = wait_for(out, "approval_request")
    respond(server, req, "always")
    thread.join(5)
    assert result == [True]
    assert "write" in server.always_allow
    assert server.approve("write", "b.py", "+y") is True
    assert wait_for(out, "approval_auto")["title"] == "b.py"

    thread, result = approve_in_thread(server, "delete")
    req = wait_for(out, "approval_request", count=2)
    respond(server, req, "always")
    thread.join(5)
    assert result == [True]
    assert "delete" not in server.always_allow

    thread, result = approve_in_thread(server, "delete")
    req = wait_for(out, "approval_request", count=3)
    respond(server, req, "decline")
    thread.join(5)
    assert result == [False]


def test_auto_approve_never_covers_delete(tmp_path):
    server, out = make_server(tmp_path, auto_approve=True)
    assert server.approve("command", "ls", "ls") is True
    wait_for(out, "approval_auto")

    thread, result = approve_in_thread(server, "delete")
    req = wait_for(out, "approval_request")
    assert req["kind"] == "delete"
    respond(server, req, "decline")
    thread.join(5)
    assert result == [False]


def test_cancel_declines_pending_approval(tmp_path):
    server, out = make_server(tmp_path)
    thread, result = approve_in_thread(server, "write")
    wait_for(out, "approval_request")
    server.handle({"type": "cancel"})
    thread.join(5)
    assert result == [False]


def test_cancel_stops_run_between_rounds(tmp_path):
    release = threading.Event()

    def run_fn(text, model, **kw):
        hooks = kw["hooks"]
        hooks.on_frontier(1)
        release.wait(5)
        hooks.on_frontier(2)
        return FakeResult("late", FakeStats())

    server, out = make_server(tmp_path, run_fn)
    server.handle({"type": "user_message", "text": "go"})
    wait_for(out, "frontier_round")
    server.handle({"type": "cancel"})
    release.set()
    wait_for(out, "run_cancelled")
    assert of_type(out, "run_finished") == []
    assert len(of_type(out, "frontier_round")) == 1


def test_second_message_while_busy_is_queued(tmp_path):
    release = threading.Event()

    def run_fn(text, model, **kw):
        release.wait(5)
        return FakeResult("ok", FakeStats())

    server, out = make_server(tmp_path, run_fn)
    server.handle({"type": "user_message", "text": "one"})
    server.handle({"type": "user_message", "text": "two"})
    queued = wait_for(out, "queue")
    assert queued["items"] == ["two"]
    release.set()
    assert wait_for(out, "run_finished")["answer"] == "ok"


def test_run_errors_become_error_events(tmp_path):
    def fails(text, model, **kw):
        raise OrchestrationError("did not converge", None)

    server, out = make_server(tmp_path, fails)
    try:
        server.handle({"type": "user_message", "text": "x"})
        assert "did not converge" in wait_for(out, "error")["message"]
    finally:
        server.close()

    other = tmp_path / "b"
    other.mkdir()

    def boom(text, model, **kw):
        raise RuntimeError("boom")

    server2, out2 = make_server(other, boom)
    try:
        server2.handle({"type": "user_message", "text": "x"})
        assert wait_for(out2, "error")["message"] == "RuntimeError: boom"
    finally:
        server2.close()


def test_bad_messages_report_errors(tmp_path):
    cases = [
        ({"type": "user_message", "text": "   "}, "Empty"),
        ({"type": "nope"}, "Unknown message type"),
        ({"type": "approval_response", "id": "x", "decision": "maybe"}, "Unknown decision"),
        ({"type": "approval_response", "id": "missing", "decision": "approve"}, "No pending approval")
    ]
    server, out = make_server(tmp_path)
    for i, (message, expected) in enumerate(cases, start=1):
        assert server.handle(message) is True
        assert expected in wait_for(out, "error", count=i)["message"]


def test_settings_messages(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "set_auto", "enabled": True})
    assert wait_for(out, "settings")["auto_approve"] is True
    assert server.auto_approve is True
    server.handle({"type": "set_model", "model": "other-model"})
    assert wait_for(out, "settings", count=2)["model"] == "other-model"
    assert server.handle({"type": "shutdown"}) is False


def test_serve_forever_reads_lines_until_shutdown(tmp_path):
    inp = io.StringIO("not json\n[1, 2]\n\n" + '{"type": "set_auto", "enabled": true}\n' + '{"type": "shutdown"}\n' + '{"type": "set_auto", "enabled": false}\n')
    server, out = make_server(tmp_path, inp=inp)
    server.serve_forever()

    got = events(out)
    assert [e["type"] for e in got] == ["ready", "error", "error", "settings"]
    assert got[0]["model"] == "test-model"
    assert "Invalid JSON" in got[1]["message"]
    assert "JSON object" in got[2]["message"]
    assert got[3]["auto_approve"] is True
