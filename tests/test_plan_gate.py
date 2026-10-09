"""Plan first, and steering a running task.

Reported: with Auto-approve on, a long task ran with no visible plan, and a message sent while it
ran ("Lets plan first before execution") just queued behind it. Now a task that needs changes ends
with a proposed plan before anything is built -- whatever Auto-approve says -- and a short message
sent mid-run reaches the running task as a note."""
import io
import json
import threading
from unittest.mock import MagicMock, patch

import localforge.orchestrator as orch
import localforge.serve as serve_module
from localforge import trust
from localforge.hardware import HardwareProfile
from localforge.orchestrator import Conversation, RunResult, RunStats
from localforge.serve import StdioServer, classify_midrun, plan_steps

PLAN = (
    "## Proposed plan\n**Goal:** add hello\n**Steps:**\n1. Create `hello.py` with a greeting\n"
    "2. Add a test\n**Checks:** pytest\n**Needs your decision:** none"
)


def _tool_reply(name, args="{}"):
    msg = MagicMock()
    msg.tool_calls = [MagicMock(id="c", function=MagicMock(arguments=args))]
    msg.tool_calls[0].function.name = name
    msg.model_dump.return_value = {"role": "assistant", "content": None}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


def _final(text):
    msg = MagicMock(tool_calls=None, content=text)
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


class Recorder:
    def __init__(self):
        self.calls, self.catalog, self.local_tokens_generated = [], [], 0

    def dispatch(self, name, args, on_delegate=None):
        self.calls.append(name)
        return "ok"


def _run(replies, conv, gate=True):
    replies = iter(replies)
    rec = Recorder()
    with (
        patch.object(orch, "completion", side_effect=lambda **kw: next(replies)),
        patch.object(orch, "Dispatcher", lambda *a, **kw: rec),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        result = orch.run("add hello", "claude-opus-5", hardware=HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[]), conversation=conv, plan_gate=gate)
    return result, rec


def test_a_building_tool_is_refused_until_the_plan_is_approved():
    conv = Conversation()
    result, rec = _run([_tool_reply("delegate_coding_task"), _final(PLAN)], conv)
    assert rec.calls == []  # nothing ran
    assert result.stats.plan_pending is True
    assert result.answer.startswith("## Proposed plan")
    assert "Plan first" in conv.messages[0]["content"]


def test_reading_is_allowed_while_planning():
    result, rec = _run([_tool_reply("read_file"), _final(PLAN)], Conversation())
    assert rec.calls == ["read_file"] and result.stats.plan_pending


def test_a_plain_answer_is_not_a_plan():
    result, _ = _run([_final("It prints a greeting.")], Conversation())
    assert result.stats.plan_pending is False


def test_an_approved_plan_builds():
    conv = Conversation(plan_approved=True)
    result, rec = _run([_tool_reply("delegate_coding_task"), _final("Done.")], conv)
    assert rec.calls == ["delegate_coding_task"] and not result.stats.plan_pending
    assert "Plan first" not in conv.messages[0]["content"]


def test_a_model_that_keeps_reaching_for_build_tools_gets_its_plan_shown_anyway():
    replies = [_tool_reply("edit_file") for _ in range(orch.PLAN_GATE_MAX_HITS)]
    result, rec = _run(replies, Conversation())
    assert rec.calls == [] and result.stats.plan_pending
    assert result.answer.startswith("## Proposed plan")


def test_without_the_gate_nothing_changes():
    result, rec = _run([_tool_reply("delegate_coding_task"), _final("Done.")], Conversation(), gate=False)
    assert rec.calls == ["delegate_coding_task"] and not result.stats.plan_pending


def test_plan_steps_are_read_out_of_the_plan():
    steps = plan_steps(PLAN)
    assert [s["content"] for s in steps] == ["Create hello.py with a greeting", "Add a test"]
    assert steps[0]["status"] == "in_progress" and steps[1]["status"] == "pending"


def test_midrun_messages_are_classified():
    assert classify_midrun("stop") == "stop"
    assert classify_midrun("Lets plan first before execution") == "note"
    assert classify_midrun("why did you pick that file?") == "note"
    assert classify_midrun("later: add a README") == "queue"
    assert classify_midrun("do this too " + "word " * 50) == "queue"
    assert classify_midrun("look at this", has_image=True) == "queue"


# --- the desktop server -----------------------------------------------------------


def _server(tmp_path, run_fn, auto=True):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    return StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=run_fn, auto_approve=auto,
                       scratch_root=tmp_path / "scratch", conversation=Conversation()), out


def _events(out, kind=None):
    ev = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    return [e for e in ev if kind is None or e["type"] == kind]


def _wait(server):
    if server._worker:
        server._worker.join(5)
    if getattr(server, "_brief_thread", None):
        server._brief_thread.join(5)


def test_plan_then_approve_then_build_even_with_auto_approve(tmp_path):
    seen = []

    def run_fn(text, model, **kw):
        conv = kw["conversation"]
        seen.append((text, kw.get("plan_gate"), conv.plan_approved))
        stats = RunStats()
        stats.plan_pending = not conv.plan_approved
        return RunResult(answer=PLAN if stats.plan_pending else "Built it.", stats=stats)

    server, out = _server(tmp_path, run_fn, auto=True)
    server.handle({"type": "user_message", "text": "build me a hello script"})
    _wait(server)
    assert _events(out, "run_finished")[-1]["plan_pending"] is True and server._plan_pending

    server.handle({"type": "user_message", "text": "go ahead"})
    _wait(server)
    assert [e for e in _events(out, "plan_approved")]
    assert seen[1][0] == serve_module.APPROVED_TEXT and seen[1][2] is True
    todos = _events(out, "todos_updated")[0]["todos"]
    assert [t["content"] for t in todos] == ["Create hello.py with a greeting", "Add a test"]
    assert _events(out, "run_finished")[-1]["plan_pending"] is False
    assert server.conversation.plan_approved is False  # the next request is planned afresh


def test_feedback_on_a_plan_is_a_new_planning_turn_not_an_approval(tmp_path):
    seen = []

    def run_fn(text, model, **kw):
        seen.append((text, kw["conversation"].plan_approved))
        stats = RunStats()
        stats.plan_pending = True
        return RunResult(answer=PLAN, stats=stats)

    server, out = _server(tmp_path, run_fn)
    server.handle({"type": "user_message", "text": "build me a hello script"})
    _wait(server)
    server.handle({"type": "user_message", "text": "use argparse instead"})
    _wait(server)
    assert seen[1] == ("use argparse instead", False)


def test_the_approve_button_and_cancel(tmp_path):
    def run_fn(text, model, **kw):
        stats = RunStats()
        stats.plan_pending = not kw["conversation"].plan_approved
        return RunResult(answer=PLAN, stats=stats)

    server, out = _server(tmp_path, run_fn)
    server.handle({"type": "user_message", "text": "build me a hello script"})
    _wait(server)
    server.handle({"type": "plan_response", "action": "cancel"})
    assert _events(out, "plan_cancelled") and not server._plan_pending

    server.handle({"type": "user_message", "text": "build me a hello script"})
    _wait(server)
    server.handle({"type": "plan_response", "action": "approve"})
    _wait(server)
    assert _events(out, "plan_approved")


def test_plan_off_runs_without_the_gate(tmp_path):
    kwargs = []
    server, out = _server(tmp_path, lambda text, model, **kw: kwargs.append(kw) or RunResult(answer="ok", stats=RunStats()))
    server.handle({"type": "user_message", "text": "/plan off"})
    server.handle({"type": "user_message", "text": "do a thing"})
    _wait(server)
    assert "plan_gate" not in kwargs[0]
    assert _events(out, "plan_mode")[-1]["enabled"] is False


def test_a_message_sent_mid_run_reaches_the_running_task(tmp_path):
    started, release, delivered = threading.Event(), threading.Event(), []

    def run_fn(text, model, **kw):
        started.set()
        release.wait(5)
        delivered.extend(kw["hooks"].poll_notes())
        return RunResult(answer="ok", stats=RunStats())

    server, out = _server(tmp_path, run_fn)
    server.handle({"type": "user_message", "text": "start"})
    started.wait(5)
    server.handle({"type": "user_message", "text": "Lets plan first before execution"})
    assert not server._queue  # not queued behind the task
    assert _events(out, "note_added")
    release.set()
    _wait(server)
    assert delivered == ["Lets plan first before execution"]
    assert _events(out, "note_delivered")


def test_stop_mid_run_and_queued_items_can_be_sent_to_the_running_task(tmp_path):
    started, release = threading.Event(), threading.Event()

    def run_fn(text, model, **kw):
        started.set()
        release.wait(5)
        return RunResult(answer="ok", stats=RunStats())

    server, out = _server(tmp_path, run_fn)
    server.handle({"type": "user_message", "text": "start"})
    started.wait(5)
    server.handle({"type": "user_message", "text": "later: add a README"})
    assert [t for t, _ in server._queue] == ["add a README"]
    server.handle({"type": "queue_to_note", "index": 0})
    assert not server._queue and server._notes == ["add a README"]
    server.handle({"type": "user_message", "text": "stop"})
    assert server._cancel.is_set()
    release.set()
    _wait(server)
