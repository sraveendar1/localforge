"""Regression tests for desktop-app-only bugs found while auditing serve.py
for GUI/CLI parity:

- `/compact` from the desktop app always failed with `NameError:
  _installed_model_names is not defined` -- it referenced a helper that only
  ever existed in cli.py.
- `/why` raised `KeyError` for any actually-pending approval, because
  `approve()` populated `self._pending` but never `self._pending_info`,
  which `handle_why_command()` reads.
- `/auto` with no argument (or a typo) forced auto-approve off instead of
  toggling it, unlike the CLI's own bare `/auto`.
- The actual entry point the desktop app spawns, `serve_stdio()`, called
  `StdioServer(..., stream_output=stream_output)`, but `StdioServer.__init__`
  never declared a `stream_output` parameter at all -- so `localforge serve
  --stdio` (exactly what `desktop/src-tauri/src/lib.rs`'s `start_session`
  runs) raised `TypeError` and exited immediately, before writing a single
  byte of protocol JSON, every time the desktop app opened a folder. No
  existing test caught this because every other test in this file
  constructs `StdioServer` directly and never goes through `serve_stdio()`
  itself -- confirmed live by piping JSON messages into the real
  `localforge serve --stdio` subprocess and watching it crash before the
  fix and answer normally after.
- Bare `/usage` (no argument) always raised `AttributeError`:
  `usage_store.get_all_time_totals()`/`get_previous_session_totals()` don't
  exist -- the real functions are `usage_store.all_time(root)` and
  `usage_store.previous_session(root, session_id)`. Worse than the other
  bugs here: `handle()` has no try/except around dispatch, so this
  exception propagated out of `serve_forever()`'s read loop entirely,
  killing the whole backend process (and therefore the whole desktop
  session) rather than just failing that one command.
- Related: the desktop backend never called `usage_store.record()` at all,
  so even once `/usage` stopped crashing, a GUI session's tasks never
  showed up in a project's previous-session/all-time totals -- only tasks
  run through the CLI's `run` command did (via `cli.py`'s `_record_usage`).
"""

import io
import json
import sys
import threading
import time

from localforge.orchestrator import Conversation, RunStats
from localforge.serve import StdioServer, serve_stdio
from localforge import usage_store


def make_server(tmp_path, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", Conversation())
    kwargs.setdefault("run_fn", lambda *a, **k: None)
    server = StdioServer(tmp_path, "test-model", out=out, scratch_root=scratch, **kwargs)
    return server, out


def events(out):
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def of_type(out, event_type):
    return [e for e in events(out) if e["type"] == event_type]


def test_compact_command_does_not_crash_with_an_empty_conversation(tmp_path):
    server, out = make_server(tmp_path)
    server.handle_compact_command()
    assert of_type(out, "error") == []
    compacted = of_type(out, "compacted")
    assert len(compacted) == 1
    assert compacted[0]["before_messages"] == 0 and compacted[0]["after_messages"] == 0


def test_why_command_reports_a_pending_approval_without_crashing(tmp_path):
    server, out = make_server(tmp_path)

    def approve_soon():
        # Wait for the approval request to actually be registered before resolving it.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not server._pending:
            time.sleep(0.01)
        request_id = next(iter(server._pending))
        server._resolve(request_id, "decline")

    thread = threading.Thread(target=approve_soon, daemon=True)
    thread.start()
    result = []
    approve_thread = threading.Thread(
        target=lambda: result.append(server.approve("command", "Run command", "git push origin main")),
        daemon=True,
    )
    approve_thread.start()
    approve_thread.join(timeout=5)
    thread.join(timeout=5)
    assert result == [False]

    # Now check /why while a *different* approval is pending.
    out.truncate(0)
    out.seek(0)

    def resolve_later():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not server._pending:
            time.sleep(0.01)
        time.sleep(0.05)  # let handle_why_command below observe it first
        request_id = next(iter(server._pending))
        server._resolve(request_id, "decline")

    resolver = threading.Thread(target=resolve_later, daemon=True)
    resolver.start()
    pending_result = []
    pending_thread = threading.Thread(
        target=lambda: pending_result.append(server.approve("command", "Run command", "git push origin main")),
        daemon=True,
    )
    pending_thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not server._pending:
        time.sleep(0.01)
    server.handle_why_command()
    pending_thread.join(timeout=5)
    resolver.join(timeout=5)

    why = of_type(out, "why")
    assert len(why) == 1
    assert of_type(out, "error") == []
    joined = "\n".join(why[0]["lines"])
    assert "Run command" in joined and "git push origin main" in joined


def test_why_command_with_nothing_pending_reports_auto_approve_state(tmp_path):
    server, out = make_server(tmp_path)
    server.handle_why_command()
    why = of_type(out, "why")
    assert why[0]["lines"] == ["Auto-approve is off"]


def test_bare_auto_command_toggles_instead_of_forcing_off(tmp_path):
    server, out = make_server(tmp_path)
    assert server.auto_approve is False

    server.handle_auto_command("")
    assert server.auto_approve is True

    server.handle_auto_command("")
    assert server.auto_approve is False

    server.handle_auto_command("on")
    assert server.auto_approve is True

    server.handle_auto_command("off")
    assert server.auto_approve is False


def test_serve_stdio_the_real_entry_point_does_not_crash_on_startup(tmp_path, monkeypatch):
    """This is the actual function `localforge serve --stdio` calls (see
    cli.py's `serve` command) and the actual function
    desktop/src-tauri/src/lib.rs's `start_session` spawns via that CLI
    command -- as opposed to every other test in this file, which
    constructs StdioServer directly and would not have caught a mismatched
    keyword argument between serve_stdio() and StdioServer.__init__."""
    from localforge import trust

    trust.trust(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type": "shutdown"}\n'))
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    serve_stdio(tmp_path, "test-model", auto_approve=False, stream_output=True)

    lines = [json.loads(l) for l in fake_stdout.getvalue().splitlines() if l.strip()]
    assert lines[0]["type"] == "ready"
    assert lines[0]["stream_output"] is True


def test_bare_usage_command_does_not_crash_and_reports_history(tmp_path):
    server, out = make_server(tmp_path)

    # Nothing recorded yet: no crash, previous_session is None (no earlier
    # session exists), all_time is a zeroed Totals (it always exists).
    server.handle_usage_command("")
    usage_events = of_type(out, "usage")
    assert len(usage_events) == 1
    assert usage_events[0]["previous_session"] is None
    assert usage_events[0]["all_time"]["tasks"] == 0

    # Record a task's usage the way _run_turn does, then check it shows up.
    stats = RunStats(frontier_prompt_tokens=100, frontier_completion_tokens=50, frontier_cost_usd=0.01, local_tokens_generated=40)
    server._record_usage(stats)

    out.truncate(0)
    out.seek(0)
    server.handle_usage_command("")
    usage_events = of_type(out, "usage")
    assert usage_events[0]["all_time"]["tasks"] == 1
    assert usage_events[0]["all_time"]["frontier_prompt_tokens"] == 100
    assert usage_events[0]["all_time"]["local_tokens_generated"] == 40


def test_run_turn_records_usage_to_usage_store(tmp_path):
    """The desktop backend previously never called usage_store.record() at
    all, so a GUI session's completed tasks never contributed to a
    project's usage history."""
    stats = RunStats(frontier_prompt_tokens=7, frontier_completion_tokens=3, frontier_cost_usd=0.001, local_tokens_generated=2)

    class FakeResult:
        answer = "done"

        def __init__(self):
            self.stats = stats

    server, out = make_server(tmp_path, run_fn=lambda *a, **k: FakeResult())
    server._run_turn("do something")

    total = usage_store.all_time(tmp_path)
    assert total.tasks == 1
    assert total.frontier_prompt_tokens == 7
    assert total.local_tokens_generated == 2
