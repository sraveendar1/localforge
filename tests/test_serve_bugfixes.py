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
"""

import io
import json
import threading
import time

from localforge.orchestrator import Conversation
from localforge.serve import StdioServer


def make_server(tmp_path, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", Conversation())
    server = StdioServer(tmp_path, "test-model", out=out, run_fn=lambda *a, **k: None, scratch_root=scratch, **kwargs)
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
