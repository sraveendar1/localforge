"""The project's goal is the user's first real request, kept at once and added to AGENTS.md
with a running progress log as work gets done (shown in the desktop's Overall goal)."""
import io
import json
from datetime import date

from localforge import brief, trust
from localforge.orchestrator import Conversation, RunResult, RunStats
from localforge.serve import StdioServer
from localforge.task_summary import TaskLog


def _server(tmp_path, run_fn=None, auto=True):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    return StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=run_fn or (lambda *a, **k: None),
                       scratch_root=tmp_path / "scratch", conversation=Conversation(), auto_approve=auto), out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


# --- the text helpers ---------------------------------------------------------------

def test_a_greeting_is_not_a_goal_but_a_real_request_is(tmp_path):
    assert not brief.record_goal(tmp_path, "hi")
    assert not brief.record_goal(tmp_path, "/memory")
    assert not brief.record_goal(tmp_path, "thanks!")  # too short to say what is wanted
    assert brief.record_goal(tmp_path, "Build a todo app in React with a login page")
    assert brief.recorded_goal(tmp_path) == "Build a todo app in React with a login page."


def test_only_the_first_request_is_the_goal(tmp_path):
    brief.record_goal(tmp_path, "Build a todo app in React")
    assert not brief.record_goal(tmp_path, "Now add dark mode to everything")
    assert brief.recorded_goal(tmp_path) == "Build a todo app in React."


def test_a_long_request_is_kept_to_one_tidy_paragraph(tmp_path):
    brief.record_goal(tmp_path, "word " * 400)
    goal = brief.recorded_goal(tmp_path)
    assert len(goal) <= brief.MAX_GOAL_CHARS and "\n" not in goal


def test_with_goal_creates_or_adds_the_section_without_touching_the_rest():
    assert brief.with_goal("", "Build a thing here").startswith("# Project brief\n\n## What this project is\n\nBuild a thing here")
    text = "# X\n\n## How it's built\nstuff\n"
    out = brief.with_goal(text, "Build a thing here")
    assert out.index("What this project is") < out.index("How it's built") and "stuff" in out
    assert brief.with_goal(out, "Something else entirely") == out  # an existing summary is never replaced


def test_progress_is_appended_not_duplicated_and_capped():
    text = brief.with_goal("", "Build a thing here")
    text = brief.with_progress(text, "add login (created a.py)", date(2026, 10, 1))
    text = brief.with_progress(text, "add login (created a.py)", date(2026, 10, 2))  # same line again
    assert brief.progress_entries(text) == ["2026-10-01 — add login (created a.py)"]
    for i in range(60):
        text = brief.with_progress(text, f"step {i}", date(2026, 10, 3))
    assert len(brief.progress_entries(text)) == brief.MAX_PROGRESS_ENTRIES
    assert brief.progress_entries(text)[-1].endswith("step 59")
    assert "What this project is" in text  # the rest of the file survives


def test_progress_line_names_what_changed():
    line = brief.progress_line("add a login page", [{"action": "Created", "path": "login.py"}, {"action": "Updated", "path": "app.py"}])
    assert line == "add a login page (created login.py, updated app.py)"
    many = [{"action": "Created", "path": f"f{i}.py"} for i in range(7)]
    assert "+3 more" in brief.progress_line("big change", many)


# --- the desktop backend ----------------------------------------------------------------

def _run_creating(path):
    def run(text, model, **kw):
        log = TaskLog()
        log.record("delegate_coding_task", path, "ok", f"Created {path} (+1 -0, 1 lines).", "m")
        stats = RunStats()
        stats.log = log
        # the real run would write the file; the summary is what matters here
        (kw["workspace"].root / path).write_text("print('hi')\n")
        return RunResult(answer="done", stats=stats)
    return run


def test_the_first_message_is_recorded_as_the_goal_and_shown(tmp_path):
    server, out = _server(tmp_path)
    server.handle({"type": "user_message", "text": "Build a todo app in React with a login page"})
    server._worker.join(5)
    assert brief.recorded_goal(tmp_path) == "Build a todo app in React with a login page."
    assert _events(out, "memory")[0]["goal"] == "Build a todo app in React with a login page."


def test_a_greeting_first_does_not_take_the_goal(tmp_path):
    server, _ = _server(tmp_path)
    server.handle({"type": "user_message", "text": "hello"})
    server._worker.join(5)
    assert brief.recorded_goal(tmp_path) == ""


def test_a_finished_task_that_changed_files_writes_agents_md_with_goal_and_progress(tmp_path):
    server, out = _server(tmp_path, run_fn=_run_creating("login.py"))
    server.handle({"type": "user_message", "text": "Build a todo app in React with a login page"})
    server._worker.join(10)
    server._brief_thread.join(10)
    text = (tmp_path / "AGENTS.md").read_text()
    assert "## What this project is" in text and "Build a todo app in React with a login page" in text
    assert "## Progress log" in text and "created login.py" in text
    last = _events(out, "memory")[-1]
    assert last["progress"] and "login.py" in last["progress"][-1]


def test_a_second_task_adds_a_second_line(tmp_path):
    server, _ = _server(tmp_path, run_fn=_run_creating("login.py"))
    server.handle({"type": "user_message", "text": "Build a todo app in React with a login page"})
    server._worker.join(10)
    server._brief_thread.join(10)
    server.run_fn = _run_creating("todo.py")
    server.handle({"type": "user_message", "text": "add the todo list view"})
    server._worker.join(10)
    server._brief_thread.join(10)
    entries = brief.progress_entries((tmp_path / "AGENTS.md").read_text())
    assert len(entries) == 2 and "todo.py" in entries[1]


def test_a_chat_answer_with_no_changes_does_not_touch_agents_md(tmp_path):
    server, _ = _server(tmp_path, run_fn=lambda *a, **k: RunResult(answer="because", stats=RunStats()))
    server.handle({"type": "user_message", "text": "why is the sky blue anyway"})
    server._worker.join(5)
    assert not (tmp_path / "AGENTS.md").exists()
    assert brief.recorded_goal(tmp_path)  # but the goal was still kept


def test_the_goal_panel_prefers_agents_md_once_it_has_a_summary(tmp_path):
    brief.record_goal(tmp_path, "Build a todo app in React with a login page")
    (tmp_path / "AGENTS.md").write_text("# T\n\n## What this project is\n\nA small todo app.\n")
    assert brief.project_goal(tmp_path) == "A small todo app."


def test_declining_the_agents_md_change_is_respected_for_the_session(tmp_path):
    server, out = _server(tmp_path, run_fn=_run_creating("login.py"), auto=False)
    server.approve = lambda kind, title, detail: False
    server.handle({"type": "user_message", "text": "Build a todo app in React with a login page"})
    server._worker.join(10)
    server._brief_thread.join(10)
    assert not (tmp_path / "AGENTS.md").exists()
    assert server._agents_md_declined is True


def test_a_pending_agents_md_approval_does_not_hold_up_the_next_queued_task(tmp_path):
    import threading

    server, out = _server(tmp_path, run_fn=_run_creating("login.py"), auto=False)  # nobody answers the approval
    server.handle({"type": "user_message", "text": "Build a todo app in React with a login page"})
    server._worker.join(10)
    assert not server.busy, "the worker must be free even though the AGENTS.md approval is still open"
    import time
    for _ in range(60):
        if _events(out, "approval_request"):
            break
        time.sleep(0.05)
    assert _events(out, "approval_request")
    server._decline_all_pending()
    server._brief_thread.join(5)


def test_an_older_project_shows_the_goal_from_its_last_session_note(tmp_path, monkeypatch):
    from localforge import memory

    monkeypatch.setattr(memory, "load", lambda root: "## Goal\nTrack tech stocks and suggest debit spreads.\n\n## Decisions and constraints\n* Python\n")
    assert brief.project_goal(tmp_path) == "Track tech stocks and suggest debit spreads."
    brief.record_goal(tmp_path, "Build a stock watchlist dashboard")  # a recorded goal wins over the old note
    assert brief.project_goal(tmp_path) == "Build a stock watchlist dashboard."
