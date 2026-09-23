"""Tasks in the background, the prompt always live -- like Claude Code.

In an interactive terminal, a task runs on a worker thread while the prompt
stays usable: slash commands work mid-task, `/summary` says who is doing
what, new messages queue up behind the current task, `/tell` adds a note to
the running one, and `/stop` cancels it.

The terminal belongs to prompt_toolkit for the whole session: output from
the worker goes through `patch_stdout` and lands above the prompt, and live
status is the prompt's bottom toolbar. So nothing here draws with Rich's
Live/status (two libraries moving the cursor at once garbles the screen),
and local-model output is not streamed line by line -- it's summarized in
the toolbar (model, what it's writing, tokens, speed) and each finished
step prints one line. Approvals are handed from the worker to the main
thread, which owns the keyboard.

One-off `localforge run` and every non-terminal path stay synchronous.
"""

from __future__ import annotations

import collections
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from collections.abc import Callable

# Just a small spinner: it has to keep moving so a slow step doesn't look
# like a hang, but the line stays plain text in the session's own colors.
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_ASCII = "|/-\\"
# After this long with no new step, the toolbar says how long it's been --
# "still working" beats a frozen-looking line.
QUIET_AFTER_SECONDS = 20
# Lines of the local model's output kept under the status line, so you can
# see the code taking shape without it scrolling the session away.
PREVIEW_LINES = 3


@dataclass
class Approval:
    kind: str
    title: str
    detail: str = ""  # the diff, command or listing the user is being asked about
    answered: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False
    always: bool = False  # "(a)lways": allow this kind for the rest of the session


@dataclass
class TaskState:
    """What's happening right now; read by the toolbar and /summary."""

    task: str = ""
    started: float = 0.0
    orchestrator: str = ""
    phase: str = ""
    local_model: str = ""
    local_what: str = ""
    local_tokens: int = 0  # the delegation running right now
    local_total: int = 0  # every local token this task has produced
    local_started: float = 0.0
    answer_words: int = 0
    answer_chars: int = 0

    def tokens(self) -> int:
        """Tokens produced so far: local models exactly, the orchestrator's
        own answer approximated from its length (~4 characters a token)."""
        return self.local_total + self.answer_chars // 4
    downloading: str = ""
    waiting_for: str = ""  # e.g. "usage limit resets in 2h 14m"
    last_event: float = 0.0  # when the last step/token landed, for the "quiet for Xs" note
    preview: collections.deque = field(default_factory=lambda: collections.deque(maxlen=PREVIEW_LINES))
    partial: str = ""  # the line the local model is part-way through writing
    todos: list[dict] = field(default_factory=list)
    steps: collections.deque = field(default_factory=lambda: collections.deque(maxlen=12))


class TaskRunner:
    """Runs tasks one at a time on a worker thread, with a queue behind it."""

    def __init__(self, run_task: Callable[[str], None]):
        self.run_task = run_task
        self.queue: collections.deque[str] = collections.deque()
        self.state = TaskState()
        self.thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.approval: Approval | None = None
        self._notes: list[str] = []
        # Answers "why is this needed?" while a prompt is waiting; set by the
        # CLI, which knows how to explain a request.
        self.question_handler = None
        self._lock = threading.Lock()
        # Set and cleared under _lock, together with the queue check -- asking
        # thread.is_alive() instead would strand a task submitted in the
        # instant the worker decides the queue is empty and exits.
        self._running = False

    # --- control (main thread) -------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._running

    @property
    def waiting(self) -> bool:
        """Paused on a usage limit: the task is alive but doing nothing, so
        settings commands (/model) are safe and useful right now."""
        return bool(self.state.waiting_for)

    def submit(self, task: str) -> int:
        """Start `task`, or queue it behind the running one. Returns its
        position in the queue (0 = started now)."""
        with self._lock:
            if self._running:
                self.queue.append(task)
                return len(self.queue)
            self._running = True
            # Set the task up here, before the thread starts, so a /summary or
            # /stop in the very next instant sees (and acts on) this task.
            self._begin(task)
            self.thread = threading.Thread(target=self._work, args=(task,), name="localforge-task", daemon=True)
            self.thread.start()
            return 0

    def cancel(self) -> bool:
        """Stop the running task (it notices at its next step). Returns
        whether there was one."""
        if not self.busy:
            return False
        self.cancel_event.set()
        approval = self.approval
        if approval is not None:
            approval.allowed = False
            approval.answered.set()
        return True

    def tell(self, note: str) -> bool:
        """A note for the running task, delivered at the orchestrator's next step."""
        if not self.busy:
            return False
        with self._lock:
            self._notes.append(note)
        return True

    def answer(self, allowed: bool, always: bool = False) -> None:
        approval = self.approval
        if approval is not None:
            approval.allowed = allowed
            approval.always = always
            approval.answered.set()

    def shutdown(self, wait: float = 5.0) -> None:
        self.queue.clear()
        self.cancel()
        if self.thread is not None:
            self.thread.join(timeout=wait)

    # --- worker side ------------------------------------------------------------

    def _begin(self, task: str) -> None:
        self.cancel_event.clear()
        self._notes.clear()
        self.state = TaskState(task=task, started=time.monotonic(), last_event=time.monotonic())

    def note_event(self) -> None:
        """Something happened (a step, a token): resets the quiet timer."""
        self.state.last_event = time.monotonic()

    def note_output(self, chunk: str) -> None:
        """Local-model output for the live preview: complete lines are kept,
        the unfinished one is shown as it grows."""
        state = self.state
        text = state.partial + chunk
        *lines, state.partial = text.split("\n")
        for line in lines:
            state.preview.append(line)
        self.note_event()

    def clear_preview(self) -> None:
        self.state.preview.clear()
        self.state.partial = ""

    def _work(self, task: str) -> None:
        while True:
            try:
                self.run_task(task)
            except BaseException:  # noqa: BLE001 - one task's failure must not kill the queue
                pass
            finally:
                self.approval = None
            with self._lock:
                if not self.queue:
                    self._running = False
                    return
                task = self.queue.popleft()
                self._begin(task)

    def check_cancel(self) -> None:
        """Called from the worker at each step; raising KeyboardInterrupt
        lands in the orchestrator's existing cancel handling (which trims
        the half-finished turn so the conversation stays valid)."""
        if self.cancel_event.is_set():
            raise KeyboardInterrupt

    def take_notes(self) -> list[str]:
        with self._lock:
            notes, self._notes = self._notes, []
        return notes

    def ask(self, kind: str, title: str, detail: str = "") -> Approval:
        """Block the worker until the user answers on the main thread."""
        approval = Approval(kind, title, detail)
        self.approval = approval
        while not approval.answered.wait(0.2):
            if self.cancel_event.is_set():
                break
        self.approval = None
        self.check_cancel()
        return approval

    def ask_question(self, question: str) -> bool:
        """The user typed something other than y/n/a at a prompt: treat it as
        a question about what's being asked. Returns whether it was handled."""
        if self.question_handler is None:
            return False
        self.question_handler(question, self.approval)
        return True

    # --- what the user sees -------------------------------------------------------

    def status_lines(self, unicode: bool | None = None) -> list[tuple[str, str]]:
        """(kind, text) lines shown directly above the input prompt: the
        centered status, then the code preview. Rendered as part of the
        prompt rather than a bar pinned to the bottom of the terminal, so
        the status sits with the work and the typing line stays last.
        """
        lines: list[tuple[str, str]] = []
        status = self.toolbar(unicode)
        if self.approval is not None or self.state.waiting_for:
            lines.append(("attention", status))
        elif self.busy:
            lines.append(("status", status))
        else:
            return []
        lines += [("preview", line) for line in self._preview_lines()]
        return lines

    def toolbar(self, unicode: bool | None = None) -> str:
        """The live status line, in the shape Claude Code uses:

            ⠹ Forging with qwen2.5-coder:7b… (2m 14s · ↓ 3.1k tokens · writing app.py, 41 tok/s)

        Always moving while a task runs, so a slow step never looks like a hang.
        """
        spinner = SPINNER if (_unicode_ok() if unicode is None else unicode) else SPINNER_ASCII
        if self.approval is not None:
            return f" ⏸ waiting for you: {self.approval.title} — answer (y)es / (n)o / (a)lways below"
        if self.state.waiting_for:  # before the busy check: a paused task is still a task
            return f" ⏸ {self.state.waiting_for} — /model to switch and continue now · /stop"
        if not self.busy:
            return " ready — type a task, or / for commands" + (f" · queue: {len(self.queue)}" if self.queue else "")

        s = self.state
        now = time.monotonic()
        icon = spinner[int(now * 8) % len(spinner)]
        # One verb, and the model doing the work right now -- so the line
        # always answers "who is working?" at a glance.
        if s.downloading:
            who, detail = s.orchestrator, s.downloading
        elif s.local_model:
            rate = s.local_tokens / max(now - s.local_started, 0.001)
            who, detail = s.local_model, f"{s.local_what}, {rate:.0f} tok/s"
        elif s.answer_words:
            who, detail = s.orchestrator, f"writing the answer, {s.answer_words} words"
        else:
            who, detail = s.orchestrator, s.phase or "starting"
        quiet = now - (s.last_event or s.started)
        if quiet > QUIET_AFTER_SECONDS:
            detail += f", quiet for {_elapsed(now - quiet)}"
        parts = [_elapsed(s.started), f"↓ {_thousands(s.tokens())} tokens", detail]
        queue = f" · queue: {len(self.queue)}" if self.queue else ""
        status = f"{icon} Forging with {who or 'a model'}… ({' · '.join(parts)}){queue} │ /summary · /stop"
        return _center(status, self.width)

    def _preview_lines(self) -> list[str]:
        """The last few lines the local model has written, so its work is
        visible without filling the session with code."""
        if not (self.state.preview or self.state.partial):
            return []
        lines = [*self.state.preview, self.state.partial] if self.state.partial else list(self.state.preview)
        room = max(self.width - 6, 20)
        return [f"  │ {line[:room]}" for line in lines[-PREVIEW_LINES:]]

    @property
    def width(self) -> int:
        try:
            return max(shutil.get_terminal_size().columns, 40)
        except OSError:
            return 80

    def summary(self) -> list[str]:
        """Plain lines for /summary: instant, from state -- no model call."""
        if not self.busy and not self.state.waiting_for:
            lines = ["Nothing is running."]
        else:
            s = self.state
            lines = [
                f"Task: {s.task}  ({_elapsed(s.started)}, ↓ {_thousands(s.tokens())} tokens)",
                f"Orchestrator: {s.orchestrator} — {s.phase or 'starting'}",
            ]
            if s.waiting_for:
                lines.append(f"Paused: {s.waiting_for} (the task continues by itself; /model to switch now)")
            if s.local_model:
                lines.append(f"Local model: {s.local_model} {s.local_what} — {s.local_tokens} tokens so far")
            elif s.downloading:
                lines.append(s.downloading)
            if self.approval is not None:
                lines.append(f"Waiting for you: {self.approval.title} — (y)es / (n)o / (a)lways")
            if s.todos:
                done = sum(t.get("status") == "completed" for t in s.todos)
                lines.append(f"Plan ({done}/{len(s.todos)} done):")
                marks = {"completed": "[x]", "in_progress": "[~]"}
                lines += [f"  {marks.get(t.get('status'), '[ ]')} {t.get('content', '')}" for t in s.todos]
            if s.steps:
                lines.append("Recent steps:")
                lines += [f"  {step}" for step in s.steps]
        if self.queue:
            lines.append(f"Queued ({len(self.queue)}):")
            lines += [f"  {i}. {task}" for i, task in enumerate(self.queue, 1)]
        return lines


def _center(text: str, width: int) -> str:
    """Centered, because a status line in the middle reads as "working" at a
    glance where a left-hugging one blends into the output above it."""
    return " " * max((width - len(text)) // 2, 1) + text


def _thousands(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def _unicode_ok() -> bool:
    """Emoji only where the terminal can show them."""
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding


def _elapsed(since: float) -> str:
    secs = int(time.monotonic() - since)
    return f"{secs // 60}m {secs % 60:02d}s" if secs >= 60 else f"{secs}s"
