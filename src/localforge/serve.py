"""JSON-lines server over stdin/stdout that lets the desktop app drive the orchestrator.

Each stdin line is one JSON message from the app; each stdout line is one JSON
event to the app. Every object has a "type" key. stdout carries only protocol
JSON; everything else goes to stderr.
"""
from __future__ import annotations

import dataclasses, json, sys, threading, uuid
from pathlib import Path
from typing import Callable, IO
from localforge import memory
from localforge.orchestrator import Conversation, OrchestrationError
from localforge.orchestrator import run as run_orchestrator
from localforge.scratchpad import Scratchpad
from localforge.tools import ActivityHooks
from localforge.workspace import Workspace

class Cancelled(Exception):
    """Raised from on_frontier to stop a run between rounds."""

def _jsonable_stats(stats) -> dict | None:
    if stats is None:
        return None
    if dataclasses.is_dataclass(stats):
        return dataclasses.asdict(stats)
    return dict(vars(stats))

class StdioServer:
    def __init__(self, root: Path, frontier_model: str, cli_provider: str | None = None,
                 out: IO[str] | None = None, inp: IO[str] | None = None,
                 run_fn: Callable = run_orchestrator, auto_approve: bool = False,
                 conversation: Conversation | None = None, scratch_root: Path | None = None):
        self.root = Path(root).resolve()
        self.frontier_model = frontier_model
        self.cli_provider = cli_provider
        self.out = out or sys.stdout
        self.inp = inp or sys.stdin
        self.run_fn = run_fn
        self.auto_approve = auto_approve
        self.always_allow: set[str] = set()
        if scratch_root is not None:
            self.scratch_root = scratch_root
            self._scratchpad = None
        else:
            self._scratchpad = Scratchpad(self.root)
            self._scratchpad.ensure()
            self.scratch_root = self._scratchpad.root
        self.conversation = conversation if conversation is not None else Conversation(memory=memory.load(self.root), facts=memory.facts_for_prompt(self.root))
        self._write_lock = threading.Lock()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._pending: dict[str, tuple[threading.Event, list[str]]] = {}
        self._pending_lock = threading.Lock()

    def emit(self, event_type: str, **fields) -> None:
        with self._write_lock:
            event = {"type": event_type, **fields}
            self.out.write(json.dumps(event, default=str) + "\n")
            self.out.flush()

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def approve(self, kind, title, detail) -> bool:
        if self._cancel.is_set():
            return False
        if kind != "delete" and (self.auto_approve or kind in self.always_allow):
            self.emit("approval_auto", kind=kind, title=title, detail=detail)
            return True
        request_id = uuid.uuid4().hex
        event = threading.Event()
        holder: list[str] = ["decline"]
        with self._pending_lock:
            self._pending[request_id] = (event, holder)
        self.emit("approval_request", id=request_id, kind=kind, title=title, detail=detail)
        event.wait()
        with self._pending_lock:
            self._pending.pop(request_id, None)
        if self._cancel.is_set():
            return False
        if holder[0] == "always":
            if kind != "delete":
                self.always_allow.add(kind)
            return True
        return holder[0] == "approve"

    def _resolve(self, request_id: str, decision: str) -> bool:
        with self._pending_lock:
            entry = self._pending.get(request_id)
        if entry is None:
            return False
        entry[1][0] = decision
        entry[0].set()
        return True

    def _decline_all_pending(self):
        with self._pending_lock:
            for event, holder in self._pending.values():
                holder[0] = "decline"
                event.set()

    def _on_frontier(self, round_number: int) -> None:
        # run() calls this before every round, so raising here is how a cancel
        # stops it between rounds.
        if self._cancel.is_set():
            raise Cancelled()
        self.emit("frontier_round", round=round_number)

    def _hooks(self) -> ActivityHooks:
        return ActivityHooks(
            on_frontier=self._on_frontier,
            on_delegate=lambda modality, entry: self.emit("delegate_started", modality=modality, model=entry.name),
            on_token=lambda text: self.emit("delegate_token", text=text),
            on_done=lambda modality, entry, tokens, seconds: self.emit("delegate_finished", modality=modality, model=entry.name, tokens=tokens, seconds=round(seconds, 2)),
            on_pull=lambda model, event: self.emit("model_pull", model=model, progress=event),
            on_tool=lambda name, summary: self.emit("tool_call_started", name=name, summary=summary),
            on_tool_result=lambda name, result: self.emit("tool_call_finished", name=name, result=result),
            on_todos=lambda todos: self.emit("todos_updated", todos=todos),
            on_answer_text=lambda text: self.emit("text_delta", text=text)
        )

    def _run_turn(self, text: str) -> None:
        workspace = Workspace(self.root, approver=self.approve, scratch=self.scratch_root)
        try:
            result = self.run_fn(text, self.frontier_model, cli_provider=self.cli_provider, hooks=self._hooks(), conversation=self.conversation, workspace=workspace)
            self.emit("run_finished", answer=result.answer, stats=_jsonable_stats(result.stats))
        except Cancelled:
            self.emit("run_cancelled")
        except OrchestrationError as exc:
            self.emit("error", message=str(exc), stats=_jsonable_stats(getattr(exc, "stats", None)))
        except Exception as exc:
            self.emit("error", message=f"{type(exc).__name__}: {exc}")

    def handle(self, message: dict) -> bool:
        message_type = message.get("type")
        if message_type == "user_message":
            text = str(message.get("text", "")).strip()
            if not text:
                self.emit("error", message="Empty message.")
            elif self.busy:
                self.emit("error", message="A run is already in progress.")
            else:
                self._cancel.clear()
                self.emit("run_started", text=text)
                self._worker = threading.Thread(target=self._run_turn, args=(text,), daemon=True)
                self._worker.start()
        elif message_type == "approval_response":
            decision = message.get("decision")
            if decision not in ("approve", "decline", "always"):
                self.emit("error", message=f"Unknown decision: {decision}")
            elif not self._resolve(str(message.get("id")), decision):
                self.emit("error", message=f"No pending approval with id {message.get('id')}")
        elif message_type == "cancel":
            self._cancel.set()
            self._decline_all_pending()
        elif message_type == "set_auto":
            self.auto_approve = bool(message.get("enabled"))
            self.emit("settings", auto_approve=self.auto_approve, model=self.frontier_model)
        elif message_type == "set_model":
            if self.busy:
                self.emit("error", message="Can't change the model during a run.")
            else:
                model = str(message.get("model") or "").strip()
                if not model:
                    self.emit("error", message="Model cannot be empty.")
                else:
                    self.frontier_model = model
                    self.cli_provider = message.get("cli_provider")
                    self.emit("settings", auto_approve=self.auto_approve, model=self.frontier_model)
        elif message_type == "shutdown":
            return False
        else:
            self.emit("error", message=f"Unknown message type: {message_type!r}")
        return True

    def serve_forever(self) -> None:
        self.emit("ready", root=str(self.root), model=self.frontier_model, auto_approve=self.auto_approve, scratchpad=str(self.scratch_root))
        try:
            for line in self.inp:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    self.emit("error", message=f"Invalid JSON: {line}")
                    continue
                if not isinstance(msg, dict):
                    self.emit("error", message="Each message must be a JSON object.")
                    continue
                if not self.handle(msg):
                    break
        finally:
            self.close()

    def close(self) -> None:
        self._cancel.set()
        self._decline_all_pending()
        if self._worker is not None:
            self._worker.join(timeout=5)
        if self._scratchpad is not None:
            self._scratchpad.remove()
            self._scratchpad = None

def serve_stdio(root: Path, frontier_model: str, cli_provider: str | None = None, auto_approve: bool = False) -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        server = StdioServer(root, frontier_model, cli_provider, out=real_stdout, inp=sys.stdin, auto_approve=auto_approve)
        server.serve_forever()
    finally:
        sys.stdout = real_stdout
