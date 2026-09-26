"""JSON-lines server over stdin/stdout that lets the desktop app drive the orchestrator.

Each stdin line is one JSON message from the app; each stdout line is one JSON
event to the app. Every object has a "type" key. stdout carries only protocol
JSON; everything else goes to stderr.
"""
from __future__ import annotations

import dataclasses, json, shutil, sys, threading, uuid
from pathlib import Path
from typing import Callable, IO, List, Optional
from localforge import brief, memory, trust
from localforge.orchestrator import Conversation, OrchestrationError
from localforge.orchestrator import run as run_orchestrator
from localforge.scratchpad import Scratchpad
from localforge.tools import ActivityHooks, Dispatcher
from localforge.workspace import Workspace
from localforge import usage_store

import psutil
from localforge.hardware import detect_hardware
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import load_catalog, recommendations

class Cancelled(Exception):
    """Raised from on_frontier to stop a run between rounds."""

def _jsonable_stats(stats) -> dict | None:
    if stats is None:
        return None
    if dataclasses.is_dataclass(stats):
        return dataclasses.asdict(stats)
    return dict(vars(stats))

MAX_IMAGE_BASE64_CHARS = 30_000_000  # ~22 MB decoded; well above any real screenshot, a guard against a mistake

def _parse_image(raw) -> dict | None:
    """The GUI's `user_message.image` field (`{"mime_type", "data"}`,
    `data` base64) validated into the shape content_blocks.user_content()
    expects, or None -- never raises, since a malformed/missing image
    should just mean "no image", not fail the whole message."""
    if not isinstance(raw, dict):
        return None
    mime_type = str(raw.get("mime_type", ""))
    data = str(raw.get("data", ""))
    if not mime_type.startswith("image/") or not data or len(data) > MAX_IMAGE_BASE64_CHARS:
        return None
    return {"mime_type": mime_type, "data": data}

def _models():
    hw = detect_hardware()
    recs = recommendations(hw)
    models = []
    for modality, entry in recs.items():
        if entry is not None:
            models.append({'modality': modality, 'model': entry.model_dump()})
    return models

def _installed():
    ollama = OllamaBackend()
    try:
        installed = ollama.list_installed()
    except Exception:
        installed = []
    return [{'model': model['name'], 'status': 'installed'} for model in installed]

def _installed_model_names(ollama: OllamaBackend) -> set[str]:
    try:
        return {m["name"] for m in ollama.list_installed()}
    except Exception:
        return set()

def _project_goal(root: Path) -> str:
    """AGENTS.md's own "What this project is" section (see cli.py's
    `_print_project_summary`, which surfaces the same text at the start of
    a terminal session) -- empty if there's no AGENTS.md yet, or one
    without that section."""
    try:
        return brief.section(brief.existing_brief(root), ("what this project is",))
    except Exception:
        return ""

def _catalog():
    catalog = load_catalog()
    return [{'name': entry.name, 'modality': entry.modality, 'runtime': entry.runtime, 'min_vram_gb': entry.min_vram_gb, 'min_ram_gb': entry.min_ram_gb, 'disk_gb': entry.disk_gb, 'quality_tier': entry.quality_tier} for entry in catalog]

def _doctor():
    checks = []
    ollama_path = shutil.which('ollama')
    checks.append({'name': 'Ollama installed', 'ok': ollama_path is not None, 'detail': ollama_path or 'Ollama not installed. Install it from https://ollama.com.'})
    try:
        hardware = detect_hardware()
        checks.append({'name': 'Hardware detected', 'ok': True, 'detail': f'OS: {hardware.os} ({hardware.arch}), CPU cores: {hardware.cpu_cores}, RAM: {hardware.ram_gb} GB, Free disk: {hardware.free_disk_gb} GB.'})
    except Exception as e:
        checks.append({'name': 'Hardware detection', 'ok': False, 'detail': str(e)})
    return checks

def _hardware():
    return detect_hardware().model_dump()

NOTE_PREFIX = "Note from the user: "

class StdioServer:
    def __init__(self, root: Path, frontier_model: str, cli_provider: str | None = None,
                 out: IO[str] | None = None, inp: IO[str] | None = None,
                 run_fn: Callable = run_orchestrator, auto_approve: bool = False,
                 conversation: Conversation | None = None, scratch_root: Path | None = None,
                 stream_output: bool = False):
        self.root = Path(root).resolve()
        self.frontier_model = frontier_model
        self.cli_provider = cli_provider
        self.out = out or sys.stdout
        self.inp = inp or sys.stdin
        self.run_fn = run_fn
        self.session_id = uuid.uuid4().hex[:12]  # for usage_store, mirrors cli.py's _SessionState.id
        self.auto_approve = auto_approve
        self.always_allow: set[str] = set()
        self.stream_output = stream_output
        if scratch_root is not None:
            self.scratch_root = scratch_root
            self._scratchpad = None
        else:
            self._scratchpad = Scratchpad(self.root)
            self._scratchpad.ensure()
            self.scratch_root = self._scratchpad.root
        self.conversation = conversation if conversation is not None else Conversation(memory=memory.load(self.root), facts=memory.facts_for_prompt(self.root))
        # The terminal REPL asks this interactively before a session ever
        # starts (trust.py's decide()/apply_choice() were written to be
        # shared with "a native desktop front end" -- see its docstring --
        # but nothing actually called them here, so the desktop app just
        # failed outright with "run localforge there once interactively",
        # forcing a trip to a terminal for every new folder). Here it's
        # resolved over the protocol instead: serve_forever() tells the
        # frontend once at startup, handle() answers it, and everything
        # else is a no-op until it's settled.
        self.trusted = trust.is_trusted(self.root)
        self._write_lock = threading.Lock()
        self._cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._pending: dict[str, tuple[threading.Event, list[str]]] = {}
        self._pending_lock = threading.Lock()
        self._hardware: HardwareProfile | None = None
        self._todos: list[dict] = []
        self._queue: List[tuple[str, Optional[dict]]] = []  # (text, image) pairs
        self._queue_lock = threading.Lock()  # Lock for the queue
        self._notes: list[str] = []  # New notes attribute
        self._notes_lock = threading.Lock()  # Lock for the notes attribute

        # Add _pending_info attribute
        self._pending_info: dict[str, tuple[str, str, str]] = {}

    def emit(self, event_type: str, **fields) -> None:
        with self._write_lock:
            event = {"type": event_type, **fields}
            self.out.write(json.dumps(event, default=str) + "\n")
            self.out.flush()

    def _emit_queue(self) -> None:
        with self._queue_lock:
            items = [text for text, _ in self._queue]  # images aren't shown in the queue strip
        self.emit("queue", items=items)

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
            self._pending_info[request_id] = (kind, title, detail)
        self.emit("approval_request", id=request_id, kind=kind, title=title, detail=detail)
        event.wait()
        with self._pending_lock:
            self._pending.pop(request_id, None)
            self._pending_info.pop(request_id, None)
        if self._cancel.is_set():
            return False
        if holder[0] == "always":
            if kind != "delete":
                self.always_allow.add(kind)
            return True
        return holder[0] == "approve"

    def _resolve(self, request_id: str, decision: str) -> bool:
        if decision not in {"approve", "allow", "yes", "decline", "deny", "no", "always", "all"}:
            self.emit("error", message=f"Unknown decision: {decision}")
            return False
        with self._pending_lock:
            entry = self._pending.get(request_id)
        if entry is None:
            self.emit("error", message=f"No pending approval for id {request_id}")
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
            on_todos=self._on_todos,
            on_answer_text=lambda text: self.emit("text_delta", text=text)
        )

    def _run_turn(self, text: str, image: dict | None = None) -> None:
        with self._notes_lock:
            notes = self._notes.copy()
            self._notes.clear()
        if notes:
            text = "\n".join([NOTE_PREFIX + note for note in notes]) + "\n" + text
        workspace = Workspace(self.root, approver=self.approve, scratch=self.scratch_root)
        try:
            result = self.run_fn(text, self.frontier_model, cli_provider=self.cli_provider, hooks=self._hooks(), conversation=self.conversation, workspace=workspace, image=image)
            self.emit("run_finished", answer=result.answer, stats=_jsonable_stats(result.stats))
            self._record_usage(result.stats)
        except Cancelled:
            self.emit("run_cancelled")
        except OrchestrationError as exc:
            self.emit("error", message=str(exc), stats=_jsonable_stats(getattr(exc, "stats", None)))
            self._record_usage(getattr(exc, "stats", None))
        except Exception as exc:
            self.emit("error", message=f"{type(exc).__name__}: {exc}")
            self._record_usage(getattr(exc, "stats", None))
        finally:
            self._start_next_queued()

    def _record_usage(self, stats) -> None:
        """Mirrors cli.py's _record_usage: keep this task in the project's
        usage history so /usage still shows it after the session ends. This
        was missing entirely from the desktop backend -- the GUI never
        wrote to usage.json at all, so its tasks never contributed to a
        project's previous-session/all-time totals.

        Catches broadly, not just OSError: this is a nice-to-have (the
        task itself already succeeded or already failed by the time this
        runs), and it must never turn an otherwise-successful run into a
        surfaced "error" event just because usage bookkeeping hit an edge
        case -- e.g. a stats object missing a field `Totals.add()` reads.
        """
        if stats is None:
            return
        try:
            usage_store.record(self.root, self.session_id, self.frontier_model, stats)
        except Exception:
            pass

    def _on_todos(self, todos: list[dict]) -> None:
        self._todos = list(todos)
        self.emit("todos_updated", todos=self._todos)

    def _start_next_queued(self) -> None:
        with self._queue_lock:
            if not self._queue:
                return
            next_text, next_image = self._queue.pop(0)
        self._emit_queue()
        self._cancel.clear()
        self.emit("run_started", text=next_text)
        self._worker = threading.Thread(target=self._run_turn, args=(next_text, next_image), daemon=True)
        self._worker.start()

    def handle(self, message: dict) -> bool:
        message_type = message.get("type")
        if message_type == "shutdown":
            return False
        if message_type == "trust_response":
            decision = "yes" if message.get("trust") else "no"
            self.trusted = trust.apply_choice(self.root, decision)
            self.emit("trust_result", trusted=self.trusted, folder=str(self.root))
            # "No" ends the session, same as the terminal prompt.
            return self.trusted
        if not self.trusted:
            self.emit("trust_required", folder=str(self.root))
            return True
        if message_type == "user_message":
            text = str(message.get("text", "")).strip()
            image = _parse_image(message.get("image"))
            if not text:
                self.emit("error", message="Empty message.")
            elif text.startswith("/") and len(text) > 1:
                cmd, arg = text.split(" ", 1) if " " in text else (text, "")
                cmd = cmd.lower()
                if cmd == "/memory":
                    self.handle_memory_command(arg)
                elif cmd == "/scratch":
                    self.handle_scratch_command(arg)
                elif cmd == "/queue":
                    self.handle_queue_command(arg)
                elif cmd == "/stop":
                    self.handle_stop_command()
                elif cmd == "/clear" or cmd == "/new":
                    self.handle_clear_command()
                elif cmd == "/usage":
                    self.handle_usage_command(arg)
                elif cmd == "/models" or cmd == "/installed" or cmd == "/catalog" or cmd == "/doctor" or cmd == "/scan":
                    self.handle_catalog_command(cmd)
                elif cmd == "/model":
                    self.handle_model_command(arg)
                elif cmd == "/auto":
                    self.handle_auto_command(arg)
                elif cmd == "/run":
                    self.handle_run_command(arg)
                elif cmd == "/help":
                    self.handle_help_command(arg)
                elif cmd == "/compact":
                    self.handle_compact_command()
                elif cmd == "/summary":
                    self.handle_summary_command()
                elif cmd == "/tell":
                    self.handle_tell_command(arg)
                elif cmd == "/why":
                    self.handle_why_command()
                elif cmd == "/stream":
                    self.handle_stream_command(arg)
                else:
                    self.emit("error", message=f"Unknown command: {cmd}")
            elif self.busy:
                with self._queue_lock:
                    self._queue.append((text, image))
                self._emit_queue()
            else:
                self._cancel.clear()
                self.emit("run_started", text=text)
                self._worker = threading.Thread(target=self._run_turn, args=(text, image), daemon=True)
                self._worker.start()
        elif message_type == "memory_list":
            self.handle_memory_command('')
        elif message_type == "memory_forget":
            name = str(message.get('name', '')).strip()
            if not name:
                self.emit('error', message='Missing name to forget.')
            else:
                self.handle_memory_command(f'forget {name}')
        elif message_type == "memory_clear":
            self.handle_memory_command('clear')
        elif message_type == "scratch_list":
            self.handle_scratch_command('')
        elif message_type == "scratch_clear":
            self.handle_scratch_command('clear')
        elif message_type == "queue_list":
            self.handle_queue_command('')
        elif message_type == "queue_clear":
            self.handle_queue_command('clear')
        elif message_type == "new_session":
            self.handle_clear_command()
        elif message_type == "usage_request":
            self.handle_usage_command("")
        elif message_type == "configured_models_request":
            self.emit("configured_models", models=_models())
        elif message_type == "get_state":
            self.emit('settings', auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)
            self.emit('todos_updated', todos=self._todos)
            self._emit_queue()
        elif message_type == "set_model":
            model = str(message.get('model', '')).strip()
            if model:
                self.frontier_model = model
                self.emit('settings', auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)
        elif message_type == "set_auto":
            self.auto_approve = bool(message.get('enabled', False))
            self.emit('settings', auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)
        elif message_type == "set_stream":
            self.stream_output = message.get('enabled', False)
            self.emit('settings', auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)
        elif message_type == "approval_response":
            self._resolve(str(message.get('id', '')), str(message.get('decision', 'decline')))
        elif message_type == "cancel":
            self.handle_stop_command()
        elif message_type == "system_stats":
            try:
                cpu_percent = psutil.cpu_percent(interval=0.01)
                memory = psutil.virtual_memory()
                ram_used_gb = round(memory.used / (1024 ** 3), 1)
                ram_total_gb = round(memory.total / (1024 ** 3), 1)
            except Exception:
                cpu_percent = 0
                ram_used_gb = 0
                ram_total_gb = 0
            self.emit('system_stats', hardware=_hardware(), cpu_percent=cpu_percent, ram_used_gb=ram_used_gb, ram_total_gb=ram_total_gb)
        else:
            self.emit("error", message=f"Unknown message type: {message_type}")
        return True

    def handle_memory_command(self, arg: str):
        if not arg:
            pass
        elif arg.startswith("clear"):
            memory.forget(self.root)
        elif arg.startswith("forget"):
            name = arg.split(" ")[1].strip()
            memory.forget_fact(self.root, name)
        facts = memory.list_facts(self.root)
        narrative = memory.load(self.root)
        goal = _project_goal(self.root)
        self.emit("memory", facts=facts, narrative=narrative, goal=goal)

    def handle_scratch_command(self, arg: str):
        if not arg:
            scratchpad = self._scratchpad or Scratchpad(self.root)
            files = [{'path': str(f.relative_to(scratchpad.root)).replace("\\", "/"), 'size': f.stat().st_size} for f in scratchpad.files()]
            self.emit("scratch", files=files)
        elif arg.startswith("clear"):
            if self.busy:
                self.emit("error", message="A run is already in progress.")
            else:
                scratchpad = self._scratchpad or Scratchpad(self.root)
                scratchpad.clear()
                files = []
                self.emit("scratch", files=files)

    def handle_queue_command(self, arg: str):
        if not arg:
            self._emit_queue()
        elif arg.startswith("clear"):
            with self._queue_lock:
                self._queue.clear()
            self._emit_queue()

    def handle_stop_command(self):
        self._cancel.set()
        with self._queue_lock:
            self._queue.clear()
        self._decline_all_pending()
        self._emit_queue()

    def handle_clear_command(self):
        if self.busy:
            self.emit("error", message="A run is already in progress.")
        else:
            self.conversation = Conversation(memory=memory.load(self.root), facts=memory.facts_for_prompt(self.root))
            self._todos = []  # Reset todos when starting a new session
            self.emit("session_reset")
            self.emit("todos_updated", todos=self._todos)  # Emit empty todos on session reset
            with self._queue_lock:
                self._queue.clear()
            self._emit_queue()

    def handle_usage_command(self, arg: str = "") -> None:
        if arg:
            command = arg.split(" ")[0]
            if command == "/memory":
                self.handle_memory_command("")
            elif command == "/scratch":
                self.handle_scratch_command("")
            elif command == "/queue":
                self.handle_queue_command("")
            elif command == "/stop":
                self.handle_stop_command()
            elif command == "/clear" or command == "/new":
                self.handle_clear_command()
            elif command == "/models" or command == "/installed" or command == "/catalog" or command == "/doctor" or command == "/scan":
                self.handle_catalog_command(command)
            elif command == "/model":
                self.handle_model_command("")
            elif command == "/auto":
                self.handle_auto_command("")
            elif command == "/run":
                self.handle_run_command("")
            elif command == "/help":
                self.handle_help_command()
            elif command == "/compact":
                self.handle_compact_command()
            elif command == "/summary":
                self.handle_summary_command()
            elif command == "/tell":
                self.handle_tell_command("")
            elif command == "/why":
                self.handle_why_command()
            elif command == "/stream":
                self.handle_stream_command("")
            else:
                self.emit("usage", command=command, message=f"Usage for {command} not found.")
        else:
            # Bare /usage crashed the whole backend process before this fix:
            # usage_store has no get_all_time_totals()/get_previous_session_totals()
            # -- the real functions are all_time(root)/previous_session(root, id) --
            # and handle() has no try/except around dispatch, so the AttributeError
            # propagated out of serve_forever()'s loop and killed the session.
            try:
                previous = usage_store.previous_session(self.root, self.session_id)
                total = usage_store.all_time(self.root)
            except OSError:
                previous, total = None, None
            self.emit(
                "usage",
                previous_session=_jsonable_stats(previous) if previous else None,
                all_time=_jsonable_stats(total) if total else None,
            )

    def handle_catalog_command(self, cmd: str):
        if cmd == "/models":
            self.emit("models", models=_models())
        elif cmd == "/installed":
            self.emit("installed", models=_installed())
        elif cmd == "/catalog":
            self.emit("catalog", items=_catalog())
        elif cmd == "/doctor":
            self.emit("doctor", checks=_doctor())
        elif cmd == "/scan":
            self.emit("system_stats", hardware=_hardware(), cpu_percent=psutil.cpu_percent(interval=0.01), ram_used_gb=round(psutil.virtual_memory().used / (1024 ** 3), 1), ram_total_gb=round(psutil.virtual_memory().total / (1024 ** 3), 1))

    def handle_model_command(self, arg: str):
        if not arg:
            self.emit("model", model=self.frontier_model)
        else:
            model = arg.strip()
            if not model:
                self.emit("error", message="Model cannot be empty.")
            else:
                self.frontier_model = model
                self.emit("model", model=self.frontier_model)

    def handle_auto_command(self, arg: str):
        arg = arg.strip().lower()
        if arg == "on":
            self.auto_approve = True
        elif arg == "off":
            self.auto_approve = False
        else:
            # Matches the CLI's `/auto` with no argument: toggle rather than
            # force off, so a bare "/auto" from the GUI behaves the same way.
            self.auto_approve = not self.auto_approve
        self.emit("settings", auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)

    def handle_run_command(self, arg: str):
        if self.busy:
            self.emit("error", message="A run is already in progress.")
        else:
            self._cancel.clear()
            self.emit("run_started", text=arg.strip())
            self._worker = threading.Thread(target=self._run_turn, args=(arg.strip(),), daemon=True)
            self._worker.start()

    def handle_help_command(self, arg: str = "") -> None:
        commands = [
            {"name": "/memory", "description": "Show or clear this folder's memory"},
            {"name": "/scratch", "description": "List or clear this session's scratchpad"},
            {"name": "/queue", "description": "Show or clear the task queue"},
            {"name": "/stop", "description": "Stop the running task"},
            {"name": "/clear, /new", "description": "Start a new session"},
            {"name": "/usage", "description": "Show token usage for the last task and this session"},
            {"name": "/models", "description": "Show best-fit local model per modality"},
            {"name": "/installed", "description": "Show installed models"},
            {"name": "/catalog", "description": "Show full model catalog"},
            {"name": "/doctor", "description": "Check everything's configured correctly"},
            {"name": "/scan", "description": "Show detected hardware"},
            {"name": "/model <id>", "description": "Show or switch the orchestrator model"},
            {"name": "/auto on|off", "description": "Approve file changes and commands without asking"},
            {"name": "/run <task>", "description": "Work on a task in this folder"},
            {"name": "/help", "description": "Show this list of commands"},
            {"name": "/compact", "description": "Compact the conversation history"},
            {"name": "/summary", "description": "Show the session summary"},
            {"name": "/tell <note>", "description": "Append a note for the running task"},
            {"name": "/why", "description": "Explain the pending approval request"},
            {"name": "/stream [on|off]", "description": "Toggle stream output on or off"}
        ]
        self.emit("command_help", commands=commands)

    def handle_compact_command(self):
        if self.busy:
            self.emit("error", message="A run is already in progress.")
        else:
            try:
                before_messages = len(self.conversation.messages)
                dispatcher = Dispatcher(detect_hardware(), installed=_installed_model_names(OllamaBackend()), hooks=self._hooks())
                changed = memory.compact(self.conversation, dispatcher, root=self.root)
                after_messages = len(self.conversation.messages)
                self.emit("compacted", before_messages=before_messages, after_messages=after_messages, changed=changed, message=f"Conversation history compacted from {before_messages} to {after_messages} turns.")
            except Exception as e:
                self.emit("error", message=str(e))

    def handle_summary_command(self):
        summary = {"memory": self.conversation.memory, "message_count": len(self.conversation.messages), "chars": self.conversation.chars(), "model": self.frontier_model}
        self.emit("summary", **summary)

    def handle_tell_command(self, arg: str):
        if not arg:
            self.emit("error", message="No note text provided.")
        else:
            with self._notes_lock:
                self._notes.append(arg)
            self.emit("note_added", note=arg)

    def handle_why_command(self):
        with self._pending_lock:
            lines = []
            for request_id, (event, decision) in self._pending.items():
                kind, title, detail = self._pending_info[request_id]
                lines.append(f"{kind.capitalize()}: {title} - {detail}")
            lines.append(f"Auto-approve is {'on' if self.auto_approve else 'off'}")
            self.emit("why", lines=lines)

    def handle_stream_command(self, arg: str):
        if not arg:
            self.stream_output = not self.stream_output
        elif arg.lower() == "on":
            self.stream_output = True
        elif arg.lower() == "off":
            self.stream_output = False
        self.emit("settings", auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)

    def serve_forever(self) -> None:
        self.emit("ready", root=str(self.root), model=self.frontier_model, auto_approve=self.auto_approve, scratchpad=str(self.scratch_root), stream_output=self.stream_output)
        if not self.trusted:
            self.emit("trust_required", folder=str(self.root))
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
        self._save_memory_at_exit()
        if self._scratchpad is not None:
            self._scratchpad.remove()
            self._scratchpad = None

    def _save_memory_at_exit(self) -> None:
        """Mirrors cli.py's `_save_memory_at_exit()` -- the desktop app has
        no `/exit`, so this is the only place a session ends. Before this,
        switching folders (or quitting) just killed the backend process
        outright (lib.rs's `kill_session()` writes `{"type": "shutdown"}`
        then kills the child), so even the one piece of cross-session state
        the CLI gives a folder -- the condensed session note and remembered
        facts, see memory.py -- was silently lost every time, on top of the
        raw chat transcript that was never going to survive a process restart
        anyway (see Conversation's own docstring: no cross-run persistence
        by design). Reported as "when I switch folders the chats are
        completely lost" -- this at least means the *next* time you open
        that folder, the orchestrator still has last session's summary and
        anything it was told to remember, same as ending a terminal session
        with /exit already gave the CLI.
        """
        try:
            if not any(m.get("role") == "user" for m in self.conversation.messages[1:]):
                return
            dispatcher = Dispatcher(detect_hardware(), installed=_installed_model_names(OllamaBackend()), hooks=self._hooks())
            if memory.keeper(dispatcher) is None:
                return
            memory.extract_facts(self.conversation, dispatcher, self.root, self._hooks())
            memory.compact(self.conversation, dispatcher, self._hooks(), keep_recent_turns=0, root=self.root)
        except Exception:  # noqa: BLE001 - best-effort only; never block shutdown over this
            pass

def serve_stdio(root: Path, frontier_model: str, cli_provider: str | None = None, auto_approve: bool = False, stream_output: bool = False) -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        server = StdioServer(root, frontier_model, cli_provider, out=real_stdout, inp=sys.stdin, auto_approve=auto_approve, stream_output=stream_output)
        server.serve_forever()
    finally:
        sys.stdout = real_stdout
