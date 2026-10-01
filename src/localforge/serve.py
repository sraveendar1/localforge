"""JSON-lines server over stdin/stdout that lets the desktop app drive the orchestrator.

Each stdin line is one JSON message from the app; each stdout line is one JSON
event to the app. Every object has a "type" key. stdout carries only protocol
JSON; everything else goes to stderr.
"""
from __future__ import annotations

import dataclasses, json, os, shutil, sys, threading, uuid
from pathlib import Path
from typing import Callable, IO, List, Optional
from localforge import brief, cli_transport, config, delegate_target, local_transport, memory, model_fit, project_models, provider_check, spend, task_summary, trust
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

def _describe_active_target(modality: str, recs: dict, installed: set[str]) -> str:
    """What's actually handling `modality` right now -- names the concrete
    model even for "auto" rather than just saying the word "auto", which a
    user reported as unhelpful (they can't tell what's actually running).
    """
    if modality in delegate_target.GENERATIVE:
        return delegate_target.describe_modality(modality)
    target = delegate_target.get(modality)
    if target.kind != "auto":
        return delegate_target.describe(target)
    entry = recs.get(modality)
    if entry is None:
        return "auto -- no local model fits this machine"
    on_disk = "installed" if entry.name in installed else "not installed"
    return f"{entry.name} (auto, local, {on_disk})"

def _auth_for_typed_model(model: str) -> dict | None:
    """The auth settings to save with a model id typed in by hand (not one of
    the listed choices): local for an Ollama model, else whichever way this
    machine can reach that model's provider. None when it can't be told --
    then only the model is saved and the run treats it as explicit."""
    from localforge import cli

    if model.startswith(local_transport.PREFIXES):
        return {config.AUTH_METHOD_ENV_VAR: config.AUTH_LOCAL, config.FRONTIER_PROVIDER_ENV_VAR: "local"}
    provider = cli._cloud_provider(model)
    if provider is None:
        return None
    env_var = config.FRONTIER_PROVIDERS.get(provider)
    if env_var and os.environ.get(env_var):
        return {config.AUTH_METHOD_ENV_VAR: config.AUTH_API_KEY, config.FRONTIER_PROVIDER_ENV_VAR: provider}
    if config.FRONTIER_CLI_AUTH.get(provider) and cli_transport.available(provider):
        return {config.AUTH_METHOD_ENV_VAR: config.AUTH_CLI_LOGIN, config.FRONTIER_PROVIDER_ENV_VAR: provider}
    return None


_SETUP_PROVIDERS = ("anthropic", "openai", "gemini")
_PROVIDER_LABELS = {"anthropic": "Anthropic (Claude)", "openai": "OpenAI (GPT)", "gemini": "Google (Gemini)"}


def _orchestrator_readiness(model: str, cli_provider: str | None) -> tuple[bool, str]:
    """Whether the orchestrator that would really run can run, and if not, why
    in plain words. The route matters: a Claude login only counts once it has
    been chosen as the way in (auth method saved), not merely because `claude`
    is installed -- otherwise the run goes to the API and fails."""
    from localforge import cli

    if model.startswith(local_transport.PREFIXES):
        name = local_transport.model_name(model)
        installed = _installed_model_names(OllamaBackend())
        if name in installed or f"{name}:latest" in installed:
            return True, ""
        return False, f"{name} isn't downloaded (or Ollama isn't running)"
    if cli_provider:
        if cli_transport.available(cli_provider):
            return True, ""
        return False, cli_transport.requirements_message(cli_provider)
    provider = cli._cloud_provider(model)
    env_var = config.FRONTIER_PROVIDERS.get(provider) if provider else None
    if provider is None:
        return True, ""  # some other LiteLLM model: can't tell, so don't get in the way
    if env_var and os.environ.get(env_var):
        return True, ""
    return False, f"{model} needs a {_PROVIDER_LABELS.get(provider, provider)} API key or login, and none is set up yet"


def _setup_status(model: str, cli_provider: str | None) -> dict:
    """Everything the first-run screen needs, without any secret in it."""
    ready, reason = _orchestrator_readiness(model, cli_provider)
    ollama = OllamaBackend()
    running = ollama.is_running()
    sizes = model_fit.installed_sizes() if running else None
    hardware = detect_hardware()
    providers = []
    for provider in _SETUP_PROVIDERS:
        spec = config.FRONTIER_CLI_AUTH.get(provider)
        env_var = config.FRONTIER_PROVIDERS.get(provider)
        providers.append({
            "id": provider,
            "label": _PROVIDER_LABELS[provider],
            "key_set": bool(env_var and os.environ.get(env_var)),
            "key_url": config.FRONTIER_CONSOLE_URLS.get(provider),
            "cli": {
                "command": spec["command"],
                "installed": cli_transport.available(provider),
                "install_hint": spec["install_hint"],
                "login_hint": spec["login_hint"],
            } if spec else None,
        })
    return {
        "needs_setup": not ready,
        "orchestrator": {"model": model, "ready": ready, "reason": reason},
        "providers": providers,
        "ollama": {
            "installed": shutil.which("ollama") is not None,
            "running": running,
            "models": [
                {"name": name, "problem": model_fit.selection_problem(name, hardware, sizes, fetch_sizes=False)}
                for name in sorted(sizes or {})
            ],
            "suggested": _suggested_local_models(hardware, sizes),
        },
    }


MAX_UNFIT_SUGGESTIONS = 4


def _suggested_local_models(hardware, sizes: dict[str, float] | None) -> list[dict]:
    """Open-weight models to offer on the setup screen that aren't on disk yet:
    the ones this machine can run, best first, then a few that it can't (greyed
    out, with the reason -- they can't be downloaded from the app)."""
    from localforge.catalog import fit_problem

    if sizes is None:  # Ollama can't be asked, so nothing can be downloaded anyway
        return []
    installed = set(sizes)
    seen: set[str] = set()
    fit, unfit = [], []
    for entry in sorted(load_catalog(), key=lambda m: (-m.quality_tier, m.disk_gb)):
        if entry.runtime != "ollama" or entry.modality not in delegate_target.MODALITIES or entry.name in seen:
            continue
        if entry.name in installed:
            continue
        seen.add(entry.name)
        problem = fit_problem(entry, hardware, installed)
        row = {"name": entry.name, "disk_gb": entry.disk_gb, "quality_tier": entry.quality_tier, "modality": entry.modality, "problem": problem}
        (unfit if problem else fit).append(row)
    return fit[:8] + unfit[:MAX_UNFIT_SUGGESTIONS]


def _friendly_failure(exc: Exception, model: str, cli_provider: str | None) -> tuple[str, bool]:
    """(message, is a set-up problem). A first task with no key or login used
    to end in LiteLLM's raw `AuthenticationError: Missing Anthropic API Key -
    A call is being made to anthropic but...`; say what to do instead."""
    ready, reason = _orchestrator_readiness(model, cli_provider)
    if not ready and (type(exc).__name__ == "AuthenticationError" or "api key" in str(exc).lower()):
        return f"{reason}. Use “Set up” to add an API key or sign in, then send your message again.", True
    return f"{type(exc).__name__}: {exc}", False


def _orchestrator_options() -> list[dict]:
    """Every orchestrator usable on this machine, for the header dropdown:
    the models already in Ollama plus each cloud provider whose API key or
    CLI login exists -- the same list `/model` shows in the terminal, so the
    two can't drift. (The dropdown used to be a fixed list that ignored both.)
    """
    from localforge import cli  # local: cli imports this module lazily too

    options = []
    hardware, sizes = detect_hardware(), model_fit.installed_sizes()
    for model, label, auth in cli._model_choices():
        problem = None
        if model.startswith("ollama/"):
            group = "Local (Ollama, free)"
            problem = model_fit.selection_problem(local_transport.model_name(model), hardware, sizes, fetch_sizes=False)
        else:
            provider = auth.get(config.FRONTIER_PROVIDER_ENV_VAR, "")
            via = "login" if auth.get(config.AUTH_METHOD_ENV_VAR) == config.AUTH_CLI_LOGIN else "API key"
            group = f"{provider.title()} ({via})"
        options.append({"id": model, "label": label, "group": group, "auth": auth, "problem": problem})
    return options

def _advanced_model_snapshot() -> dict:
    """Current delegate target for every modality, for the Active LLMs
    panel and the /advanced-model chat command alike."""
    installed = _installed_model_names(OllamaBackend())
    recs = recommendations(detect_hardware(), installed=installed)
    snapshot = {}
    for m in delegate_target.ALL_MODALITIES:
        row = {
            "target": delegate_target.render(delegate_target.get(m)),
            "description": _describe_active_target(m, recs, installed),
        }
        entry = recs.get(m) if m not in delegate_target.GENERATIVE else None
        if entry is not None and delegate_target.get(m).kind == "auto":
            # What "auto" resolves to, so the setup screen can offer to download it.
            row.update(auto_model=entry.name, installed=entry.name in installed, disk_gb=entry.disk_gb)
        snapshot[m] = row
    return snapshot

MAX_MODELS_PER_PROVIDER = 8  # in a picker, per provider and way of signing in


def _delegate_options(modality: str) -> dict:
    """Everything the GUI's "Change" picker offers for one modality: every
    local catalog model (with its fit-relevant fields and install status,
    same shape `localforge catalog` already shows) plus a cloud entry for
    each model of each provider that's actually usable right now -- has an
    API key set, or its CLI is on PATH.
    """
    if modality in delegate_target.GENERATIVE:
        # Image: every image model of each provider that has an API key -- for
        # Gemini, the ones that key can actually use -- with the price per
        # image where it's known. Video: nothing.
        images = []
        if modality == "image":
            for provider in config.IMAGE_MODEL_CHOICES:
                key = os.environ.get(config.FRONTIER_PROVIDERS.get(provider) or "")
                if not key:
                    continue
                for model in provider_check.image_models_for(provider, key):
                    images.append({"kind": "api", "provider": provider, "model": model, "price_usd": provider_check.image_price(model)})
            # plus a provider's own CLI login where its CLI has an image tool (Codex)
            images += [
                {"kind": "cli", "provider": provider, "model": model}
                for provider, models in config.IMAGE_CLI_CHOICES.items()
                if cli_transport.available(provider)
                for model in models
            ]
        return {"local": [], "cloud": images, "current": delegate_target.render(delegate_target.get(modality))}
    installed = _installed_model_names(OllamaBackend())
    hardware, sizes = detect_hardware(), model_fit.installed_sizes()
    local = [
        {
            "name": m.name, "quality_tier": m.quality_tier, "disk_gb": m.disk_gb, "installed": m.name in installed,
            # Why it can't run here (too big for memory, or a download that won't fit the disk); None if it can.
            "problem": model_fit.selection_problem(m.name, hardware, sizes, fetch_sizes=False),
        }
        for m in load_catalog() if m.modality == modality
    ]
    catalog_names = {m["name"] for m in local}
    # Installed models that aren't in the catalog (typed in earlier) can be picked too.
    local += [
        {"name": name, "quality_tier": 0, "disk_gb": round(size, 1), "installed": True,
         "problem": model_fit.selection_problem(name, hardware, sizes, fetch_sizes=False)}
        for name, size in sorted((sizes or {}).items()) if name not in catalog_names and name not in {f"{n}:latest" for n in catalog_names}
    ]
    from localforge import cli

    cloud = []
    for provider, env_var in config.FRONTIER_PROVIDERS.items():
        if provider == "local" or not env_var or not os.environ.get(env_var):
            continue
        # Every model the provider offers (Gemini's from the key itself), not
        # just its default: "Claude orchestrates, Gemini Flash writes docs"
        # needs a choice of Gemini models.
        try:
            models = cli._provider_models(provider)
        except Exception:  # noqa: BLE001 - the curated list is the fallback
            models = list(config.FRONTIER_MODEL_CHOICES.get(provider, []))
        for model in models[:MAX_MODELS_PER_PROVIDER]:
            cloud.append({"kind": "api", "provider": provider, "model": model})
    for provider in config.FRONTIER_CLI_AUTH:
        if not cli_transport.available(provider):
            continue
        for model in config.FRONTIER_MODEL_CHOICES.get(provider, [])[:MAX_MODELS_PER_PROVIDER]:
            cloud.append({"kind": "cli", "provider": provider, "model": model})
    return {
        "local": local,
        "cloud": cloud,
        "current": delegate_target.render(delegate_target.get(modality)),
    }

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

def _doctor(model: str = "", cli_provider: str | None = None):
    checks = []
    ollama_path = shutil.which('ollama')
    checks.append({'name': 'Ollama installed', 'ok': ollama_path is not None, 'detail': ollama_path or 'Ollama not installed. Install it from https://ollama.com.'})
    if ollama_path is not None:
        running = OllamaBackend().is_running()
        checks.append({'name': 'Ollama running', 'ok': running, 'detail': 'Ollama is answering.' if running else 'Ollama is installed but not running. Start it (open the Ollama app, or `ollama serve`).'})
    if model:
        ready, reason = _orchestrator_readiness(model, cli_provider)
        checks.append({'name': 'Orchestrator', 'ok': ready, 'detail': f'{model} can run.' if ready else f'{reason}. Use Set up.'})
    for provider in _SETUP_PROVIDERS:
        env_var = config.FRONTIER_PROVIDERS.get(provider)
        key = os.environ.get(env_var or '')
        label = _PROVIDER_LABELS[provider]
        if key:
            status, detail = provider_check.check_key(provider, key)
            checks.append({'name': f'{label} API key', 'ok': status != provider_check.REJECTED, 'detail': detail if status != provider_check.OK else f'{env_var} is set and {provider} accepts it.'})
            if provider == 'gemini' and status == provider_check.OK:
                images = provider_check.gemini_image_models(key)
                checks.append({'name': 'Gemini image models', 'ok': bool(images), 'detail': f'{len(images)} available to this key: ' + ', '.join(m.removeprefix('gemini/') for m in images[:4]) if images else 'None of Google\'s image models are available to this key (image generation with Gemini needs one).'})
        spec = config.FRONTIER_CLI_AUTH.get(provider)
        if spec and cli_transport.available(provider):
            checks.append({'name': f'{spec["command"]} CLI', 'ok': True, 'detail': f'`{spec["command"]}` is installed (a login through it is used only once chosen in Set up).'})
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
                 stream_output: bool = False, model_explicit: bool = False):
        self.root = Path(root).resolve()
        self.frontier_model = frontier_model
        # An orchestrator named on the command line (--model) beats the
        # project's saved one, which otherwise applies once the folder is trusted.
        self._model_explicit = model_explicit
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
        if self.trusted:
            self._adopt_project_models()
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
            self._emit_summary(result.stats, "completed")
            self._record_usage(result.stats)
        except Cancelled as exc:
            self.emit("run_cancelled")
            self._emit_summary(getattr(exc, "stats", None), "stopped")
        except OrchestrationError as exc:
            self.emit("error", message=str(exc), stats=_jsonable_stats(getattr(exc, "stats", None)))
            self._emit_summary(getattr(exc, "stats", None), "stopped", str(exc))
            self._record_usage(getattr(exc, "stats", None))
        except Exception as exc:
            message, needs_setup = _friendly_failure(exc, self.frontier_model, self.cli_provider)
            self.emit("error", message=message, detail=f"{type(exc).__name__}: {exc}", setup_needed=needs_setup)
            self._emit_summary(getattr(exc, "stats", None), "failed", message)
            self._record_usage(getattr(exc, "stats", None))
            if needs_setup:
                self._emit_setup_status()
        finally:
            self._start_next_queued()

    def _emit_summary(self, stats, outcome: str, error: str | None = None) -> None:
        """What the task did and what went wrong, after every task -- the same
        summary the terminal prints. A plain question answered with no tools
        used has nothing to list, so a completed task with no steps sends none;
        a failure or stop always does. Never allowed to fail the task."""
        try:
            summary = task_summary.build(getattr(stats, "log", None), outcome, error)
            if outcome == "completed" and task_summary.is_empty(summary):
                return
            self.emit("task_summary", summary=summary)
        except Exception:  # noqa: BLE001, S110 - a nicety after the fact; the run itself already ended
            pass

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
            if self.trusted:
                # Only now may this folder's saved models apply (see project_models.py).
                self._adopt_project_models()
                self.emit("settings", auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)
                self._emit_advanced_model()
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
                elif cmd == "/budget":
                    self.handle_budget_command(arg)
                elif cmd == "/advanced-model":
                    self.handle_advanced_model_command(arg)
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
        elif message_type == "budget_request":
            self._emit_budget()
        elif message_type == "set_budget":
            self._set_budget(str(message.get("provider", "")).strip().lower(), message.get("usd"))
        elif message_type == "setup_status_request":
            threading.Thread(target=self._emit_setup_status, daemon=True).start()
        elif message_type == "save_api_key":
            threading.Thread(
                target=self._save_api_key, args=(str(message.get("provider", "")), str(message.get("key", ""))), daemon=True
            ).start()
        elif message_type == "setup_use_login":
            threading.Thread(target=self._use_login, args=(str(message.get("provider", "")),), daemon=True).start()
        elif message_type == "setup_choose_orchestrator":
            self._choose_orchestrator(str(message.get("model", "")).strip(), typed=bool(message.get("typed")))
        elif message_type == "pull_model":
            use_as = message.get("use_as") if isinstance(message.get("use_as"), dict) else None
            threading.Thread(target=self._pull_model, args=(str(message.get("model", "")), use_as), daemon=True).start()
        elif message_type == "finish_project_setup":
            self._finish_project_setup()
        elif message_type == "orchestrator_options_request":
            # Off the reader thread: a Gemini key makes the list fetch hit the network.
            threading.Thread(target=self._emit_orchestrator_options, daemon=True).start()
        elif message_type == "set_model":
            model = str(message.get('model', '')).strip()
            if model:
                if problem := self._switch_orchestrator(model):
                    self.emit("error", message=problem)
                self.emit('settings', auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)
        elif message_type == "advanced_model_request":
            self._emit_advanced_model()
        elif message_type == "delegate_options_request":
            modality = str(message.get('modality', '')).strip().lower()
            if modality not in delegate_target.ALL_MODALITIES:
                self.emit("error", message=f"Unknown task type: {modality}")
            else:
                self.emit("delegate_options", modality=modality, **_delegate_options(modality))
        elif message_type == "set_delegate_target":
            modality = str(message.get('modality', '')).strip().lower()
            target = str(message.get('target', '')).strip()
            if modality not in delegate_target.ALL_MODALITIES:
                self.emit("error", message=f"Unknown task type: {modality}")
            else:
                try:
                    delegate_target.apply(modality, target)
                except delegate_target.InvalidTarget as exc:
                    if message.get("inline"):  # from the setup screen: show it there, not in the chat
                        self.emit("setup_result", ok=False, message=str(exc))
                    else:
                        self.emit("error", message=str(exc))
                else:
                    self._emit_advanced_model()
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
            # Off the reader thread: it asks each provider whether its key works.
            threading.Thread(target=lambda: self.emit("doctor", checks=_doctor(self.frontier_model, self.cli_provider)), daemon=True).start()
        elif cmd == "/scan":
            self.emit("system_stats", hardware=_hardware(), cpu_percent=psutil.cpu_percent(interval=0.01), ram_used_gb=round(psutil.virtual_memory().used / (1024 ** 3), 1), ram_total_gb=round(psutil.virtual_memory().total / (1024 ** 3), 1))

    def _emit_orchestrator_options(self):
        try:
            options = _orchestrator_options()
        except Exception:  # noqa: BLE001 - a nicer dropdown, never a reason to fail
            options = []
        self.emit("orchestrator_options", options=options, current=self.frontier_model)

    # --- paid image spending: this month's total and the optional monthly limit ---

    def _emit_budget(self) -> None:
        self.emit("budget", providers=[spend.summary(p) for p in sorted(config.IMAGE_MODEL_CHOICES)])

    def handle_budget_command(self, arg: str) -> None:
        """`/budget` shows this month's paid image spending; `/budget gemini 10`
        sets a monthly limit and `/budget gemini off` removes it -- the same as
        `localforge budget` in the terminal."""
        parts = arg.split()
        if len(parts) >= 2:
            raw = parts[1].lstrip("$").lower()
            if raw in ("off", "none", "no"):
                self._set_budget(parts[0].lower(), None)
            else:
                try:
                    amount: float | None = float(raw)
                except ValueError:
                    amount = None
                if amount is None:
                    return self.emit("error", message=f"{parts[1]!r} isn't an amount. Use `/budget gemini 10` or `/budget gemini off`.")
                self._set_budget(parts[0].lower(), amount)
        lines = []
        for provider in sorted(config.IMAGE_MODEL_CHOICES):
            row = spend.summary(provider)
            limit = f"of ${row['limit']:.2f}" if row["limit"] else "(no limit set)"
            lines.append(f"{provider}: ${row['spent']:.2f} {limit} this month, {row['images']} image{'s' if row['images'] != 1 else ''}")
        self.emit("system_text", text="\n".join(lines))
        self._emit_budget()

    def _set_budget(self, provider: str, usd) -> None:
        """Set a provider's monthly limit for paid image generation, or (usd
        null) remove it. Saved globally, with the keys: the credit and the bill
        belong to the account, not to a project."""
        if provider not in config.IMAGE_MODEL_CHOICES:
            return self.emit("error", message=f"Unknown provider {provider!r}.")
        if usd is not None:
            if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not 0 < usd <= 100_000:
                return self.emit("error", message=f"{usd!r} isn't an amount. Give a number of dollars, or clear the limit.")
        spend.set_budget(provider, None if usd is None else float(usd))
        self._emit_budget()

    # --- first-run setup: keys, logins and the default orchestrator --------------

    def _emit_setup_status(self) -> None:
        try:
            self.emit("setup_status", **_setup_status(self.frontier_model, self.cli_provider))
        except Exception:  # noqa: BLE001 - a status panel must never take the session down
            pass

    def _setup_result(self, ok: bool, message: str) -> None:
        self.emit("setup_result", ok=ok, message=message)
        self._emit_setup_status()
        self.emit("settings", auto_approve=self.auto_approve, model=self.frontier_model, busy=self.busy, stream_output=self.stream_output)
        self._emit_orchestrator_options()
        self._emit_advanced_model()

    def _make_default_orchestrator(self, model: str, auth: dict) -> None:
        """Set `model` as the orchestrator: as the default every project starts
        from (what `localforge setup` saves), and for this session. A project
        that already has its own saved models is updated too, or its file
        would just put the old one back."""
        from localforge import cli

        config.save({config.FRONTIER_MODEL_ENV_VAR: model, **auth})
        if project_models.source() == "project":
            project_models.save({config.FRONTIER_MODEL_ENV_VAR: model, **auth})
        self.frontier_model = model
        self.cli_provider = cli._cli_provider_for(model, explicit=False)

    def _save_api_key(self, provider: str, raw_key: str) -> None:
        """Save an API key (globally, in the private config file -- keys aren't
        per project), after asking the provider whether it accepts it. The key
        is never echoed back or put in an event."""
        from localforge import cli

        env_var = config.FRONTIER_PROVIDERS.get(provider) if provider in _SETUP_PROVIDERS else None
        if not env_var:
            return self._setup_result(False, f"Unknown provider {provider!r}.")
        label = _PROVIDER_LABELS[provider]
        key = provider_check.clean_key(raw_key)
        if not provider_check.looks_like_a_key(key):
            return self._setup_result(False, "That doesn't look like an API key (it's empty, too short, or has spaces in it). Copy the whole key from the provider's page.")
        status, detail = provider_check.check_key(provider, key)
        if status == provider_check.REJECTED:
            return self._setup_result(False, f"{label} rejected that key ({detail}). Check it was copied completely, and that it's a key for {label}.")
        # Asked *before* saving: the new key itself would make the current
        # (default, never-chosen) model look ready and nothing would be chosen.
        was_ready, _ = _orchestrator_readiness(self.frontier_model, self.cli_provider)
        config.save({env_var: key})
        note = "" if status == provider_check.OK else f" I couldn't check it right now — {detail} — so it's saved as typed."
        if was_ready:
            return self._setup_result(True, f"Saved your {label} key.{note}")
        try:
            models = cli._provider_models(provider)
        except Exception:  # noqa: BLE001
            models = []
        model = (models or config.FRONTIER_MODEL_CHOICES.get(provider) or [""])[0]
        self._make_default_orchestrator(model, {config.AUTH_METHOD_ENV_VAR: config.AUTH_API_KEY, config.FRONTIER_PROVIDER_ENV_VAR: provider})
        self._setup_result(True, f"Saved your {label} key and chose {model} to plan and review your work.{note} You can change it any time in Models.")

    def _use_login(self, provider: str) -> None:
        """Use a provider's own CLI login (a subscription) as the orchestrator,
        after checking the CLI is installed and actually answers -- signed out,
        it doesn't, and saving it would only fail on the first task."""
        from localforge import cli

        spec = config.FRONTIER_CLI_AUTH.get(provider) if provider in _SETUP_PROVIDERS else None
        if spec is None:
            return self._setup_result(False, f"Unknown provider {provider!r}.")
        if not cli_transport.available(provider):
            return self._setup_result(False, cli_transport.requirements_message(provider))
        works, why = cli_transport.probe(provider)
        if not works:
            return self._setup_result(False, f"`{spec['command']}` didn't answer: {why}. {cli._cli_failure_advice(provider, why)}")
        model = config.FRONTIER_MODEL_CHOICES[provider][0]
        self._make_default_orchestrator(model, {config.AUTH_METHOD_ENV_VAR: config.AUTH_CLI_LOGIN, config.FRONTIER_PROVIDER_ENV_VAR: provider})
        self._setup_result(True, f"Using your {spec['command']} login: {model} will plan and review your work (it draws on your subscription).")

    def _choose_orchestrator(self, model: str, typed: bool = False) -> None:
        """Pick one of the orchestrators that can run here (as listed in the
        Models section): a downloaded local model, or a cloud one with a key or
        login. A local model too big for this machine is refused."""
        from localforge import cli

        match = next((auth for m, _label, auth in cli._model_choices() if m == model), None)
        if match is None and typed and model:
            # A model id typed in by hand: same rules as `/model <id>` (refused, with
            # the reason, when a local model is too big for this machine).
            if problem := self._switch_orchestrator(model):
                return self._setup_result(False, problem)
            return self._setup_result(True, f"{model} will plan and review your work.")
        if match is None:
            return self._setup_result(False, f"{model or 'That model'} isn't available here yet.")
        if model.startswith(local_transport.PREFIXES):
            if why := model_fit.selection_problem(local_transport.model_name(model)):
                return self._setup_result(False, f"Can't use {local_transport.model_name(model)}: {why}.")
        self._make_default_orchestrator(model, match)
        self._setup_result(True, f"{model} will plan and review your work.")

    # --- downloading open-weight models from the setup screen ---------------------

    _pulling: set[str] = set()

    @staticmethod
    def _normalize_model_name(raw: str) -> str:
        name = raw.strip()
        for prefix in local_transport.PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name if ":" in name else f"{name}:latest"

    def _pull_model(self, raw_name: str, use_as: dict | None) -> None:
        """Download an Ollama model (progress as `pull_progress` events, the end
        as `pull_result`) and, if asked, make it the orchestrator or a task
        type's model. A model that isn't recommended for this machine -- too big
        for its memory, no room on the disk, or a size that can't be confirmed --
        is refused *before* anything is downloaded."""
        import re
        import time as _time

        from localforge import upgrades

        name = self._normalize_model_name(raw_name)

        def result(ok: bool, message: str, **extra) -> None:
            self.emit("pull_result", model=name, ok=ok, message=message, **extra)
            self._emit_setup_status()
            self._emit_orchestrator_options()
            if use_as and use_as.get("modality"):
                self.emit("delegate_options", modality=str(use_as["modality"]), **_delegate_options(str(use_as["modality"])))

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._\-/]*:[A-Za-z0-9._\-]+", name):
            return result(False, f"{raw_name.strip()!r} isn't a model name Ollama understands (like llama3.1:8b).", blocked=True)
        if name in StdioServer._pulling:
            return result(False, f"{name} is already downloading.")
        ollama = OllamaBackend()
        if not ollama.is_running():
            return result(False, "Ollama isn't running. Open the Ollama app (or run `ollama serve`), then try again.")
        try:
            problem = model_fit.download_problem(name)
        except Exception as exc:  # noqa: BLE001
            problem = f"couldn't check whether it fits ({exc})"
        if problem:
            return result(False, f"Not downloaded: {name} isn't recommended for this machine: {problem}.", blocked=True)

        StdioServer._pulling.add(name)
        failure: list[str] = []
        last = [0.0]

        def on_progress(event: dict) -> None:
            if event.get("error"):
                failure.append(str(event["error"]))
                return
            now = _time.monotonic()
            if now - last[0] >= 0.25 or event.get("status") == "success":
                last[0] = now
                self.emit("pull_progress", model=name, status=str(event.get("status", "")),
                          completed=event.get("completed"), total=event.get("total"))

        try:
            self.emit("pull_progress", model=name, status="starting", completed=None, total=None)
            ollama.ensure_available(name, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001
            StdioServer._pulling.discard(name)
            return result(False, f"Couldn't download {name}: {exc}")
        StdioServer._pulling.discard(name)
        if failure:
            return result(False, f"Couldn't download {name}: {failure[0]}")
        upgrades.mark_managed(name)

        used = ""
        if use_as and use_as.get("orchestrator"):
            self._choose_orchestrator(f"ollama/{name}")
            used = " and set as the orchestrator"
        elif use_as and use_as.get("modality"):
            modality = str(use_as["modality"])
            try:
                delegate_target.apply(modality, name)
            except delegate_target.InvalidTarget as exc:
                return result(False, f"Downloaded {name}, but it can't be used for {modality}: {exc}")
            self._emit_advanced_model()
            used = f" and set for {modality}"
        result(True, f"Downloaded {name}{used}.")

    def _finish_project_setup(self) -> None:
        """Save the models chosen on the setup screen to this project's own file
        (.localforge/models.json), so the folder opens straight to chat next time."""
        project_models.save({})
        self._emit_advanced_model()

    def _adopt_project_models(self) -> None:
        """Apply this folder's saved models (.localforge/models.json) and send
        later changes to it. The orchestrator it names becomes this session's
        unless one was given on the command line."""
        from localforge import cli

        if project_models.activate(self.root) and not self._model_explicit:
            model = os.environ.get(config.FRONTIER_MODEL_ENV_VAR)
            if model:
                self.frontier_model = model
                self.cli_provider = cli._cli_provider_for(model, explicit=False)

    def _emit_advanced_model(self) -> None:
        self.emit("advanced_model", targets=_advanced_model_snapshot(), source=project_models.source())

    def _switch_orchestrator(self, model: str) -> str | None:
        """Make `model` this project's orchestrator, the way `/model` does in
        the terminal: a pick from the list saves its auth method + provider
        with it (a model from a different provider than the last one must not
        keep routing through the old provider's CLI login), the choice goes
        into the project's models file, and the run path follows. Returns why
        it was refused (a local model too big for this machine's memory or
        disk) with nothing changed, or None."""
        from localforge import cli

        if model.startswith(local_transport.PREFIXES):
            name = local_transport.model_name(model)
            if why := model_fit.selection_problem(name):
                return f"Can't use {name} as the orchestrator: {why}. Nothing was changed."
        try:
            match = next((auth for m, _label, auth in cli._model_choices() if m == model), None)
        except Exception:  # noqa: BLE001
            match = None
        auth = match if match is not None else _auth_for_typed_model(model)
        project_models.save({config.FRONTIER_MODEL_ENV_VAR: model, **(auth or {})})
        self.frontier_model = model
        self.cli_provider = cli._cli_provider_for(model, explicit=auth is None)
        return None

    def handle_model_command(self, arg: str):
        if not arg:
            self.emit("model", model=self.frontier_model)
        else:
            model = arg.strip()
            if not model:
                self.emit("error", message="Model cannot be empty.")
            else:
                if problem := self._switch_orchestrator(model):
                    self.emit("error", message=problem)
                self.emit("model", model=self.frontier_model)

    def handle_advanced_model_command(self, arg: str):
        parts = arg.strip().split(None, 1)
        if not parts:
            self._emit_advanced_model()
            return
        modality = parts[0].lower()
        if len(parts) == 1:
            if modality not in delegate_target.ALL_MODALITIES:
                self.emit("error", message=f"Unknown task type: {modality}. Use coding, docs, general, image, or video.")
                return
            self._emit_advanced_model()
            return
        try:
            delegate_target.apply(modality, parts[1])
        except delegate_target.InvalidTarget as exc:
            self.emit("error", message=str(exc))
            return
        self._emit_advanced_model()

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

def serve_stdio(root: Path, frontier_model: str, cli_provider: str | None = None, auto_approve: bool = False, stream_output: bool = False, model_explicit: bool = False) -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        server = StdioServer(root, frontier_model, cli_provider, out=real_stdout, inp=sys.stdin, auto_approve=auto_approve, stream_output=stream_output, model_explicit=model_explicit)
        server.serve_forever()
    finally:
        sys.stdout = real_stdout
