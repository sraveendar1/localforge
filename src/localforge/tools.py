"""Defines the tools exposed to the frontier orchestrator model, and dispatches
each tool call to the best-fitting local model via the matching backend.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from localforge import web
from localforge.workspace import Workspace, WorkspaceError
from localforge.backends import BACKENDS
from localforge.catalog import ModelEntry, best_match, candidates, load_catalog
from localforge.hardware import HardwareProfile

# Called right before a subtask is handed to a local model, so callers (the
# CLI, the wizard) can show the user what's actually doing the work and why.
DelegateCallback = Callable[[str, ModelEntry], None]


class DownloadDeclined(RuntimeError):
    """The user said no to downloading a model; never retried automatically."""


@dataclass
class ActivityHooks:
    """Optional callbacks so a caller can show a run's activity live. Every
    field is optional; without them a run is silent until it returns.
    """

    # before each frontier turn, with the 1-based round number
    on_frontier: Callable[[int], None] | None = None
    # (modality, entry) right before a subtask goes to a local model
    on_delegate: DelegateCallback | None = None
    # each chunk of local-model output, as it's generated
    on_token: Callable[[str], None] | None = None
    # (modality, entry, tokens, seconds) when a local model finishes a subtask
    on_done: Callable[[str, ModelEntry, int, float], None] | None = None
    # (model name, raw Ollama pull event) if a model has to be downloaded first
    on_pull: Callable[[str, dict], None] | None = None
    # (tool name, one-line summary) when the orchestrator uses a tool that
    # isn't a delegation: reading files, searching, running a command, the web
    on_tool: Callable[[str, str], None] | None = None
    # (tool name, short result) after it ran, e.g. "Read 120 lines" or a
    # command's last output lines
    on_tool_result: Callable[[str, str], None] | None = None
    # (list of {"content", "status"}) when the orchestrator updates its plan
    on_todos: Callable[[list[dict]], None] | None = None
    # each piece of the orchestrator's own answer text, as it's written
    on_answer_text: Callable[[str], None] | None = None


# Modality -> tool name + description. Each delegate tool takes
# `instructions` and an optional `path`: with a path, the local model writes
# that file (the dispatcher saves its output after the user approves the
# diff), so generated code never has to pass through the frontier model.
TASK_MODALITIES = {
    "coding": {
        "tool_name": "delegate_coding_task",
        "description": (
            "Have a local coding model write code. Give `path` to create or rewrite that file "
            "with its output (the current file content is sent to it automatically), or omit "
            "it to just get code back."
        ),
    },
    "docs": {
        "tool_name": "delegate_docs_task",
        "description": "Have a local model write documentation. Give `path` (e.g. README.md) to write it to that file.",
    },
    "general": {
        "tool_name": "delegate_general_task",
        "description": "Have a local model do a general text task. Give `path` to write the result to a file.",
    },
}


def _params(required: list[str], **props: tuple[str, str]) -> dict:
    return {
        "type": "object",
        "properties": {name: {"type": typ, "description": desc} for name, (typ, desc) in props.items()},
        "required": required,
    }


# Tools the orchestrator runs itself (no local model involved), modeled on
# Claude Code's: Read, Glob, Grep, Edit, Bash, WebSearch, WebFetch, TodoWrite.
# Local models have no internet and no tools; the orchestrator researches and
# investigates, then puts what matters into each delegated instruction.
DIRECT_TOOLS = {
    "read_file": {
        "description": "Read a file in the project, with line numbers. Read before you change anything.",
        "parameters": _params(
            ["path"],
            path=("string", "Path relative to the project folder."),
            offset=("integer", "First line to read (default 1)."),
            limit=("integer", "How many lines (default and max 400)."),
        ),
    },
    "list_files": {
        "description": "List files in the project (skips .git, node_modules, virtualenvs).",
        "parameters": _params(
            [],
            path=("string", "Folder to list, relative to the project (default: whole project)."),
            pattern=("string", "Optional glob such as '*.py' or 'src/**/*.ts'."),
        ),
    },
    "search": {
        "description": "Search file contents with a regular expression, like grep. Returns path:line: text.",
        "parameters": _params(
            ["pattern"],
            pattern=("string", "Python regular expression."),
            path=("string", "File or folder to search (default: whole project)."),
            glob=("string", "Only files whose name matches, e.g. '*.py'."),
        ),
    },
    "edit_file": {
        "description": (
            "Replace one exact, unique snippet in a file. Only for small fix-ups (a few lines) -- "
            "new files and real code changes go to delegate_coding_task with a path. The user approves the diff."
        ),
        "parameters": _params(
            ["path", "old_string", "new_string"],
            path=("string", "File to edit."),
            old_string=("string", "Exact text to replace, copied from read_file (without line numbers)."),
            new_string=("string", "Replacement text."),
        ),
    },
    "make_dir": {
        "description": "Create a folder (and any missing parents) in the project. The user approves it.",
        "parameters": _params(["path"], path=("string", "Folder to create, relative to the project.")),
    },
    "move_path": {
        "description": "Move or rename a file or folder inside the project. The user approves it.",
        "parameters": _params(
            ["source", "destination"],
            source=("string", "Existing file or folder."),
            destination=("string", "New path; must not exist yet."),
        ),
    },
    "delete_path": {
        "description": (
            "Delete a file or folder in the project. A non-empty folder needs recursive=true. "
            "The user sees exactly what will be removed and approves it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or folder to delete."},
                "recursive": {"type": "boolean", "description": "Required to delete a folder that has files in it."},
            },
            "required": ["path"],
        },
    },
    "run_command": {
        "description": (
            "Run a shell command in the project folder (git clone, tests, installs, builds). "
            "The user approves every command first. Output is returned (long output is cut)."
        ),
        "parameters": _params(
            ["command"],
            command=("string", "The shell command."),
            timeout=("integer", "Seconds before it's stopped (default 300)."),
        ),
    },
    "web_search": {
        "description": "Search the web. Returns titles, URLs and snippets; use fetch_url to read a page.",
        "parameters": _params(["query"], query=("string", "The search query.")),
    },
    "fetch_url": {
        "description": "Read a public web page as text.",
        "parameters": _params(["url"], url=("string", "Full http(s) URL.")),
    },
    "remember": {
        "description": (
            "Save a lasting fact for future sessions in this project (a user preference or correction, "
            "a project decision, a pointer to a resource). Reusing a name updates that memory."
        ),
        "parameters": _params(
            ["name", "content"],
            name=("string", "Short kebab-case name, e.g. prefers-pytest."),
            content=("string", "The fact, and why it matters."),
            description=("string", "One line summary."),
            type=("string", "user, feedback, project or reference."),
        ),
    },
    "forget": {
        "description": "Delete a saved memory that is wrong or no longer applies.",
        "parameters": _params(["name"], name=("string", "The memory's name.")),
    },
    "update_todos": {
        "description": (
            "Show the user your plan as a checklist, and keep it current as you work. "
            "Use it for any task with more than two steps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
}

# Context the dispatcher attaches for a local model (context_files), and how
# much of a written file the orchestrator is shown back.
MAX_CONTEXT_FILES = 8
MAX_CONTEXT_CHARS = 60_000
RESULT_PREVIEW_LINES = 12

# Modalities whose models are interchangeable in a pinch: all produce text.
TEXT_MODALITIES = ("coding", "docs", "general")

# kept for callers that only care about the web pair
WEB_TOOLS = {name: DIRECT_TOOLS[name] for name in ("web_search", "fetch_url")}


def _schema(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def build_tool_schemas(
    hardware: HardwareProfile, catalog: list[ModelEntry] | None = None, installed: set[str] | None = None
) -> list[dict]:
    """OpenAI/LiteLLM-style tool schemas: a delegate tool for every modality
    with a fitting local model on this machine, plus the direct tools. A
    modality with no fitting catalog entry (hardware too limited) is left out
    entirely, rather than exposing a tool the frontier model could call only
    to get a NoFittingModelError back.
    """
    schemas = []
    for modality, meta in TASK_MODALITIES.items():
        if not candidates(modality, hardware, catalog, installed):
            continue
        schemas.append(
            _schema(
                meta["tool_name"],
                meta["description"],
                {
                    "type": "object",
                    "properties": {
                        "instructions": {
                            "type": "string",
                            "description": "What to build or answer, and the constraints. Describe it -- never write the code yourself.",
                        },
                        "path": {"type": "string", "description": "Optional file to write the result to, relative to the project."},
                        "context_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Project files the local model should see (e.g. an interface it must match, a module to "
                                "summarize). localforge gives them to it directly -- you don't need to read them first."
                            ),
                        },
                    },
                    "required": ["instructions"],
                },
            )
        )
    for name, meta in DIRECT_TOOLS.items():
        schemas.append(_schema(name, meta["description"], meta["parameters"]))
    return schemas


def _extract_file_content(text: str) -> str:
    """A model asked for a whole file still tends to wrap it in a code fence
    and add a sentence around it; keep only the (longest) fenced block.
    """
    blocks = re.findall(r"```[^\n`]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).rstrip("\n") + "\n"
    return text.strip("\n") + "\n"


def _prompt_for(modality: str, instructions: str) -> str:
    prefaces = {
        "coding": "You are a focused coding assistant. Produce working code for this task:\n\n",
        "docs": "You are a technical writer. Write clear documentation for this task:\n\n",
        "general": "",
    }
    return prefaces.get(modality, "") + instructions


# Cheap, deterministic tripwires for the clearest local-model failure modes
# (empty output, an outright refusal, or something absurdly short given the
# request). This is not a correctness check -- it can't tell if generated
# code actually works -- it's a floor that catches obviously broken results
# before they reach the frontier model unflagged, so they get escalated to a
# better model or clearly marked instead of silently accepted.
_REFUSAL_MARKERS = (
    "as an ai language model",
    "i cannot ",
    "i can't ",
    "i'm sorry, but",
    "i am unable to",
    "i do not have the ability",
)


def _looks_suspect(content: str, instructions: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return "empty output"
    lowered = stripped.lower()
    for marker in _REFUSAL_MARKERS:
        if marker in lowered:
            return f"looks like a refusal (contains {marker!r})"
    if len(stripped) < 20 and len(instructions) > 80:
        return "suspiciously short for the size of the request"
    return None


class Dispatcher:
    """Resolves a tool call to modality -> catalog entry -> backend, and runs it."""

    def __init__(
        self,
        hardware: HardwareProfile,
        catalog: list[ModelEntry] | None = None,
        installed: set[str] | None = None,
        hooks: ActivityHooks | None = None,
        workspace: Workspace | None = None,
    ):
        self.hardware = hardware
        self.workspace = workspace
        self.catalog = catalog if catalog is not None else load_catalog()
        # Model tags already on disk, or None if unknown. When known, an
        # installed fitting model wins over a higher-tier one that would have
        # to be downloaded -- otherwise a run silently pulls gigabytes mid-task
        # (a real bug: qwen2.5-coder:14b was fetched during a run while setup
        # had picked the already-installed 7b).
        self.installed = installed
        self.hooks = hooks or ActivityHooks()
        self._resolved_models: dict[str, ModelEntry] = {}
        self.local_tokens_generated = 0  # running total, for usage metrics

    def _tool_name_to_modality(self, tool_name: str) -> str:
        for modality, meta in TASK_MODALITIES.items():
            if meta["tool_name"] == tool_name:
                return modality
        raise ValueError(f"Unknown tool: {tool_name}")

    def resolve(self, modality: str) -> ModelEntry:
        if modality not in self._resolved_models:
            entry = best_match(modality, self.hardware, self.catalog, installed=self.installed)
            if self.installed is not None and entry.name not in self.installed and modality in TEXT_MODALITIES:
                # Nothing installed for this modality: an installed model of
                # another text modality can do the job (a coder writes a fine
                # README) rather than pulling gigabytes mid-task. Seen live: a
                # "general" subtask silently downloaded qwen2.5:3b.
                stand_ins = [
                    m
                    for other in TEXT_MODALITIES
                    if other != modality
                    for m in candidates(other, self.hardware, self.catalog, self.installed)
                    if m.name in self.installed and m.runtime == entry.runtime
                ]
                if stand_ins:
                    entry = max(stand_ins, key=lambda m: m.quality_tier)
            self._resolved_models[modality] = entry
        return self._resolved_models[modality]

    def _retry_candidate(self, modality: str, current: ModelEntry) -> ModelEntry | None:
        """A different model for `modality` to retry with, if this
        hardware fits more than one. `resolve()` already picks the single
        best-fitting model up front, so on most machines there is no
        "better" model to escalate to -- this only helps when several
        models tie or otherwise fit; the fallback for everyone else is
        `_reinforced_instructions()` retrying the *same* model. Prefers the
        highest quality_tier among the alternatives, on a one-off basis
        that never changes `resolve()`'s cached pick for the rest of the run.
        """
        alternatives = [c for c in candidates(modality, self.hardware, self.catalog, self.installed) if c.name != current.name]
        if self.installed is not None:
            # A retry is not worth a surprise multi-GB download; with nothing
            # else installed, the caller retries the same model instead.
            alternatives = [c for c in alternatives if c.name in self.installed]
        return max(alternatives, key=lambda m: m.quality_tier) if alternatives else None

    def _reinforced_instructions(self, instructions: str, reason: str) -> str:
        return (
            f"{instructions}\n\n(Your previous attempt at this was rejected: {reason}. "
            "Provide a complete, direct response this time -- do not refuse and do not "
            "leave it incomplete.)"
        )

    def _run(self, modality: str, entry: ModelEntry, instructions: str, on_delegate: DelegateCallback | None) -> dict:
        hooks = self.hooks
        on_delegate = on_delegate or hooks.on_delegate
        if on_delegate is not None:
            on_delegate(modality, entry)
        backend = BACKENDS[entry.runtime]
        on_pull = (lambda event: hooks.on_pull(entry.name, event)) if hooks.on_pull else None
        if self.installed is not None and entry.name not in self.installed and entry.runtime == "ollama":
            # A download is a big, visible change: ask like any other.
            approver = self.workspace.approver if self.workspace is not None else None
            size = f"about {entry.disk_gb:g} GB" if entry.disk_gb else "a large download"
            if approver is None or not approver("download", f"Download {entry.name}", f"{entry.name} ({size}) for {modality} work"):
                raise DownloadDeclined(
                    f"{entry.name} isn't downloaded and the download wasn't approved; no {modality} model is available. "
                    "Tell the user, or use a different tool."
                )
            self.installed.add(entry.name)
        backend.ensure_available(entry.name, on_progress=on_pull)
        started = time.monotonic()
        if hooks.on_token is not None:
            result = backend.generate(entry.name, _prompt_for(modality, instructions), on_token=hooks.on_token)
        else:
            result = backend.generate(entry.name, _prompt_for(modality, instructions))
        tokens = result.get("tokens", 0)
        self.local_tokens_generated += tokens  # every attempt costs local compute, retries included
        if hooks.on_done is not None:
            hooks.on_done(modality, entry, tokens, time.monotonic() - started)
        return result

    def dispatch(self, tool_name: str, args: dict | str, on_delegate: DelegateCallback | None = None) -> str:
        """Run one tool call. `args` is the call's arguments object; a bare
        string is accepted as {"instructions": ...} for older callers.
        """
        if isinstance(args, str):
            args = {"instructions": args}
        if tool_name in DIRECT_TOOLS:
            return self._direct(tool_name, args)
        modality = self._tool_name_to_modality(tool_name)
        instructions = str(args.get("instructions") or "") + self._context_block(args.get("context_files"))
        path = args.get("path")
        if path:
            return self._delegate_to_file(modality, instructions, str(path), on_delegate)
        content, warning = self._delegate(modality, instructions, on_delegate)
        return f"{warning}\n\n{content}" if warning else content

    def _context_block(self, paths) -> str:
        """Files named in context_files, attached for the local model. They go
        from disk straight to the local model: the orchestrator never reads
        them, which is the point (reading them itself costs frontier tokens
        twice -- once to read, again to paste them into instructions)."""
        if not paths or self.workspace is None:
            return ""
        if isinstance(paths, str):
            paths = [paths]
        blocks, used = [], 0
        for raw in list(paths)[:MAX_CONTEXT_FILES]:
            try:
                target = self.workspace.resolve(str(raw))
                text = target.read_text(errors="replace")
            except (WorkspaceError, OSError) as exc:
                blocks.append(f"\n[context file {raw} unavailable: {exc}]")
                continue
            room = MAX_CONTEXT_CHARS - used
            if room <= 0:
                blocks.append(f"\n[context file {raw} left out: context limit reached]")
                continue
            if len(text) > room:
                text = text[:room] + "\n[... cut to fit ...]"
            used += len(text)
            blocks.append(f"\n\nFile `{self.workspace.rel(target)}`:\n```\n{text}\n```")
        return "\n\nReference files:" + "".join(blocks) if blocks else ""

    def _delegate(self, modality: str, instructions: str, on_delegate: DelegateCallback | None) -> tuple[str, str | None]:
        """(content, warning or None), with one automatic retry when the first
        attempt errors (a crash, timeout, out of memory) or looks suspect."""
        entry = self.resolve(modality)
        try:
            result = self._run(modality, entry, instructions, on_delegate)
        except DownloadDeclined:
            raise  # the user's decision, not a failure: asking again wouldn't help
        except Exception as first_error:  # noqa: BLE001 - recover here before bothering the orchestrator
            return self._recover(modality, entry, instructions, on_delegate, first_error)
        if result["type"] == "file":
            return f"[generated file: {result['content']}]", None

        reason = _looks_suspect(result["content"], instructions)
        if reason is None:
            return result["content"], None

        retry_entry = self._retry_candidate(modality, entry) or entry
        retry_instructions = instructions if retry_entry is not entry else self._reinforced_instructions(instructions, reason)
        retry_result = self._run(modality, retry_entry, retry_instructions, on_delegate)
        if retry_result["type"] == "file":
            return f"[generated file: {retry_result['content']}]", None
        retry_reason = _looks_suspect(retry_result["content"], instructions)
        if retry_reason is None:
            return retry_result["content"], None
        return retry_result["content"], (
            f"[WARNING: this {modality} result may be unreliable ({retry_reason}) -- verify before use, "
            "or delegate again with clearer/simpler instructions]"
        )

    def _recover(
        self, modality: str, entry: ModelEntry, instructions: str, on_delegate: DelegateCallback | None, error: Exception
    ) -> tuple[str, str | None]:
        """The first attempt raised. Try once more -- another installed model
        if there is one, else the same model told what went wrong -- and
        only if that fails too, raise an error naming everything tried."""
        retry_entry = self._retry_candidate(modality, entry) or entry
        retry_instructions = (
            instructions
            if retry_entry is not entry
            else f"{instructions}\n\n(A previous attempt at this failed with: {error}. Try again, keeping the answer focused.)"
        )
        try:
            result = self._run(modality, retry_entry, retry_instructions, on_delegate)
        except Exception as second_error:  # noqa: BLE001 - reported with both causes
            tried = entry.name if retry_entry is entry else f"{entry.name}, then {retry_entry.name}"
            raise RuntimeError(
                f"the local {modality} model failed twice (tried {tried}). First: {error}. Then: {second_error}"
            ) from second_error
        if result["type"] == "file":
            return f"[generated file: {result['content']}]", None
        reason = _looks_suspect(result["content"], instructions)
        if reason is None:
            return result["content"], None
        return result["content"], (
            f"[WARNING: this {modality} result may be unreliable ({reason}; an earlier attempt failed: {error}) "
            "-- verify before use, or delegate again with clearer/simpler instructions]"
        )

    def _delegate_to_file(self, modality: str, instructions: str, path: str, on_delegate: DelegateCallback | None) -> str:
        if self.workspace is None:
            return "No project folder is open, so nothing can be written; omit `path` to get the text back."
        try:
            target = self.workspace.resolve(path)
        except WorkspaceError as exc:
            return f"Cannot write {path}: {exc}"
        rel = self.workspace.rel(target)
        current = target.read_text(errors="replace") if target.is_file() else None
        brief = (
            f"{instructions}\n\nWrite the COMPLETE contents of the file `{rel}`. "
            "Reply with only the file's contents in one code block -- no explanation before or after."
        )
        if current is not None:
            brief += f"\n\nCurrent contents of `{rel}` (change what the task needs, keep the rest):\n```\n{current}\n```"
        content, warning = self._delegate(modality, brief, on_delegate)
        if warning:
            return f"{warning}\n\nNothing was written to {rel}. The local model returned:\n{content[:2000]}"
        written = _extract_file_content(content)
        result = self.workspace.write_file(rel, written)
        if self.hooks.on_tool_result is not None:
            self.hooks.on_tool_result("write", result.splitlines()[0])
        if not result.startswith(("Created ", "Updated ")):
            return result
        # The user saw the full diff when approving. The orchestrator gets a
        # summary and a short preview -- every line it's handed is paid for
        # again on each later step -- and can read_file to check more.
        lines = written.splitlines()
        preview = "\n".join(lines[:RESULT_PREVIEW_LINES])
        more = f"\n[... {len(lines) - RESULT_PREVIEW_LINES} more lines; read_file {rel} if you need to check them]" if len(lines) > RESULT_PREVIEW_LINES else ""
        return f"{result.splitlines()[0]}\nFirst lines:\n{preview}{more}"

    def _direct(self, tool_name: str, args: dict) -> str:
        """A tool the orchestrator runs itself. Failures come back as text so
        the orchestrator can adjust, never as an exception.
        """
        hooks = self.hooks
        if tool_name == "update_todos":
            todos = [t for t in (args.get("todos") or []) if isinstance(t, dict) and t.get("content")]
            if hooks.on_todos is not None:
                hooks.on_todos(todos)
            return f"Plan updated ({sum(t.get('status') == 'completed' for t in todos)}/{len(todos)} done)."

        if hooks.on_tool is not None:
            hooks.on_tool(tool_name, _summarize(tool_name, args))
        try:
            if tool_name in ("remember", "forget"):
                result = self._memory_call(tool_name, args)
            elif tool_name in WEB_TOOLS:
                target = args.get("query") or args.get("url") or args.get("instructions") or ""
                runner = web.web_search if tool_name == "web_search" else web.fetch_url
                result = runner(str(target))
            else:
                if self.workspace is None:
                    return f"{tool_name} needs a project folder, and none is open."
                result = self._workspace_call(tool_name, args)
        except (web.WebError, WorkspaceError) as exc:
            result = f"{tool_name} failed: {exc}"
        except KeyError as exc:
            result = f"{tool_name} is missing the required argument {exc}; check the tool's parameters."
        except (TypeError, ValueError) as exc:
            result = f"{tool_name} got bad arguments ({exc}); check the tool's parameters."
        if hooks.on_tool_result is not None:
            hooks.on_tool_result(tool_name, result)
        return result

    def _memory_call(self, tool_name: str, args: dict) -> str:
        from localforge import memory

        if self.workspace is None:
            return f"{tool_name} needs a project folder, and none is open."
        root = self.workspace.root
        if tool_name == "remember":
            name = memory.remember(root, args["name"], args["content"], args.get("description", ""), args.get("type", "project"))
            return f"Remembered {name}."
        return f"Forgot {args['name']}." if memory.forget_fact(root, args["name"]) else f"No memory named {args['name']!r}."

    def _workspace_call(self, tool_name: str, args: dict) -> str:
        ws = self.workspace
        if tool_name == "read_file":
            return ws.read_file(args["path"], args.get("offset") or 1, args.get("limit") or 400)
        if tool_name == "list_files":
            return ws.list_files(args.get("path") or ".", args.get("pattern") or "")
        if tool_name == "search":
            return ws.search(args["pattern"], args.get("path") or ".", args.get("glob") or "")
        if tool_name == "edit_file":
            return ws.edit_file(args["path"], args["old_string"], args["new_string"])
        if tool_name == "make_dir":
            return ws.make_dir(args["path"])
        if tool_name == "move_path":
            return ws.move_path(args["source"], args["destination"])
        if tool_name == "delete_path":
            return ws.delete_path(args["path"], bool(args.get("recursive")))
        if tool_name == "run_command":
            return ws.run_command(args.get("command") or args.get("instructions") or "", args.get("timeout") or 300)
        raise ValueError(f"unknown tool {tool_name}")


def _summarize(tool_name: str, args: dict) -> str:
    """One line for the activity feed, e.g. `Read src/app.py`."""
    if tool_name == "read_file":
        return str(args.get("path", ""))
    if tool_name == "list_files":
        return " ".join(x for x in (str(args.get("path") or "."), str(args.get("pattern") or "")) if x)
    if tool_name == "search":
        return repr(args.get("pattern", "")) + (f" in {args['path']}" if args.get("path") else "")
    if tool_name in ("remember", "forget"):
        return str(args.get("name", ""))
    if tool_name in ("edit_file", "make_dir", "delete_path"):
        return str(args.get("path", ""))
    if tool_name == "move_path":
        return f"{args.get('source', '')} → {args.get('destination', '')}"
    if tool_name == "run_command":
        return str(args.get("command") or args.get("instructions") or "")
    return str(args.get("query") or args.get("url") or args.get("instructions") or "")
