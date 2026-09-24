"""Defines the tools exposed to the frontier orchestrator model, and dispatches
each tool call to the best-fitting local model via the matching backend.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from collections.abc import Callable

from localforge import brief, verify, web
from localforge.workspace import Workspace, WorkspaceError
from localforge.backends import BACKENDS
from localforge.backends.ollama import (
    MAX_OUTPUT_TOKENS,
    MIN_NUM_CTX,
    LocalModelOutOfMemory,
    LocalPromptTooLarge,
    max_context,
    prompt_budget,
)
from localforge.catalog import ModelEntry, best_match, candidates, context_limit, load_catalog
from localforge.hardware import HardwareProfile

# Called right before a subtask is handed to a local model, so callers (the
# CLI) can show the user what's actually doing the work and why.
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
    # notes the user added while the task runs (/tell), fetched each step
    poll_notes: Callable[[], list[str]] | None = None
    # the account hit a usage limit: return "retry" to try again (after
    # waiting), ("switch", model, provider) to carry on with another
    # orchestrator, or None to give up. Without this hook the run fails.
    on_limit: Callable[[Exception], object] | None = None


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

# A model that ran out of memory at some window size is given half of it for
# a while (other apps may be holding memory); after this long, it's tried
# at its full size again.
SHRUNK_WINDOW_SECONDS = 15 * 60
_shrunk_windows: dict[str, tuple[int, float]] = {}  # model -> (window, when)


def _shrunk_window(name: str) -> int | None:
    found = _shrunk_windows.get(name)
    if found is None or time.monotonic() - found[1] > SHRUNK_WINDOW_SECONDS:
        _shrunk_windows.pop(name, None)
        return None
    return found[0]


# The orchestrator is the paid model: code is the local models' job. These
# stop it from writing code itself -- in edit_file, or pasted into a
# delegation's instructions (which also makes the local model a copy-typist).
MAX_EDIT_LINES = 12
MAX_CODE_LINES_IN_INSTRUCTIONS = 15
_FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.S)

# A command that checks work: tests, linters, type checkers, builds. Shared
# with orchestrator.py's completion check, which imports it from here.
CHECK_COMMAND = re.compile(
    r"\b(test|tests|pytest|unittest|jest|vitest|mocha|spec|check|lint|ruff|flake8|pylint|mypy|pyright|tsc|eslint|"
    r"build|compile|cargo|go (?:vet|build|test)|make|gradle|mvn|dotnet)\b",
    re.I,
)


def _code_lines(text: str) -> int:
    return sum(len(block.strip("\n").splitlines()) for block in _FENCED.findall(text or ""))


# The fixed text wrapped around a delegated prompt (preface, notes).
PROMPT_OVERHEAD = 500


def _with_notes(result: str, notes: list[str]) -> str:
    """A delegation's result plus what the orchestrator should know about
    how it was produced (reference files cut or left out)."""
    return result + "\n\n[localforge: " + "; ".join(notes) + ".]" if notes else result


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


_FENCE_LINE = re.compile(r"^\s*```[^`]*$")


def _extract_file_content(text: str) -> str | None:
    """A model asked for a whole file still tends to wrap it in a code fence
    and add a sentence around it; keep only the file.

    A reply that opens with a fence is the file, from that fence to the LAST
    fence line: a Markdown file has fences of its own, and matching the first
    closing fence kept only its opening paragraph (a README cut down to its
    title). Otherwise the longest fenced block wins. None when the reply
    opens a fence and never closes it -- the output was cut off, and writing
    it would save half a file.
    """
    lines = text.strip("\n").splitlines()
    if lines and _FENCE_LINE.match(lines[0]):
        closing = [i for i in range(1, len(lines)) if lines[i].strip() == "```"]
        if not closing:
            return None
        body = lines[1 : closing[-1]]
        return "\n".join(body).rstrip("\n") + "\n"
    blocks = re.findall(r"```[^\n`]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).rstrip("\n") + "\n"
    return text.strip("\n") + "\n"


def _cut_off_warning(modality: str) -> str:
    return (
        f"[WARNING: the local {modality} model's output hit its {MAX_OUTPUT_TOKENS}-token limit and was cut off "
        "mid-way. Split the work into smaller pieces (one function, class or section per call) rather than retrying it whole.]"
    )


# Modality -> most recent delegation's (path, instructions), so a
# re-delegation of the same file can carry its previous attempt and the
# failure forward, instead of the local model starting from nothing again.
_PREFACES = {
    "coding": "You are a focused coding assistant working in an existing project. Produce working code for this task.",
    "docs": "You are a technical writer working in an existing project. Write clear documentation for this task.",
    "general": "You are helping with an existing software project.",
}
# Keeps a local model inside the project instead of guessing.
LOCAL_RULES = (
    "Rules: match the style and conventions of the existing code and the project notes. Use only functions, "
    "modules and APIs that exist in the files you're given, the standard library, or the project's declared "
    "dependencies -- never invent them. If something you need is missing or unclear, make the smallest sensible "
    "assumption and say so in your NOTES."
)


def _prompt_for(modality: str, instructions: str, grounding: str = "", preferences: str = "") -> str:
    """What a local model is sent: who it is, the rules, what this project is
    built with and how (from the brief and remembered preferences), then the task."""
    parts = [_PREFACES.get(modality, _PREFACES["general"]), LOCAL_RULES]
    if grounding:
        parts.append("Project notes (from the project's brief):\n" + grounding)
    if preferences:
        parts.append("Remembered preferences and corrections for this project -- follow these:\n" + preferences)
    return "\n\n".join(parts) + "\n\nTask:\n" + instructions


_NOTES = re.compile(r"^\s*NOTES:\s*", re.I | re.M)
MAX_NOTES_CHARS = 500


def _split_notes(reply: str) -> tuple[str, str]:
    """(the reply without its NOTES, the notes). The local model is asked to
    put a few NOTES lines after the file's code block: what it assumed or
    didn't do. They're for the orchestrator, never written into the file."""
    lines = reply.rstrip().splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    start = fences[-1] + 1 if len(fences) >= 2 else 0
    for i in range(start, len(lines)):
        if _NOTES.match(lines[i]):
            notes = _NOTES.sub("", "\n".join(lines[i:]), count=1).strip()
            if notes.lower().rstrip(".") in ("none", "n/a", ""):
                notes = ""
            return "\n".join(lines[:i]), notes[:MAX_NOTES_CHARS]
    return reply, ""


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
        self._grounding: str | None = None
        self._preferences: str | None = None
        # The last failing check (run_command matching a test/lint/build
        # pattern) not yet passed on to a delegation, so a re-delegation
        # after a test failure carries the exact error instead of the
        # frontier model having to paste it in by hand.
        self._last_check_failure: str | None = None
        self._check_failure_shown = False
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

    def context_limit(self, entry: ModelEntry) -> int | None:
        """The context window `entry` can run with on this machine (None:
        the default cap). Every prompt it's given is sized to fit this: the
        memory limit, the length the model was trained for (Ollama silently
        caps the window there and cuts the prompt), and a smaller window
        after it recently ran out of memory."""
        backend = BACKENDS.get(entry.runtime)
        trained = backend.trained_context(entry.name) if hasattr(backend, "trained_context") else None
        limits = [x for x in (context_limit(entry, self.hardware), trained, _shrunk_window(entry.name)) if x]
        return min(limits) if limits else None

    def grounding(self) -> str:
        """The brief's stack/commands/conventions, for every delegated task
        (read once per task: /init may have changed it since the last one)."""
        if self._grounding is None:
            self._grounding = brief.grounding_for_local(self.workspace.root) if self.workspace is not None else ""
        return self._grounding

    def preferences(self) -> str:
        """Remembered facts, in the form the local models get (feedback/user/
        project types only -- not raw pointers/references)."""
        if self._preferences is None:
            from localforge import memory

            self._preferences = memory.facts_for_local(self.workspace.root) if self.workspace is not None else ""
        return self._preferences

    def prompt_chars(self, modality: str, output_tokens: int = MAX_OUTPUT_TOKENS) -> int:
        """How much task text the model for `modality` can take beside its
        reply, after the fixed preamble (rules and project notes)."""
        preamble = len(_prompt_for(modality, "", self.grounding(), self.preferences()))
        try:
            entry = self.resolve(modality)
        except Exception:  # noqa: BLE001 - no model: the call will fail with its own error
            return prompt_budget(None, output_tokens) - preamble
        return prompt_budget(self.context_limit(entry), output_tokens) - preamble

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
        # The prompt was sized for the current model's window; a model with a
        # smaller one would have it cut off.
        window = self.context_limit(current) or 0
        alternatives = [c for c in alternatives if (self.context_limit(c) or 0) >= window]
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
            backend.ensure_available(entry.name, on_progress=on_pull)
            self.installed.add(entry.name)
            from localforge import upgrades

            upgrades.mark_managed(entry.name)  # localforge downloaded it, so an upgrade may replace it
        backend.ensure_available(entry.name, on_progress=on_pull)
        started = time.monotonic()
        window = {"context_limit": self.context_limit(entry)}
        prompt = _prompt_for(modality, instructions, self.grounding(), self.preferences())
        if hooks.on_token is not None:
            result = backend.generate(entry.name, prompt, on_token=hooks.on_token, **window)
        else:
            result = backend.generate(entry.name, prompt, **window)
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
        try:
            return self._dispatch_delegate(modality, args, on_delegate)
        except LocalModelOutOfMemory as exc:
            # Not enough memory for this window right now. Halve it, remember
            # that for a while, and run the whole step again: the prompt is
            # re-sized (reference files trimmed, and the orchestrator told).
            entry = self.resolve(modality)
            window = self.context_limit(entry) or max_context()
            smaller = window // 2
            short = (
                "so memory is short on this machine right now (other apps, or another model still loaded). "
                "Nothing was written. Tell the user: closing other apps, or /upgrade to a model that fits, would help."
            )
            if smaller < MIN_NUM_CTX:
                raise LocalModelOutOfMemory(f"{exc}. It doesn't fit even with the smallest context window, {short}") from None
            _shrunk_windows[entry.name] = (smaller, time.monotonic())
            if self.hooks.on_tool is not None:
                self.hooks.on_tool("retry", f"{entry.name} didn't fit in memory with a {window:,}-token window; retrying with {smaller:,}")
            try:
                result = self._dispatch_delegate(modality, args, on_delegate)
            except LocalModelOutOfMemory as again:
                raise LocalModelOutOfMemory(f"{again}. It still didn't fit with half the context window ({smaller:,} tokens), {short}") from None
            return _with_notes(result, [f"{entry.name} ran with a {smaller:,}-token context window because memory was short ({exc})"])

    def _dispatch_delegate(self, modality: str, args: dict, on_delegate: DelegateCallback | None) -> str:
        instructions = str(args.get("instructions") or "")
        if args.get("path") and self._last_check_failure and not self._check_failure_shown:
            # A check failed since the last write and hasn't been carried
            # forward yet: hand the exact failure to the model writing the
            # fix, rather than relying on the orchestrator to paste it in.
            self._check_failure_shown = True
            instructions = (
                f"{instructions}\n\n(The project's checks failed after the last change -- fix this too:\n"
                f"{self._last_check_failure}\n)"
            )
        if (code := _code_lines(instructions)) > MAX_CODE_LINES_IN_INSTRUCTIONS:
            return (
                f"{TASK_MODALITIES[modality]['tool_name']} failed: your instructions contain {code} lines of code, and "
                "writing code is the local model's job (you're the paid model). Nothing was sent. Describe what's "
                "needed instead: the behaviour, names, signatures and constraints; name existing files in context_files."
            )
        path = args.get("path")
        if path:
            return self._delegate_to_file(modality, instructions, args.get("context_files"), str(path), on_delegate)
        room = self.prompt_chars(modality) - len(instructions) - PROMPT_OVERHEAD
        context, notes = self._context_block(args.get("context_files"), room)
        content, warning = self._delegate(modality, instructions + context, on_delegate)
        return _with_notes(f"{warning}\n\n{content}" if warning else content, notes)

    def _context_block(self, paths, room: int) -> tuple[str, list[str]]:
        """Files named in context_files, attached for the local model, and
        notes for the orchestrator about any that were cut or left out. They
        go from disk straight to the local model: the orchestrator never
        reads them, which is the point (reading them itself costs frontier
        tokens twice -- once to read, again to paste them into instructions).

        `room` is what's left of the local model's context window. Anything
        past it would be dropped by Ollama without a word -- and the model
        would lose its instructions, not the files -- so files are cut here,
        visibly, and the orchestrator is told which."""
        if not paths or self.workspace is None:
            return "", []
        if isinstance(paths, str):
            paths = [paths]
        room = max(0, min(room, MAX_CONTEXT_CHARS))
        blocks, notes, used = [], [], 0
        for raw in list(paths)[:MAX_CONTEXT_FILES]:
            try:
                target = self.workspace.resolve(str(raw))
                text = target.read_text(errors="replace")
            except (WorkspaceError, OSError) as exc:
                blocks.append(f"\n[context file {raw} unavailable: {exc}]")
                notes.append(f"context file {raw} was unavailable ({exc})")
                continue
            left = room - used
            if left <= 0:
                blocks.append(f"\n[context file {raw} left out: context limit reached]")
                notes.append(f"context file {raw} was left out: the local model's context was full")
                continue
            if len(text) > left:
                notes.append(f"context file {raw} was cut to its first {left:,} of {len(text):,} characters to fit")
                text = text[:left] + "\n[... cut to fit ...]"
            used += len(text)
            blocks.append(f"\n\nFile `{self.workspace.rel(target)}`:\n```\n{text}\n```")
        if len(paths) > MAX_CONTEXT_FILES:
            notes.append(f"only the first {MAX_CONTEXT_FILES} context files were attached")
        return ("\n\nReference files:" + "".join(blocks) if blocks else ""), notes

    def _delegate(self, modality: str, instructions: str, on_delegate: DelegateCallback | None) -> tuple[str, str | None]:
        """(content, warning or None), with one automatic retry when the first
        attempt errors (a crash, timeout, out of memory) or looks suspect."""
        entry = self.resolve(modality)
        try:
            result = self._run(modality, entry, instructions, on_delegate)
        except (DownloadDeclined, LocalModelOutOfMemory, LocalPromptTooLarge):
            # The user's decision, or a size problem: the same prompt would
            # fail the same way. Memory is handled in dispatch() with a
            # smaller window; a too-large prompt goes to the orchestrator.
            raise
        except Exception as first_error:  # noqa: BLE001 - recover here before bothering the orchestrator
            return self._recover(modality, entry, instructions, on_delegate, first_error)
        if result["type"] == "file":
            return f"[generated file: {result['content']}]", None
        if result.get("truncated"):
            return result["content"], _cut_off_warning(modality)  # retrying the same thing would cut off again

        reason = _looks_suspect(result["content"], instructions)
        if reason is None:
            return result["content"], None

        retry_entry = self._retry_candidate(modality, entry) or entry
        retry_instructions = instructions if retry_entry is not entry else self._reinforced_instructions(instructions, reason)
        retry_result = self._run(modality, retry_entry, retry_instructions, on_delegate)
        if retry_result["type"] == "file":
            return f"[generated file: {retry_result['content']}]", None
        if retry_result.get("truncated"):
            return retry_result["content"], _cut_off_warning(modality)
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
        if result.get("truncated"):
            return result["content"], _cut_off_warning(modality)
        reason = _looks_suspect(result["content"], instructions)
        if reason is None:
            return result["content"], None
        return result["content"], (
            f"[WARNING: this {modality} result may be unreliable ({reason}; an earlier attempt failed: {error}) "
            "-- verify before use, or delegate again with clearer/simpler instructions]"
        )

    def _delegate_to_file(
        self, modality: str, instructions: str, context_files, path: str, on_delegate: DelegateCallback | None
    ) -> str:
        if self.workspace is None:
            return "No project folder is open, so nothing can be written; omit `path` to get the text back."
        try:
            target = self.workspace.resolve(path)
        except WorkspaceError as exc:
            return f"Cannot write {path}: {exc}"
        rel = self.workspace.rel(target)
        current = target.read_text(errors="replace") if target.is_file() else None
        directive = (
            f"\n\nWrite the COMPLETE contents of the file `{rel}`. Reply with the file's contents in one code block, "
            "then, after the block, at most three short lines starting with NOTES: -- anything you assumed or "
            "couldn't do (or NOTES: none). Nothing before the code block."
        )
        current_block = ""
        if current is not None:
            current_block = (
                f"\n\nCurrent contents of `{rel}` (change what the task needs, keep the rest):\n```\n{current}\n```"
                f"\n\nNow write the complete new contents of `{rel}`."  # restated: a long file buries the task
            )
        room = self.prompt_chars(modality) - len(instructions) - len(directive) - len(current_block) - PROMPT_OVERHEAD
        tool = TASK_MODALITIES[modality]["tool_name"]
        if current is not None and room < 0:
            # The file alone doesn't fit next to room for the rewrite. Cutting
            # it and asking for "the complete file" would lose the rest of it.
            return (
                f"{tool} failed: {rel} is too large ({len(current.splitlines())} lines) for a local model to rewrite "
                "whole within its context window. Nothing was sent or written. Split the change: delegate without "
                "`path` for just the part that changes and apply it with edit_file, or move part of the file into "
                "a new, smaller module first."
            )
        context, notes = self._context_block(context_files, room)
        task = instructions + context + directive + current_block
        content, warning = self._delegate(modality, task, on_delegate)
        if warning:
            return _with_notes(f"{warning}\n\nNothing was written to {rel}. The local model returned:\n{content[:2000]}", notes)
        body, model_notes = _split_notes(content)
        written = _extract_file_content(body)
        if written is None:
            return _with_notes(
                f"{_cut_off_warning(modality)}\n\nNothing was written to {rel}: the reply opened a code block and never closed it.",
                notes,
            )

        # Check it parses before the user is asked about it. One free local
        # retry with the exact error; the paid orchestrator only hears about
        # it if the local model can't fix it.
        language, error = verify.check(rel, written)
        if error:
            self._log_correction("syntax", rel, error)
            if self.hooks.on_tool is not None:
                self.hooks.on_tool("retry", f"{rel}: {error[:120]}; asking the local model to fix it")
            retry = f"{task}\n\nYour previous version of `{rel}` failed a syntax check: {error}\nFix it and write the complete file again."
            content, warning = self._delegate(modality, retry, on_delegate)
            if not warning:
                body, model_notes = _split_notes(content)
                fixed = _extract_file_content(body)
                if fixed is not None:
                    written = fixed
                    language, error = verify.check(rel, written)
            if error or warning:
                self._log_correction("syntax (still failing)", rel, error or warning)
                return _with_notes(
                    f"{tool} failed: the local model's {rel} doesn't parse, even after one retry with the error "
                    f"({error or warning}). Nothing was written. Try smaller or clearer instructions, or split the file.",
                    notes,
                )
        checked = f"Syntax check: {language} parses." if language else ""

        result = self.workspace.write_file(rel, written)
        if self.hooks.on_tool_result is not None:
            self.hooks.on_tool_result("write", result.splitlines()[0])
        if not result.startswith(("Created ", "Updated ")):
            return _with_notes(result, notes)
        # The user saw the full diff when approving. The orchestrator gets a
        # summary and a short preview -- every line it's handed is paid for
        # again on each later step -- and can read_file to check more.
        lines = written.splitlines()
        preview = "\n".join(lines[:RESULT_PREVIEW_LINES])
        more = f"\n[... {len(lines) - RESULT_PREVIEW_LINES} more lines; read_file {rel} if you need to check them]" if len(lines) > RESULT_PREVIEW_LINES else ""
        report = [result.splitlines()[0]]
        if checked:
            report.append(checked)
        if model_notes:
            report.append(f"The local model's notes: {model_notes}")
        return _with_notes("\n".join(report) + f"\nFirst lines:\n{preview}{more}", notes)

    def _log_correction(self, kind: str, path: str, detail: str) -> None:
        if self.workspace is not None:
            from localforge import memory

            memory.note_correction(self.workspace.root, kind, path, detail)

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
            new_lines = len(str(args["new_string"]).splitlines())
            if new_lines > MAX_EDIT_LINES:
                return (
                    f"edit_file failed: that edit writes {new_lines} lines, and edit_file is for small fix-ups (up to "
                    f"{MAX_EDIT_LINES}). Nothing was changed. Have a local model write it: delegate_coding_task with "
                    f"path={args['path']!r}, describing the change."
                )
            return ws.edit_file(args["path"], args["old_string"], args["new_string"])
        if tool_name == "make_dir":
            return ws.make_dir(args["path"])
        if tool_name == "move_path":
            return ws.move_path(args["source"], args["destination"])
        if tool_name == "delete_path":
            return ws.delete_path(args["path"], bool(args.get("recursive")))
        if tool_name == "run_command":
            command = args.get("command") or args.get("instructions") or ""
            result = ws.run_command(command, args.get("timeout") or 300)
            if CHECK_COMMAND.search(command):
                if result.startswith("exit code 0"):
                    self._last_check_failure = None
                elif not result.startswith("The user declined"):
                    self._last_check_failure = result[:600]
                    self._check_failure_shown = False
            return result
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
