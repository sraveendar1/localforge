"""Memory, modeled on Claude Code's: kept per project, in the project.

    <project>/.localforge/
        .gitignore          "*": git ignores the folder unless the user
                              deletes this to share memory with the team
        session.md          what the last session left off with (condensed)
        memory/MEMORY.md    index: one line per memory, loaded every session
        memory/<name>.md    one fact per file, with frontmatter:
                              name, description, type (user | feedback |
                              project | reference)
        usage.json          /usage history (usage_store.py)
        brief-update.md     a drafted AGENTS.md update awaiting /init (brief.py)

It used to live under ~/.config/localforge/projects/<folder>-<hash>/; the
user asked for it in the (trusted) project folder instead, so it moves with
the folder. An old folder there is moved in on first use.

Two things keep it current:

* In-session compaction. The whole conversation is resent every turn (and,
  under CLI login, flattened into one prompt), so once it passes
  COMPACT_AT_CHARS -- or on /compact -- a local model folds older turns into
  the session summary, which replaces them in the system prompt.
* On /exit, the local "memory keeper" model condenses the session into
  session.md and extracts durable facts (preferences, decisions, project
  facts) into memory files. The orchestrator can also save or drop facts
  itself with the remember/forget tools, the way Claude Code writes memory
  when asked to remember something.

Summarizing is bulk text work, which is what local models are for: it costs
no frontier tokens, and memory never triggers a model download. If no local
model is usable, compaction drops the oldest turns instead so the size cap
still holds, and the exit-time extraction is skipped.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from localforge import config
from localforge.backends import BACKENDS
from localforge.catalog import NoFittingModelError

COMPACT_AT_CHARS = 60_000  # compact before a turn once history passes this
# An open-weight orchestrator has a far smaller context window, so its
# history is condensed much sooner (a too-big prompt is silently cut by
# Ollama, which corrupts what the model reads).
COMPACT_AT_CHARS_LOCAL = 20_000
KEEP_RECENT_TURNS = 1  # most recent user turns kept verbatim
MAX_TRANSCRIPT_CHARS = 40_000  # what a small local model is given to fold in per pass
MEMORY_MODALITIES = ("general", "docs", "coding")  # first that fits does the summarizing

PROMPT = """You maintain the working memory of a coding session between a user and an AI orchestrator.
Merge the existing memory and the new conversation below into ONE updated memory, in markdown, with these sections:
## Goal
## Decisions and constraints
## Files and commands (paths created or changed, commands run and their results)
## Current state
## Open items
Keep exact names, paths, versions, commands, errors and user preferences. Drop small talk and anything superseded.
Stay under 400 words. Output only the memory.

Existing memory:
{memory}

New conversation to fold in:
{transcript}
"""


# --- where it lives ------------------------------------------------------------

MEMORY_TYPES = ("user", "feedback", "project", "reference")
INDEX = "MEMORY.md"
MAX_FACTS_CHARS = 8_000  # fact contents given to the orchestrator each session


LOCAL_DIR = ".localforge"
_GITIGNORE = (
    "# localforge's memory and usage for this project (see AGENTS.md for the shared brief).\n"
    "# Kept out of git by default; delete this file to share it with your team.\n"
    "*\n"
)


def _central_dir(root: Path) -> Path:
    """Where an older localforge kept this project's memory."""
    root = Path(root).resolve()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", root.name).strip("-") or "project"
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:10]
    return config.CONFIG_DIR / "projects" / f"{slug}-{digest}"


def project_dir(root: Path) -> Path:
    """localforge's folder for one project: `.localforge/` inside it."""
    folder = Path(root).resolve() / LOCAL_DIR
    central = _central_dir(root)
    if central.is_dir() and not folder.exists():
        try:  # memory from before it lived in the project: move it in
            import shutil

            shutil.move(str(central), str(folder))
            (folder / ".gitignore").write_text(_GITIGNORE)
        except OSError:
            pass
    return folder


def ensure_dir(root: Path) -> Path:
    """project_dir(), created (with its .gitignore) before anything is written."""
    folder = project_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    ignore = folder / ".gitignore"
    if not ignore.exists():
        ignore.write_text(_GITIGNORE)
    return folder


def memory_file(root: Path) -> Path:
    """The session summary (what the last session left off with)."""
    return project_dir(root) / "session.md"


def memory_dir(root: Path) -> Path:
    return project_dir(root) / "memory"


def _migrate(root: Path) -> None:
    """Move a summary saved by an older localforge (memory/<folder>-<hash>.md)."""
    new = memory_file(root)
    old = (config.CONFIG_DIR / "memory" / _central_dir(root).name).with_suffix(".md")
    if old.is_file() and not new.exists():
        ensure_dir(root)
        old.replace(new)


def load(root: Path) -> str:
    _migrate(root)
    path = memory_file(root)
    try:
        return path.read_text().strip() if path.is_file() else ""
    except OSError:
        return ""


def save(root: Path, text: str) -> None:
    ensure_dir(root)
    memory_file(root).write_text(text.strip() + "\n")


def forget(root: Path) -> None:
    """Delete everything remembered for this project."""
    import shutil

    shutil.rmtree(project_dir(root), ignore_errors=True)
    _migrate(root)  # an old-format file would otherwise reappear
    memory_file(root).unlink(missing_ok=True)


# --- facts: one per file, indexed in MEMORY.md --------------------------------------


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")[:60]


def _parse(text: str) -> dict:
    meta, body = {}, text
    if text.startswith("---\n"):
        head, sep, rest = text[4:].partition("\n---\n")
        if sep:
            body = rest
            for line in head.splitlines():
                key, _, value = line.partition(":")
                if key.strip() in ("name", "description", "type"):
                    meta[key.strip()] = value.strip()
    meta["content"] = body.strip()
    return meta


def list_facts(root: Path) -> list[dict]:
    folder = memory_dir(root)
    facts = []
    for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
        if path.name == INDEX:
            continue
        try:
            fact = _parse(path.read_text())
        except OSError:
            continue
        fact.setdefault("name", path.stem)
        facts.append(fact)
    return facts


def _write_index(root: Path) -> None:
    folder = memory_dir(root)
    facts = list_facts(root)
    if not facts:
        (folder / INDEX).unlink(missing_ok=True)
        return
    lines = [f"- [{f['name']}]({f['name']}.md) — {f.get('description', '')}" for f in facts]
    (folder / INDEX).write_text("\n".join(lines) + "\n")


def remember(root: Path, name: str, content: str, description: str = "", type: str = "project") -> str:
    """Create or update one fact. Returns its (normalized) name."""
    slug = _slug(name)
    if not slug:
        raise ValueError("a memory needs a name")
    if not str(content).strip():
        raise ValueError("a memory needs content")
    kind = type if type in MEMORY_TYPES else "project"
    description = " ".join(str(description or content).split())[:150]
    ensure_dir(root)
    folder = memory_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: {description}\ntype: {kind}\n---\n\n{str(content).strip()}\n"
    )
    _write_index(root)
    return slug


def forget_fact(root: Path, name: str) -> bool:
    path = memory_dir(root) / f"{_slug(name)}.md"
    if not path.is_file():
        return False
    path.unlink()
    _write_index(root)
    return True


def facts_for_prompt(root: Path) -> str:
    """The index plus each fact's content, for the system prompt."""
    facts = list_facts(root)
    if not facts:
        return ""
    parts = []
    for f in facts:
        parts.append(f"- {f['name']} ({f.get('type', 'project')}): {f['content']}")
    text = "\n".join(parts)
    return text if len(text) <= MAX_FACTS_CHARS else text[:MAX_FACTS_CHARS] + "\n[... more memories not shown]"


# What the code-writing (local) models are told from memory: preferences,
# corrections and project decisions -- the facts that shape code. Before,
# remembered facts reached only the orchestrator, so "never use print()"
# never reached the model actually writing the code.
LOCAL_FACT_TYPES = ("feedback", "user", "project")
LOCAL_FACTS_CHARS = 1_200


def facts_for_local(root: Path, limit: int = LOCAL_FACTS_CHARS) -> str:
    lines = [f"- {f['content']}" for f in list_facts(root) if f.get("type", "project") in LOCAL_FACT_TYPES]
    text = "\n".join(" ".join(line.split()) for line in lines)
    return text if len(text) <= limit else text[:limit].rsplit("\n", 1)[0] + "\n[...]"


# --- corrections given to local models --------------------------------------------
#
# Every time a local model's work is corrected (it didn't parse, a check
# failed after it, the orchestrator re-delegated the same file), a line is
# logged here. At /exit the memory keeper reads the log, and a correction
# that keeps coming up becomes a lasting "feedback" rule -- which then goes
# to every code-writing model (facts_for_local).

CORRECTIONS = "corrections.jsonl"
MAX_CORRECTIONS = 200


def note_correction(root: Path, kind: str, path: str, detail: str) -> None:
    import json

    try:
        folder = ensure_dir(root)
        log = folder / CORRECTIONS
        lines = log.read_text().splitlines() if log.is_file() else []
        entry = {"when": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "path": path, "detail": " ".join(str(detail).split())[:400]}
        lines.append(json.dumps(entry))
        log.write_text("\n".join(lines[-MAX_CORRECTIONS:]) + "\n")
    except OSError:
        pass  # a nice-to-have; never a reason to fail the task


def recent_corrections(root: Path, limit: int = 40) -> list[dict]:
    import json

    log = project_dir(root) / CORRECTIONS
    try:
        lines = log.read_text().splitlines() if log.is_file() else []
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


# --- unfinished work ---------------------------------------------------------------
#
# A task that ends before it's done -- stopped, paused past a usage limit,
# out of steps, or finished with plan items open -- is saved here, not left
# to a summary a small model might drop. The next session offers to continue
# it, and the orchestrator sees it in its instructions meanwhile.

OPEN_WORK = "open-work.json"
MAX_OPEN_WORK = 5
RESUME_PREFIX = "[Continuing unfinished work] "


def open_work(root: Path) -> list[dict]:
    import json

    path = project_dir(root) / OPEN_WORK
    try:
        data = json.loads(path.read_text()) if path.is_file() else []
    except (OSError, ValueError):
        return []
    return [d for d in data if isinstance(d, dict) and d.get("task")] if isinstance(data, list) else []


def _save_open_work(root: Path, items: list[dict]) -> None:
    import json

    path = ensure_dir(root) / OPEN_WORK
    if items:
        path.write_text(json.dumps(items[-MAX_OPEN_WORK:], indent=1) + "\n")
    else:
        path.unlink(missing_ok=True)


def save_open_work(root: Path, task: str, why: str, open_items: list[str], unchecked: list[str]) -> None:
    items = [w for w in open_work(root) if w.get("task") != task]
    items.append(
        {"task": task, "why": why, "open_items": open_items[:12], "unchecked": unchecked[:12], "when": time.strftime("%Y-%m-%d %H:%M")}
    )
    try:
        _save_open_work(root, items)
    except OSError:
        pass


def clear_open_work(root: Path, task: str) -> None:
    items = open_work(root)
    kept = [w for w in items if w.get("task") != task]
    if len(kept) != len(items):
        try:
            _save_open_work(root, kept)
        except OSError:
            pass


def describe_open_work(entry: dict) -> str:
    lines = [f"Task: {entry['task']}", f"Stopped because: {entry.get('why') or 'unknown'} ({entry.get('when', '')})"]
    if entry.get("open_items"):
        lines.append("Plan items not done:\n" + "\n".join(f"- {i}" for i in entry["open_items"]))
    if entry.get("unchecked"):
        lines.append("Changed but not checked yet: " + ", ".join(entry["unchecked"]))
    return "\n".join(lines)


def open_work_for_prompt(root: Path) -> str:
    return "\n\n".join(describe_open_work(w) for w in open_work(root))


# --- compaction ----------------------------------------------------------------


def _turn_starts(messages: list[dict]) -> list[int]:
    """Indices (after the system message) where a user turn begins. Notes
    and nudges localforge added mid-task are user-role too, but not turns."""
    from localforge.orchestrator import (
        CHECKPOINT_PREFIX,
        COMPLETION_PREFIX,
        NOTE_PREFIX,
        NUDGE,
        STEPS_LEFT_PREFIX,
    )

    return [
        i
        for i, m in enumerate(messages)
        if i > 0 and m.get("role") == "user" and not str(m.get("content") or "").startswith((NOTE_PREFIX, NUDGE, STEPS_LEFT_PREFIX, CHECKPOINT_PREFIX, COMPLETION_PREFIX))
    ]


def _calls(message: dict) -> str:
    """An API-path assistant message's tool calls, in the CLI transport's
    `[requested: ...]` form. Their content is empty, so without this the
    memory keeper never saw which files were written or commands run."""
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {} if isinstance(call, dict) else getattr(call, "function", None)
        name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
        arguments = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", "")
        if name:
            calls.append(f"{name}({str(arguments or '')[:300]})")
    return f"[requested: {', '.join(calls)}]" if calls else ""


def _render(messages: list[dict], limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    lines = []
    for m in messages:
        content = str(m.get("content") or "").strip()
        if content.startswith("[superseded:"):
            content = content.partition(" It began: ")[2]  # only how the step went
        if m.get("role") == "assistant" and m.get("tool_calls"):
            content = "\n".join(x for x in (content, _calls(m)) if x)
        if content:
            lines.append(f"[{m.get('role', '?')}] {content}")
    text = "\n\n".join(lines)
    if len(text) > limit:
        text = text[: limit // 2] + "\n\n[... middle omitted ...]\n\n" + text[-(limit // 2) :]
    return text


def _room(dispatcher, entry, *fixed: str) -> int:
    """Transcript characters the keeper can take: its context window on this
    machine, less the prompt around the transcript and room for the reply."""
    from localforge.backends.ollama import MAX_QUIET_OUTPUT_TOKENS, prompt_budget

    budget = prompt_budget(dispatcher.context_limit(entry), MAX_QUIET_OUTPUT_TOKENS)
    return max(2_000, min(MAX_TRANSCRIPT_CHARS, budget - sum(len(f) for f in fixed) - 200))


def keeper(dispatcher):
    """The local model that keeps session memory: the first installed model
    that fits, preferring general-purpose ones. None if there isn't one --
    memory never triggers a download."""
    installed = dispatcher.installed  # None = unknown (Ollama unreachable): don't filter
    for modality in MEMORY_MODALITIES:
        try:
            entry = dispatcher.resolve(modality)
        except NoFittingModelError:
            continue
        if (installed is None or entry.name in installed) and entry.runtime in BACKENDS:
            return entry
    return None


def _summarize(old_memory: str, old_messages: list[dict], dispatcher, hooks) -> str | None:
    entry = keeper(dispatcher)
    if entry is None:
        if hooks is not None and hooks.on_tool_result is not None:
            hooks.on_tool_result("compact", "no local model available; dropped the oldest turns instead")
        return None
    if hooks is not None and hooks.on_tool is not None:
        hooks.on_tool("compact", f"condensing earlier turns with {entry.name}")
    backend = BACKENDS[entry.runtime]
    started = time.monotonic()
    try:
        backend.ensure_available(entry.name)
        memory_text = old_memory or "(none yet)"
        transcript = _render(old_messages, _room(dispatcher, entry, PROMPT, memory_text))
        result = backend.generate(
            entry.name, PROMPT.format(memory=memory_text, transcript=transcript), context_limit=dispatcher.context_limit(entry)
        )
    except Exception as exc:  # noqa: BLE001 - memory is best-effort; fall back to dropping turns
        if hooks is not None and hooks.on_tool_result is not None:
            hooks.on_tool_result("compact", f"{entry.name} couldn't condense them ({exc}); dropped the oldest turns instead")
        return None
    dispatcher.local_tokens_generated += result.get("tokens", 0)
    text = str(result.get("content") or "").strip()
    if hooks is not None and hooks.on_tool_result is not None:
        hooks.on_tool_result("compact", f"memory updated: {len(text.split())} words in {time.monotonic() - started:.1f}s")
    return text or None


def compact(conversation, dispatcher, hooks=None, keep_recent_turns: int = KEEP_RECENT_TURNS, root: Path | None = None) -> bool:
    """Fold all but the most recent turns into `conversation.memory`.
    Returns True if anything was compacted.
    """
    starts = _turn_starts(conversation.messages)
    if len(starts) <= keep_recent_turns:
        return False
    cut = starts[-keep_recent_turns] if keep_recent_turns else len(conversation.messages)
    old, kept = conversation.messages[1:cut], conversation.messages[cut:]

    summary = _summarize(conversation.memory, old, dispatcher, hooks)
    if summary is not None:
        conversation.memory = summary
        if root is None and dispatcher.workspace is not None:
            root = dispatcher.workspace.root
        if root is not None:
            try:
                save(root, summary)
            except OSError:
                pass

    conversation.messages = conversation.messages[:1] + kept
    conversation.reindex_tools()
    return True


# --- extracting facts at the end of a session -----------------------------------

EXTRACT_PROMPT = """You maintain long-term memory for a coding project, like a careful note-taker.
From the session below, pick out facts worth remembering in FUTURE sessions: the user's preferences and
corrections (type "feedback"), who the user is or how they work ("user"), decisions and facts about this
project that aren't obvious from its files ("project"), and pointers to outside resources ("reference").
Skip anything temporary, obvious from the code, or already in the existing memories unless it changed.

Existing memories:
{existing}

Corrections given to the code-writing models (this and earlier sessions):
{corrections}
When the same kind of correction shows up more than once, save it as a "feedback" memory phrased as a rule for
the code-writing models (e.g. "Use logging, never print()"), unless an existing memory already says it.

Session:
{transcript}

Reply with ONLY JSON: {{"memories": [{{"name": "short-kebab-name", "type": "feedback|user|project|reference",
"description": "one line", "content": "the fact, and why it matters"}}], "forget": ["name-of-a-memory-now-wrong"]}}
Use an existing name to update that memory. Empty lists are fine."""


def extract_facts(conversation, dispatcher, root: Path, hooks=None) -> int:
    """Have the memory keeper turn this session into memory files. Returns
    how many memories were written or removed. Best-effort: a small model's
    unusable reply changes nothing."""
    from localforge.cli_transport import _extract_json

    entry = keeper(dispatcher)
    turns = [m for m in conversation.messages[1:] if m.get("role") in ("user", "assistant")]
    if entry is None or not any(m.get("role") == "user" for m in turns):
        return 0
    existing = "\n".join(f"- {f['name']}: {f['content'][:200]}" for f in list_facts(root)) or "(none)"
    if hooks is not None and hooks.on_tool is not None:
        hooks.on_tool("compact", f"saving what's worth remembering with {entry.name}")
    try:
        corrections = "\n".join(
            f"- {c.get('kind')} {c.get('path') or ''}: {c.get('detail', '')[:200]}" for c in recent_corrections(root, 30)
        ) or "(none)"
        transcript = _render(turns, _room(dispatcher, entry, EXTRACT_PROMPT, existing, corrections))
        result = BACKENDS[entry.runtime].generate(
            entry.name,
            EXTRACT_PROMPT.format(existing=existing, corrections=corrections, transcript=transcript),
            context_limit=dispatcher.context_limit(entry),
        )
    except Exception:  # noqa: BLE001 - memory is best-effort
        return 0
    dispatcher.local_tokens_generated += result.get("tokens", 0)
    decision = _extract_json(str(result.get("content") or "")) or {}
    changed = 0
    for item in decision.get("memories") or []:
        if not isinstance(item, dict):
            continue
        try:
            remember(root, item.get("name", ""), item.get("content", ""), item.get("description", ""), item.get("type", "project"))
            changed += 1
        except ValueError:
            continue
    for name in decision.get("forget") or []:
        if isinstance(name, str) and forget_fact(root, name):
            changed += 1
    if hooks is not None and hooks.on_tool_result is not None:
        hooks.on_tool_result("compact", f"{changed} memor{'y' if changed == 1 else 'ies'} saved or updated")
    return changed
