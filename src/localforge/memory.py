"""Session memory, kept by a local model.

A session keeps its whole conversation so the orchestrator has context, but
that history is resent to the frontier model every turn (and, under CLI
login, flattened into one prompt). Left alone it grows until it's slow,
expensive, or too big. So when it passes COMPACT_AT_CHARS -- or the user
types /compact -- a local model folds the older turns into a short
markdown "session memory" note, which replaces them in the system prompt.
The note is also saved per project folder, so a new session in the same
folder starts with it.

Summarizing is exactly the kind of bulk text work local models are for: it
costs no frontier tokens. If no local model is usable, the oldest turns are
dropped instead, so the size cap holds either way.
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


# --- persistence ---------------------------------------------------------------


def memory_file(root: Path) -> Path:
    """Where a project's memory lives: under localforge's own config folder,
    never inside the user's project (nothing is written there unasked).
    """
    root = Path(root).resolve()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", root.name).strip("-") or "project"
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:10]
    return config.CONFIG_DIR / "memory" / f"{slug}-{digest}.md"


def load(root: Path) -> str:
    path = memory_file(root)
    try:
        return path.read_text().strip() if path.is_file() else ""
    except OSError:
        return ""


def save(root: Path, text: str) -> None:
    path = memory_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def forget(root: Path) -> None:
    memory_file(root).unlink(missing_ok=True)


# --- compaction ----------------------------------------------------------------


def _turn_starts(messages: list[dict]) -> list[int]:
    """Indices (after the system message) where a user turn begins."""
    return [i for i, m in enumerate(messages) if i > 0 and m.get("role") == "user"]


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        content = str(m.get("content") or "").strip()
        if content:
            lines.append(f"[{m.get('role', '?')}] {content}")
    text = "\n\n".join(lines)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS // 2] + "\n\n[... middle omitted ...]\n\n" + text[-MAX_TRANSCRIPT_CHARS // 2 :]
    return text


def _summarize(old_memory: str, transcript: str, dispatcher, hooks) -> str | None:
    entry = None
    for modality in MEMORY_MODALITIES:
        try:
            entry = dispatcher.resolve(modality)
            break
        except NoFittingModelError:
            continue
    if entry is None or entry.runtime not in BACKENDS:
        return None
    if hooks is not None and hooks.on_tool is not None:
        hooks.on_tool("compact", f"condensing earlier turns with {entry.name}")
    backend = BACKENDS[entry.runtime]
    started = time.monotonic()
    try:
        backend.ensure_available(entry.name)
        result = backend.generate(entry.name, PROMPT.format(memory=old_memory or "(none yet)", transcript=transcript))
    except Exception:  # noqa: BLE001 - memory is best-effort; fall back to dropping turns
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

    summary = _summarize(conversation.memory, _render(old), dispatcher, hooks)
    if summary is not None:
        conversation.memory = summary
        if root is None and dispatcher.workspace is not None:
            root = dispatcher.workspace.root
        if root is not None:
            try:
                save(root, summary)
            except OSError:
                pass
    elif hooks is not None and hooks.on_tool_result is not None:
        hooks.on_tool_result("compact", "no local model available; dropped the oldest turns instead")

    conversation.messages = conversation.messages[:1] + kept
    conversation.reindex_tools()
    return True
