"""The project brief: what this project is, kept in the project.

Claude Code has CLAUDE.md; this is the same idea, written by a local model
rather than a paid one, in AGENTS.md -- the name other coding agents already
read, so one file serves them all. `localforge goals` (`/init` still works,
as an alias) reads the project (its layout, README, manifests, entry
points) together with what localforge remembers of your sessions, and
drafts `AGENTS.md`: the objective, how the project is built, how to run it,
and the decisions worth knowing. Every later session starts with it, so the
orchestrator doesn't have to rediscover the project (which costs paid
tokens every time). The first task in a project with no AGENTS.md yet
offers to draft it right then, seeded with that task's own description
(see `gather_context`'s `goal_hint`), rather than waiting for a separate
manual step.

It's a file in the project, so it's shared with the team and reviewed like
any other change: it's written through the usual diff-and-approval path, and
never rewritten silently. After a session that changed files, the keeper
drafts an update and leaves it pending for `/goals` to review; the same
review is also offered mid-session once enough has changed, not only at
the next session's start.

An existing AGENTS.md (the user's, or another tool's) is updated in place,
through the diff, never overwritten unseen. A CLAUDE.md (Claude Code's) is
read when there's no AGENTS.md, never written. LOCALFORGE.md is this file's
old name: still read, and /goals moves it to AGENTS.md.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

BRIEF_FILE = "AGENTS.md"
LEGACY_BRIEF = "LOCALFORGE.md"  # the old name; read, and moved to AGENTS.md by /goals
# Read (in this order) when there's no AGENTS.md; never written.
OTHER_BRIEFS = (LEGACY_BRIEF, "CLAUDE.md", ".cursorrules", ".github/copilot-instructions.md")
MANIFESTS = (
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "Gemfile", "composer.json", "requirements.txt", "Makefile", "docker-compose.yml",
)
README_CHARS = 4_000
MANIFEST_CHARS = 2_000
MAX_LISTED_FILES = 120

PROMPT = """You are writing the project brief for a codebase, for an AI assistant that will work on it in future sessions.
Write it in markdown, under 500 words, with these sections and nothing else:

# Project brief
## What this project is
(its purpose and the problem it solves, in two or three sentences)
## How it's built
(language, frameworks, key dependencies, entry points -- concrete names from the files below)
## Layout
(the folders that matter and what lives in them)
## Running and testing
(the exact commands, taken from the manifest/Makefile/README below -- don't invent any)
## Definition of done
(how to tell a change is finished: the exact test, lint and build commands to run, from the files below --
don't invent any; if the project has none, say "no automated checks")
## Conventions and decisions
(how the code is written here, and choices someone should not undo by accident)

Use only what's below. Where something isn't clear, leave the line out rather than guessing.
Be specific: real names, real commands. No filler, no praise.

{existing}
Project:
{context}
"""

REFRESH_NOTE = """The brief below already exists. Update it from the newer information: keep what's still true (and the
author's wording where it holds), correct what changed, and add what's missing. Output the complete updated brief.

Current brief:
{brief}

"""


def brief_path(root: Path) -> Path:
    return Path(root) / BRIEF_FILE


def existing_brief(root: Path) -> str:
    """This project's brief, or another tool's if that's what it has."""
    for name in (BRIEF_FILE, *OTHER_BRIEFS):
        path = Path(root) / name
        try:
            if path.is_file():
                return path.read_text(errors="replace").strip()
        except OSError:
            continue
    return ""


def brief_for_prompt(root: Path, limit: int = 6_000) -> str:
    text = existing_brief(root)
    return text if len(text) <= limit else text[:limit] + "\n[... brief truncated]"


# What a local model writing code needs from the brief: the stack, how the
# project runs and is tested, and its conventions -- not the project's story.
GROUNDING_SECTIONS = ("how it's built", "running and testing", "definition of done", "conventions and decisions")
GROUNDING_CHARS = 2_000


def section(text: str, names: tuple[str, ...]) -> str:
    """The body of the first `## <name>` section found, in `names` order."""
    bodies: dict[str, list[str]] = {}
    current = None
    for line in (text or "").splitlines():
        if line.startswith("## "):
            current = line[3:].strip().lower()
            bodies.setdefault(current, [])
        elif current is not None:
            bodies[current].append(line)
    for name in names:
        if (body := "\n".join(bodies.get(name, [])).strip()):
            return body
    return ""


def grounding_for_local(root: Path, limit: int = GROUNDING_CHARS) -> str:
    """The part of the brief every delegated task gets, so local models write
    code that fits the project without the orchestrator repeating it (which
    costs paid tokens, and gets forgotten). From a brief /goals wrote, just the
    sections above; from another tool's brief (CLAUDE.md, AGENTS.md), its
    opening, since its layout is unknown."""
    text = existing_brief(root)
    if not text:
        return ""
    sections, keep, found = [], False, False
    for line in text.splitlines():
        if line.startswith("## "):
            keep = line[3:].strip().lower() in GROUNDING_SECTIONS
            found = found or keep
        if keep:
            sections.append(line)
    picked = "\n".join(sections).strip() if found else text
    return picked if len(picked) <= limit else picked[:limit].rsplit("\n", 1)[0] + "\n[...]"


def pending_path(root: Path) -> Path:
    from localforge import memory

    return memory.project_dir(root) / "brief-update.md"


def save_pending(root: Path, text: str) -> None:
    from localforge import memory

    memory.ensure_dir(root)
    pending_path(root).write_text(text.strip() + "\n")


def take_pending(root: Path) -> str:
    """The pending update, removed as it's handed over."""
    path = pending_path(root)
    try:
        text = path.read_text().strip()
    except OSError:
        return ""
    path.unlink(missing_ok=True)
    return text


def gather_context(workspace, memory_text: str = "", facts: str = "", goal_hint: str = "") -> str:
    """What the local model is given: the project's own shape, plus what
    localforge remembers of the work (so the brief reflects the objective,
    not just the file tree). `goal_hint` is the task the user just typed,
    when this is a project's first goals draft -- the objective should
    reflect what they actually asked for, not just a cold file-tree read."""
    root = Path(workspace.root)
    parts = [workspace.snapshot()]

    listing = workspace.list_files()
    parts.append("Files:\n" + "\n".join(listing.splitlines()[:MAX_LISTED_FILES]))

    for name in ("README.md", "README.rst", "README.txt", "docs/README.md"):
        path = root / name
        if path.is_file():
            parts.append(f"{name}:\n{path.read_text(errors='replace')[:README_CHARS]}")
            break

    for name in MANIFESTS:
        path = root / name
        if path.is_file():
            parts.append(f"{name}:\n{path.read_text(errors='replace')[:MANIFEST_CHARS]}")

    if goal_hint:
        parts.append("What you're being asked to build right now:\n" + goal_hint)
    if facts:
        parts.append("What localforge remembers about this project:\n" + facts)
    if memory_text:
        parts.append("Where the last session left off:\n" + memory_text)
    return "\n\n".join(parts)


def build_prompt(context: str, current_brief: str = "") -> str:
    existing = REFRESH_NOTE.format(brief=current_brief) if current_brief else ""
    return PROMPT.format(existing=existing, context=context)


def draft(entry, context: str, current_brief: str = "", context_limit: int | None = None) -> str:
    """Have the local model write the brief. Returns markdown (never partial
    JSON or prose around it). `context_limit` is the window this model can
    run with on this machine (Dispatcher.context_limit)."""
    from localforge.backends import BACKENDS

    result = BACKENDS[entry.runtime].generate(entry.name, build_prompt(context, current_brief), context_limit=context_limit)
    text = str(result.get("content") or "").strip()
    if text.startswith("```"):
        inner = text.split("```")
        text = inner[1] if len(inner) > 1 else text
        text = text.removeprefix("markdown").removeprefix("md").strip()
    if text and not text.lstrip().startswith("#"):
        text = "# Project brief\n\n" + text
    return text.strip() + "\n" if text else ""


# --- the project goal, from the user's first request, and a running progress log ----------------
#
# Asked for: "when you start the conversation and the user types what they want to build, that
# should be recorded as a goal, and appended to AGENTS.md as the project grows with the agents."
# The first substantive request is kept in the project's localforge folder at once (no approval:
# it's localforge's own note), and a finished task that changed things adds a line to AGENTS.md's
# `## Progress log` -- written through the normal diff-and-approval path, never silently. Plain text,
# no model involved, so it works with no local model and can't drift.

GOAL_FILE = "goal.md"
GOAL_SECTION = "What this project is"
PROGRESS_SECTION = "Progress log"
MAX_GOAL_CHARS = 600
MAX_PROGRESS_ENTRIES = 40  # the oldest fall off, so the file doesn't grow without bound
MIN_GOAL_CHARS = 12


def goal_path(root: Path) -> Path:
    from localforge import memory

    return memory.project_dir(root) / GOAL_FILE


def recorded_goal(root: Path) -> str:
    try:
        return goal_path(root).read_text(errors="replace").strip()
    except OSError:
        return ""


def _tidy_goal(text: str) -> str:
    one_line = re.sub(r"\s+", " ", text or "").strip()
    return one_line if len(one_line) <= MAX_GOAL_CHARS else one_line[: MAX_GOAL_CHARS - 1].rstrip() + "…"


def worth_recording_as_goal(text: str) -> bool:
    """A greeting or a one-word reply isn't a goal; wait for a real request."""
    tidy = _tidy_goal(text)
    return not tidy.startswith("/") and len(tidy) >= MIN_GOAL_CHARS and len(tidy.split()) >= 3


def record_goal(root: Path, text: str) -> bool:
    """Keep `text` as the project's goal if it has none yet and `text` is a real request.
    True if it was recorded now."""
    from localforge import memory

    if recorded_goal(root) or not worth_recording_as_goal(text):
        return False
    try:
        memory.ensure_dir(root)
        goal_path(root).write_text(_tidy_goal(text) + "\n")
    except OSError:
        return False
    return True


def project_goal(root: Path) -> str:
    """What to show as the project's goal: AGENTS.md's own summary when it has one, else
    what the user first asked for."""
    return section(existing_brief(root), (GOAL_SECTION.lower(),)) or recorded_goal(root)


def _split_sections(text: str) -> list[tuple[str | None, list[str]]]:
    """[(heading or None for the preamble, its lines including the heading)]."""
    parts: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in (text or "").splitlines():
        if line.startswith("## "):
            parts.append((line[3:].strip(), [line]))
        else:
            parts[-1][1].append(line)
    return parts


def _join_sections(parts: list[tuple[str | None, list[str]]]) -> str:
    chunks = ["\n".join(lines).strip("\n") for _, lines in parts]
    return "\n\n".join(c for c in chunks if c).strip() + "\n"


def with_goal(text: str, goal: str) -> str:
    """`text` (AGENTS.md's content) with a `## What this project is` section holding `goal`,
    added at the top if it has none. A section already there is left alone."""
    goal = _tidy_goal(goal)
    if not goal or section(text, (GOAL_SECTION.lower(),)):
        return text
    if not (text or "").strip():
        return f"# Project brief\n\n## {GOAL_SECTION}\n\n{goal}\n"
    parts = _split_sections(text)
    new = (GOAL_SECTION, [f"## {GOAL_SECTION}", "", goal])
    # an empty section of that name is replaced in place; otherwise it goes right after the title
    for i, (name, _) in enumerate(parts):
        if name and name.lower() == GOAL_SECTION.lower():
            parts[i] = new
            return _join_sections(parts)
    parts.insert(1, new)
    return _join_sections(parts)


def progress_entries(text: str) -> list[str]:
    return [ln[2:].strip() for ln in section(text, (PROGRESS_SECTION.lower(),)).splitlines() if ln.startswith("- ")]


def recent_progress(text: str, n: int = 5) -> list[str]:
    return progress_entries(text)[-n:]


def with_progress(text: str, entry: str, today: date | None = None) -> str:
    """`text` with one line added to `## Progress log` (created at the end if missing).
    The same line twice in a row isn't added again; only the newest MAX_PROGRESS_ENTRIES stay."""
    entry = re.sub(r"\s+", " ", entry or "").strip()
    if not entry:
        return text
    line = f"{(today or date.today()).isoformat()} — {entry}"
    entries = progress_entries(text)
    if entries and entries[-1].split(" — ", 1)[-1] == entry:
        return text
    entries = (entries + [line])[-MAX_PROGRESS_ENTRIES:]
    body = [f"## {PROGRESS_SECTION}", ""] + [f"- {e}" for e in entries]
    parts = _split_sections(text or "")
    for i, (name, _) in enumerate(parts):
        if name and name.lower() == PROGRESS_SECTION.lower():
            parts[i] = (name, body)
            return _join_sections(parts)
    parts.append((PROGRESS_SECTION, body))
    return _join_sections(parts)


def progress_line(task: str, files: list[dict], limit: int = 90) -> str:
    """One line for the log: the request, and what it changed ("created a.py, updated b.py, +2 more")."""
    ask = re.sub(r"\s+", " ", task or "").strip()
    ask = ask if len(ask) <= limit else ask[: limit - 1].rstrip() + "…"
    shown, extra = files[:4], max(0, len(files) - 4)
    verbs = {"create": "created", "update": "updated", "delete": "deleted", "move": "moved"}
    bits = [f"{verbs.get(str(f.get('action', '')).lower().rstrip('d'), str(f.get('action', 'changed')).lower())} {f.get('path', '')}".strip() for f in shown]
    if extra:
        bits.append(f"+{extra} more")
    return f"{ask} ({', '.join(bits)})" if bits else ask
