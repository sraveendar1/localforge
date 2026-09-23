"""The project brief: what this project is, kept in the project.

Claude Code has CLAUDE.md; this is the same idea, written by a local model
rather than a paid one. `localforge init` reads the project (its layout,
README, manifests, entry points) together with what localforge remembers of
your sessions, and drafts `LOCALFORGE.md`: the objective, how the project is
built, how to run it, and the decisions worth knowing. Every later session
starts with it, so the orchestrator doesn't have to rediscover the project
(which costs frontier tokens every time).

It's a file in the project, so it's shared with the team and reviewed like
any other change: it's written through the usual diff-and-approval path, and
never rewritten silently. After a session that changed files, the keeper
drafts an update and leaves it pending for `/init` to review.

If the project already has a CLAUDE.md or AGENTS.md, that's read instead of
duplicating it.
"""

from __future__ import annotations

from pathlib import Path

BRIEF_FILE = "LOCALFORGE.md"
# Briefs written for other tools; read rather than duplicated.
OTHER_BRIEFS = ("CLAUDE.md", "AGENTS.md", ".cursorrules", ".github/copilot-instructions.md")
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


def pending_path(root: Path) -> Path:
    from localforge import memory

    return memory.project_dir(root) / "brief-update.md"


def save_pending(root: Path, text: str) -> None:
    path = pending_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")


def take_pending(root: Path) -> str:
    """The pending update, removed as it's handed over."""
    path = pending_path(root)
    try:
        text = path.read_text().strip()
    except OSError:
        return ""
    path.unlink(missing_ok=True)
    return text


def gather_context(workspace, memory_text: str = "", facts: str = "") -> str:
    """What the local model is given: the project's own shape, plus what
    localforge remembers of the work (so the brief reflects the objective,
    not just the file tree)."""
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
