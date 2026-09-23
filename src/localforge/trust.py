"""Folder trust, like Claude Code's "Do you trust the files in this folder?".

The first time localforge runs in a folder it asks. Trusting a folder lets
the orchestrator read, create, change, move and delete files there and run
commands in it -- each change still goes through its own permission prompt
unless /auto is on. Trust covers subfolders too, and is remembered in
localforge's own config folder, never in the project.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from localforge import config


def _file() -> Path:
    return config.CONFIG_DIR / "trusted_folders.json"


def trusted_folders() -> list[str]:
    try:
        data = json.loads(_file().read_text())
    except (OSError, ValueError):
        return []
    return [str(p) for p in data if isinstance(p, str)] if isinstance(data, list) else []


def looks_like_home(folder: Path) -> bool:
    """Home or a filesystem root -- worth warning about in any front end."""
    folder = Path(folder).resolve()
    return folder == Path.home().resolve() or folder == folder.parent


def apply_choice(folder: Path, choice: str) -> bool:
    """Apply a trust choice ("yes" or "no") and say whether the folder is trusted.

    Shared by the terminal prompt and a native desktop front end -- this module
    never imports any UI code.
    """
    folder = Path(folder).resolve()
    if choice == "yes":
        trust(folder)
        return True
    elif choice == "no":
        return False
    else:
        raise ValueError(f"Unknown choice: {choice}")


def decide(folder: Path, ask: Callable[[Path], str]) -> bool:
    """Trust flow for any front end: `ask` returns "yes" or "no" for an untrusted folder."""
    folder = Path(folder).resolve()
    if is_trusted(folder):
        return True
    choice = ask(folder)
    return apply_choice(folder, choice)


def is_trusted(folder: Path) -> bool:
    """True if the folder is a trusted entry or inside one -- trust covers subfolders."""
    folder = Path(folder).resolve()
    return any(folder == root or root in folder.parents for root in map(Path, trusted_folders()))


def trust(folder: Path) -> None:
    folder = str(Path(folder).resolve())
    folders = trusted_folders()
    if folder not in folders:
        folders.append(folder)
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(folders), indent=2) + "\n")


def untrust(folder: Path) -> bool:
    folder = str(Path(folder).resolve())
    folders = trusted_folders()
    if folder not in folders:
        return False
    folders.remove(folder)
    _file().write_text(json.dumps(sorted(folders), indent=2) + "\n")
    return True
