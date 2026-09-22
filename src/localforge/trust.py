"""Folder trust, like Claude Code's "Do you trust the files in this folder?".

The first time localforge runs in a folder it asks. Trusting a folder lets
the orchestrator read, create, change, move and delete files there and run
commands in it -- each change still goes through its own permission prompt
unless /auto is on. Trust covers subfolders too, and is remembered in
localforge's own config folder, never in the project.
"""

from __future__ import annotations

import json
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


def is_trusted(folder: Path) -> bool:
    folder = Path(folder).resolve()
    for entry in trusted_folders():
        root = Path(entry)
        if folder == root or root in folder.parents:
            return True
    return False


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
