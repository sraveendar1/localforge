"""A per-session scratchpad, modeled on Claude Code's.

Temporary work -- pseudo-code, a local model's draft before it touches a
real file, a plan, long command output -- goes in a folder of its own for
this session, in the system temp directory rather than the project. So it
never shows up in the project or in git (nothing to .gitignore), writing
there needs no permission prompt, and it's deleted when the session ends.

The orchestrator addresses it as `scratchpad/...`; Workspace maps that
prefix here. Promoting a draft into the project (move_path from
scratchpad/ to a real path) goes through the normal diff approval.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

PREFIX = "scratchpad"


def base_dir() -> Path:
    """Per-user parent folder, so sessions of different users never mix."""
    uid = os.getuid() if hasattr(os, "getuid") else "user"
    return Path(tempfile.gettempdir()) / f"localforge-{uid}"


class Scratchpad:
    def __init__(self, project_root: Path):
        project_root = Path(project_root).resolve()
        slug = re.sub(r"[^A-Za-z0-9]+", "-", project_root.name).strip("-") or "project"
        self.session_dir = base_dir() / slug / uuid.uuid4().hex[:12]
        self.root = (self.session_dir / PREFIX).resolve() if self.session_dir.exists() else self.session_dir / PREFIX

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.session_dir, 0o700)  # private to this user
        self.root = self.root.resolve()
        return self.root

    def files(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*") if p.is_file()) if self.root.is_dir() else []

    def clear(self) -> None:
        if self.root.is_dir():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def remove(self) -> None:
        shutil.rmtree(self.session_dir, ignore_errors=True)
        for folder in (self.session_dir.parent, self.session_dir.parent.parent):
            try:
                folder.rmdir()  # only if empty: other sessions may still be using it
            except OSError:
                break
