"""The user's project, as the orchestrator sees it: the folder `localforge`
was started in. Modeled on Claude Code's own tools -- read_file (Read),
list_files (Glob), search (Grep), edit_file (Edit), run_command (Bash) --
plus write_file, which only the dispatcher uses to save what a *local*
model wrote (the frontier model plans; local models produce file content).

Confinement: every path is resolved (following `..` and symlinks) and must
stay inside the project root, or the call is refused. That does NOT confine
`run_command`: a shell command can `cd` anywhere, so the real control there
is the permission prompt, which shows the exact command before it runs.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

# (kind, title, detail) -> allowed? kind is "write" or "command"; detail is a
# diff or the command line. Without an approver every change is refused.
Approver = Callable[[str, str, str], bool]

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", ".idea", ".next"}
MAX_READ_LINES = 400
MAX_READ_CHARS = 40_000
MAX_LIST = 300
MAX_MATCHES = 100
MAX_FILES_SCANNED = 20_000  # a search started in a huge folder (e.g. home) must still end
MAX_OUTPUT_CHARS = 12_000
MAX_DIFF_LINES = 80
COMMAND_TIMEOUT = 300


class WorkspaceError(RuntimeError):
    """A tool call that can't be carried out; the message goes back to the orchestrator."""


def _deny_all(kind: str, title: str, detail: str) -> bool:
    return False


class Workspace:
    def __init__(self, root: Path, approver: Approver | None = None):
        self.root = Path(root).resolve()
        self.approver = approver or _deny_all

    # --- paths ----------------------------------------------------------------

    def resolve(self, path: str) -> Path:
        if not path or not str(path).strip():
            raise WorkspaceError("a path is required")
        candidate = (self.root / str(path).strip()).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(f"{path!r} is outside the project folder ({self.root})")
        return candidate

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root)) if path != self.root else "."

    def _walk(self, start: Path):
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
            for name in sorted(filenames):
                yield Path(dirpath) / name

    # --- read-only tools --------------------------------------------------------

    def read_file(self, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"{path} does not exist or is not a file")
        raw = target.read_bytes()
        if b"\0" in raw[:8000]:
            raise WorkspaceError(f"{path} looks like a binary file")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        offset = max(1, int(offset or 1))
        limit = max(1, min(int(limit or MAX_READ_LINES), MAX_READ_LINES))
        chunk = lines[offset - 1 : offset - 1 + limit]
        body = "\n".join(f"{n:>5}  {line}" for n, line in enumerate(chunk, offset))[:MAX_READ_CHARS]
        end = offset + len(chunk) - 1
        more = f"\n[lines {offset}-{end} of {len(lines)}; pass offset={end + 1} to read on]" if end < len(lines) else ""
        return f"{self.rel(target)} ({len(lines)} lines)\n{body}{more}"

    def list_files(self, path: str = ".", pattern: str = "") -> str:
        start = self.resolve(path or ".")
        if not start.is_dir():
            raise WorkspaceError(f"{path} is not a directory")
        found = []
        for file in self._walk(start):
            rel = self.rel(file)
            if pattern and not (fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(file.name, pattern)):
                continue
            found.append(rel)
            if len(found) > MAX_LIST:
                break
        if not found:
            return f"No files{' matching ' + repr(pattern) if pattern else ''} under {self.rel(start)}."
        extra = f"\n[first {MAX_LIST} shown; narrow with path or pattern]" if len(found) > MAX_LIST else ""
        return "\n".join(found[:MAX_LIST]) + extra

    def search(self, pattern: str, path: str = ".", glob: str = "") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise WorkspaceError(f"invalid regex {pattern!r}: {exc}") from exc
        start = self.resolve(path or ".")
        files = [start] if start.is_file() else self._walk(start)
        matches = []
        scanned = 0
        for file in files:
            if glob and not fnmatch.fnmatch(file.name, glob):
                continue
            scanned += 1
            if scanned > MAX_FILES_SCANNED:
                matches.append(f"[stopped after scanning {MAX_FILES_SCANNED} files; narrow the search with path or glob]")
                break
            try:
                raw = file.read_bytes()
            except OSError:
                continue
            if b"\0" in raw[:8000]:
                continue
            for n, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{self.rel(file)}:{n}: {line.strip()[:200]}")
                    if len(matches) >= MAX_MATCHES:
                        return "\n".join(matches) + f"\n[stopped at {MAX_MATCHES} matches]"
        return "\n".join(matches) if matches else f"No matches for {pattern!r}."

    # --- changes (each one approved first) -----------------------------------

    def _diff(self, target: Path, old: str, new: str) -> str:
        rel = self.rel(target)
        lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{rel}", f"b/{rel}", lineterm=""))
        if len(lines) > MAX_DIFF_LINES:
            lines = lines[:MAX_DIFF_LINES] + [f"... ({len(lines) - MAX_DIFF_LINES} more diff lines)"]
        return "\n".join(lines)

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        if target.is_dir():
            raise WorkspaceError(f"{path} is a directory")
        old = target.read_text(errors="replace") if target.exists() else ""
        if old == content:
            return f"{self.rel(target)} is unchanged."
        verb = "Update" if target.exists() else "Create"
        diff = self._diff(target, old, content)
        if not self.approver("write", f"{verb} {self.rel(target)}", diff):
            return f"The user declined the change to {self.rel(target)}; it was not written."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if content.endswith("\n") else content + "\n")
        added = sum(1 for d in diff.splitlines() if d.startswith("+") and not d.startswith("+++"))
        removed = sum(1 for d in diff.splitlines() if d.startswith("-") and not d.startswith("---"))
        return f"{verb}d {self.rel(target)} (+{added} -{removed}, {len(content.splitlines())} lines).\n{diff}"

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"{path} does not exist")
        text = target.read_text(errors="replace")
        count = text.count(old_string) if old_string else 0
        if count == 0:
            raise WorkspaceError(f"old_string not found in {path}; read the file and copy the text exactly")
        if count > 1:
            raise WorkspaceError(f"old_string appears {count} times in {path}; include more surrounding lines so it's unique")
        return self.write_file(path, text.replace(old_string, new_string, 1))

    def run_command(self, command: str, timeout: int = COMMAND_TIMEOUT) -> str:
        command = (command or "").strip()
        if not command:
            raise WorkspaceError("a command is required")
        if not self.approver("command", "Run command", command):
            return f"The user declined to run: {command}"
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=max(1, min(int(timeout or COMMAND_TIMEOUT), 1800)),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"
        output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        if len(output) > MAX_OUTPUT_CHARS:
            output = f"[... first {len(output) - MAX_OUTPUT_CHARS} chars cut ...]\n" + output[-MAX_OUTPUT_CHARS:]
        return f"exit code {proc.returncode}\n{output.strip()}"

    # --- context for the orchestrator -------------------------------------------

    def is_broad(self) -> bool:
        """True for the home folder or the filesystem root: fine to work in,
        but tools see far more than one project and searches get slow.
        """
        return self.root in (Path.home().resolve(), Path(self.root.anchor))

    def snapshot(self, max_entries: int = 60) -> str:
        """A short picture of the project, given to the orchestrator once per
        session, like Claude Code's own environment preamble.
        """
        entries = []
        for child in sorted(self.root.iterdir()):
            if child.name in SKIP_DIRS or child.name.startswith(".") and child.name not in {".gitignore", ".env.example"}:
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
            if len(entries) >= max_entries:
                entries.append("...")
                break
        branch = ""
        head = self.root / ".git" / "HEAD"
        if head.is_file():
            ref = head.read_text().strip()
            branch = f"\nGit branch: {ref.rsplit('/', 1)[-1]}" if ref.startswith("ref:") else "\nGit: detached HEAD"
        listing = "\n".join(f"  {e}" for e in entries) or "  (empty folder)"
        note = (
            "\nNote: this is the user's home folder, not a single project. Ask which project folder to work in, "
            "or create one (e.g. with run_command mkdir), before searching or writing broadly."
            if self.is_broad()
            else ""
        )
        return f"Project folder: {self.root}{branch}\nTop-level entries:\n{listing}{note}"
