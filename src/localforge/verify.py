"""A quick check that a file a local model wrote at least parses, before the
user is asked to approve it.

Not a correctness check: code that parses can still be wrong, which is what
the orchestrator's review and the project's tests are for. It catches the
cheapest failure -- a stray bracket, a half-finished block, invalid JSON --
locally and for free, so the local model can fix it before anyone (the user,
or the paid orchestrator) has to look at it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import tomllib
import yaml

TOOL_TIMEOUT_SECONDS = 15


def _python(content: str, name: str) -> str | None:
    try:
        compile(content, name, "exec")
    except SyntaxError as exc:
        where = f"line {exc.lineno}" + (f", column {exc.offset}" if exc.offset else "")
        return f"Python syntax error at {where}: {exc.msg}"
    except ValueError as exc:  # e.g. null bytes
        return f"Python can't read it: {exc}"
    return None


def _json(content: str, name: str) -> str | None:
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        return f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    return None


def _yaml(content: str, name: str) -> str | None:
    try:
        list(yaml.safe_load_all(content))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        return f"invalid YAML{where}: {getattr(exc, 'problem', None) or exc}"
    return None


def _toml(content: str, name: str) -> str | None:
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return f"invalid TOML: {exc}"
    return None


def _external(command: list[str], suffix: str, label: str):
    """A checker that runs a tool on the file, if the tool is installed."""

    def check(content: str, name: str) -> str | None:
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as handle:
            handle.write(content)
            path = handle.name
        try:
            proc = subprocess.run(
                [*command, path], capture_output=True, text=True, timeout=TOOL_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL, check=False
            )
        except (subprocess.SubprocessError, OSError):
            return None
        finally:
            Path(path).unlink(missing_ok=True)
        if proc.returncode == 0:
            return None
        detail = (proc.stderr or proc.stdout).strip().replace(path, name)
        return f"{label} syntax error: {detail[:400]}"

    return check


# suffix -> (language, checker, tool it needs or None)
CHECKERS = {
    ".py": ("Python", _python, None),
    ".json": ("JSON", _json, None),
    ".yaml": ("YAML", _yaml, None),
    ".yml": ("YAML", _yaml, None),
    ".toml": ("TOML", _toml, None),
    ".js": ("JavaScript", _external(["node", "--check"], ".js", "JavaScript"), "node"),
    ".mjs": ("JavaScript", _external(["node", "--check"], ".mjs", "JavaScript"), "node"),
    ".cjs": ("JavaScript", _external(["node", "--check"], ".cjs", "JavaScript"), "node"),
    ".sh": ("shell", _external(["bash", "-n"], ".sh", "Shell"), "bash"),
}


def check(path: str, content: str) -> tuple[str | None, str | None]:
    """(language checked, error) for a file about to be written. (None, None)
    when there's no checker for this kind of file, or its tool isn't installed."""
    suffix = Path(path).suffix.lower()
    if suffix not in CHECKERS:
        return None, None
    language, checker, tool = CHECKERS[suffix]
    if tool is not None and shutil.which(tool) is None:
        return None, None
    return language, checker(content, path)
