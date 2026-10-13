#!/usr/bin/env python3
"""Stamp one version into every file that carries it.

localforge versions are calendar versions: ``yyyymmdd.N`` (the day of the build, then 1, 2, 3...
for each build that day), e.g. ``20261013.1``.  ``src/localforge/__init__.py`` is the single
source; this script copies it where a tool wants its own copy.  Tauri, Cargo and npm insist on
three-part semver, so those files get ``yyyymmdd.N.0`` (the same number with a ``.0``).

    python scripts/set_version.py              # today's date, next free number (see --current)
    python scripts/set_version.py 20261013.2   # an exact version (a leading "v" is allowed)
    python scripts/set_version.py --current    # print the version now in the source
    python scripts/set_version.py --verify     # every file carries the source's version
    python scripts/set_version.py --newer-than origin/main   # and it is higher than that ref's

Every push to main must carry a higher version than main had (CI runs the last two on each pull
request, `.github/workflows/version-check.yml`).
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "src" / "localforge" / "__init__.py"
VERSION_RE = re.compile(r"^\d{8}\.\d+$")


def current() -> str:
    return re.search(r'__version__\s*=\s*"([^"]+)"', INIT.read_text()).group(1)


def valid(version: str) -> bool:
    """yyyymmdd.N with a real calendar date and N >= 1."""
    if not VERSION_RE.match(version):
        return False
    day, number = version.split(".")
    try:
        datetime.datetime.strptime(day, "%Y%m%d")
    except ValueError:
        return False
    return int(number) >= 1


def next_version(today: datetime.date | None = None, now: str | None = None) -> str:
    """Today's date with the next number: 20261013.1, then .2 if today already has one."""
    day = (today or datetime.date.today()).strftime("%Y%m%d")
    now = now if now is not None else current()
    if valid(now) and now.startswith(day + "."):
        return f"{day}.{int(now.split('.')[1]) + 1}"
    return f"{day}.1"


def semver(version: str) -> str:
    return version + ".0"


def _sub(path: Path, pattern: str, repl: str, count: int = 1) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, count=count, flags=re.M)
    if n == 0:
        raise SystemExit(f"{path.relative_to(ROOT)}: no version found to replace")
    path.write_text(new)


def _json_edit(path: Path, edit) -> None:
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def stamp(version: str) -> list[str]:
    if not valid(version):
        raise SystemExit(f"{version!r} isn't a version: use yyyymmdd.N, e.g. 20261013.1")
    sv = semver(version)
    touched = []

    _sub(INIT, r'^__version__ = "[^"]*"', f'__version__ = "{version}"')
    touched.append("src/localforge/__init__.py")

    _json_edit(ROOT / "desktop/src-tauri/tauri.conf.json", lambda d: d.__setitem__("version", sv))
    touched.append("desktop/src-tauri/tauri.conf.json")

    _json_edit(ROOT / "desktop/package.json", lambda d: d.__setitem__("version", sv))
    touched.append("desktop/package.json")

    def lock(d):
        d["version"] = sv
        d.get("packages", {}).get("", {})["version"] = sv

    _json_edit(ROOT / "desktop/package-lock.json", lock)
    touched.append("desktop/package-lock.json")

    _sub(ROOT / "desktop/src-tauri/Cargo.toml", r'^version = "[^"]*"', f'version = "{sv}"')
    touched.append("desktop/src-tauri/Cargo.toml")

    # Cargo.lock lists the app's own crate; cargo would rewrite it anyway, but keep it in step.
    _sub(ROOT / "desktop/src-tauri/Cargo.lock", r'(name = "desktop"\nversion = )"[^"]*"', rf'\1"{sv}"')
    touched.append("desktop/src-tauri/Cargo.lock")

    # uv.lock needs nothing: the package version is dynamic (read from __init__.py by hatch), so uv
    # records no version for it.
    return touched


def _read(path: str, pattern: str) -> str:
    m = re.search(pattern, (ROOT / path).read_text(), flags=re.M)
    return m.group(1) if m else ""


def verify() -> list[str]:
    """What is out of step with the source version (empty when everything agrees)."""
    version, sv = current(), semver(current())
    found = {
        "desktop/src-tauri/tauri.conf.json": json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text()).get("version"),
        "desktop/package.json": json.loads((ROOT / "desktop/package.json").read_text()).get("version"),
        "desktop/package-lock.json": json.loads((ROOT / "desktop/package-lock.json").read_text()).get("version"),
        "desktop/src-tauri/Cargo.toml": _read("desktop/src-tauri/Cargo.toml", r'^version = "([^"]*)"'),
        "desktop/src-tauri/Cargo.lock": _read("desktop/src-tauri/Cargo.lock", r'name = "desktop"\nversion = "([^"]*)"'),
    }
    problems = [] if valid(version) else [f"src/localforge/__init__.py: {version!r} isn't yyyymmdd.N"]
    for path, got in found.items():
        want = sv
        if got != want:
            problems.append(f"{path}: has {got!r}, expected {want!r} (run scripts/set_version.py {version})")
    return problems


def _key(version: str) -> tuple[int, int]:
    day, number = version.split(".")
    return int(day), int(number)


def newer_than(ref: str) -> str | None:
    """None when the working tree's version is higher than `ref`'s, else the reason it isn't."""
    import subprocess

    try:
        old = subprocess.run(["git", "show", f"{ref}:src/localforge/__init__.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"couldn't read the version at {ref}: {exc}"
    m = re.search(r'__version__\s*=\s*"([^"]+)"', old)
    before = m.group(1) if m else ""
    now = current()
    if not valid(now):
        return f"{now!r} isn't a yyyymmdd.N version"
    if valid(before) and _key(now) <= _key(before):
        return f"the version is {now} but {ref} is already at {before}; every push to main needs a higher one. Run: python scripts/set_version.py"
    return None


def main(argv: list[str]) -> int:
    if argv[:1] == ["--current"]:
        print(current())
        return 0
    if argv[:1] in (["--verify"], ["--newer-than"]):
        problems = verify()
        if argv[0] == "--newer-than":
            if len(argv) < 2:
                raise SystemExit("--newer-than needs a git ref, e.g. origin/main")
            problem = newer_than(argv[1])
            problems += [problem] if problem else []
        for p in problems:
            print("version:", p)
        if not problems:
            print(f"version {current()} ok")
        return 1 if problems else 0
    version = argv[0].lstrip("v") if argv else next_version()
    for path in stamp(version):
        print("stamped", path)
    print(f"version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
