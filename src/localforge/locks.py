"""Locks that work between processes (and on both Windows and macOS/Linux).

localforge's desktop app can have several windows open, each with its own backend process. Three
things must then be shared safely:

* a project folder is opened by one backend at a time (`acquire_project`),
* local model calls take turns across all backends (`model_guard`), because one Ollama can't hold two
  models' weights on a small machine and evicts the one that is mid-answer,
* settings files are written whole or not at all (`atomic_write`, `config_lock`).

They use the operating system's own file locks, so a process that crashes releases its lock by
itself: there is no stale lock file to clean up. The owner's details are written beside the lock
(`<lock>.owner`) so a refused window can say who has it. Nothing here writes inside a project folder.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

from localforge import config

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import msvcrt
else:
    import fcntl

# Windows can only lock a byte range, and a locked range can't be read by others, so the byte that is
# locked lives far past the (empty) file's end; the owner's details are in a separate file.
_WIN_LOCK_OFFSET = 1 << 30


class FileLock:
    """An exclusive lock on a file, held by this process until `release()` (or until it exits)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def _try(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if IS_WINDOWS:
                os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def acquire(self, timeout: float = 0.0, poll: float = 0.1, on_wait: Callable[[bool], None] | None = None, wait_report_after: float = 2.0) -> bool:
        """Take the lock, waiting up to `timeout` seconds. `on_wait(True)` is called once if the wait
        goes past `wait_report_after` seconds, and `on_wait(False)` when it ends either way."""
        if self._fd is not None:
            return True
        start = time.monotonic()
        reported = False
        try:
            while True:
                if self._try():
                    return True
                waited = time.monotonic() - start
                if waited >= timeout:
                    return False
                if on_wait is not None and not reported and waited >= wait_report_after:
                    reported = True
                    on_wait(True)
                time.sleep(poll)
        finally:
            if reported and on_wait is not None:
                on_wait(False)

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if IS_WINDOWS:
                os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(fd)

    def __enter__(self) -> "FileLock":
        if not self.acquire(timeout=30):
            raise TimeoutError(f"couldn't take {self.path.name}")
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def locks_dir() -> Path:
    return config.CONFIG_DIR / "locks"


# --- atomic writes ------------------------------------------------------------------------------


def atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    """Write `text` so a reader (or a second process) sees the old file or the new one, never half."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text)
        if mode is not None:
            with contextlib.suppress(OSError):
                tmp.chmod(mode)
        os.replace(tmp, path)  # atomic on Windows and POSIX
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


_config_thread_lock = threading.RLock()


@contextlib.contextmanager
def config_lock(timeout: float = 10.0) -> Iterator[None]:
    """Serialize read-modify-write of the shared settings files across windows. If the lock can't be
    had in time the write goes ahead anyway: losing a settings change is worse than a rare overlap."""
    lock = FileLock(locks_dir() / "settings.lock")
    with _config_thread_lock:
        got = lock.acquire(timeout=timeout)
        try:
            yield
        finally:
            if got:
                lock.release()


# --- one backend per project folder --------------------------------------------------------------


def _project_key(root: Path) -> str:
    resolved = str(Path(root).resolve())
    if IS_WINDOWS or sys.platform == "darwin":
        resolved = resolved.lower()  # these file systems ignore case by default
    return hashlib.sha1(resolved.encode()).hexdigest()[:16]


def _owner_path(lock_path: Path) -> Path:
    return lock_path.with_name(lock_path.name + ".owner")


def read_owner(lock_path: Path) -> dict:
    try:
        data = json.loads(_owner_path(lock_path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_owner(lock_path: Path, **details) -> None:
    atomic_write(_owner_path(lock_path), json.dumps({"pid": os.getpid(), "started": time.time(), **details}))


class ProjectBusy(Exception):
    """The folder is already open in another window."""

    def __init__(self, folder: Path, owner: dict):
        self.folder, self.owner = Path(folder), owner
        pid = owner.get("pid")
        super().__init__(f"{folder} is already open in another LocalForge window" + (f" (process {pid})" if pid else "") + ".")


def acquire_project(root: Path) -> FileLock:
    """Claim `root` for this backend, or raise ProjectBusy. Held until the process ends."""
    lock = FileLock(locks_dir() / f"project-{_project_key(root)}.lock")
    if not lock.acquire(timeout=0):
        raise ProjectBusy(root, read_owner(lock.path))
    write_owner(lock.path, folder=str(Path(root).resolve()))
    return lock


# --- local model calls take turns across windows -------------------------------------------------

MODEL_LOCK_TIMEOUT = float(os.environ.get("LOCALFORGE_MODEL_LOCK_TIMEOUT", "900"))
_wait_reporter: Callable[[bool, dict], None] | None = None


def set_wait_reporter(fn: Callable[[bool, dict], None] | None) -> None:
    """`fn(waiting, owner)` is called when a local model call has been queued behind another
    window's for a couple of seconds, and again when it stops waiting."""
    global _wait_reporter
    _wait_reporter = fn


@contextlib.contextmanager
def model_guard(thread_lock: threading.Lock) -> Iterator[None]:
    """Around every call that runs a local model: first this process's own lock (threads), then the
    machine-wide one (other windows). If another window holds it past the timeout the call goes ahead
    unlocked rather than hanging forever."""
    with thread_lock:
        lock = FileLock(locks_dir() / "model.lock")

        def report(waiting: bool) -> None:
            if _wait_reporter is not None:
                try:
                    _wait_reporter(waiting, read_owner(lock.path))
                except Exception:  # noqa: BLE001 - a status nicety must never break a call
                    pass

        got = lock.acquire(timeout=MODEL_LOCK_TIMEOUT, on_wait=report)
        if got:
            with contextlib.suppress(OSError):
                write_owner(lock.path)
        try:
            yield
        finally:
            if got:
                lock.release()
