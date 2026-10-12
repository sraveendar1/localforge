"""Several windows = several backend processes. They must not open the same project twice, must take
turns on the local model, and must not corrupt shared settings (works on Windows and macOS/Linux)."""
import json
import multiprocessing as mp
import os
import threading
import time

import pytest

from localforge import config, locks, trust


def _hold(config_dir, folder, ready, release):
    from localforge import config as c, locks as l

    c.CONFIG_DIR = type(c.CONFIG_DIR)(config_dir)
    lock = l.acquire_project(folder)
    ready.set()
    release.wait(20)
    lock.release()


def _hold_model(config_dir, ready, release):
    from localforge import config as c, locks as l

    c.CONFIG_DIR = type(c.CONFIG_DIR)(config_dir)
    lock = l.FileLock(l.locks_dir() / "model.lock")
    assert lock.acquire()
    ready.set()
    release.wait(20)
    lock.release()


def _spawn(target, *args):
    ctx = mp.get_context("spawn")
    return ctx.Process(target=target, args=args, daemon=True), ctx


def test_a_second_process_cannot_open_the_same_folder(tmp_path):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    folder = tmp_path / "proj"
    folder.mkdir()
    p = ctx.Process(target=_hold, args=(str(config.CONFIG_DIR), str(folder), ready, release), daemon=True)
    p.start()
    try:
        assert ready.wait(20)
        with pytest.raises(locks.ProjectBusy) as exc:
            locks.acquire_project(folder)
        assert exc.value.owner["pid"] == p.pid and "already open" in str(exc.value)
    finally:
        release.set()
        p.join(10)
    lock = locks.acquire_project(folder)  # the owner is gone: it can be opened again
    lock.release()


def test_a_killed_owner_does_not_leave_the_folder_locked(tmp_path):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    folder = tmp_path / "proj2"
    folder.mkdir()
    p = ctx.Process(target=_hold, args=(str(config.CONFIG_DIR), str(folder), ready, release), daemon=True)
    p.start()
    assert ready.wait(20)
    p.kill()
    p.join(10)
    locks.acquire_project(folder).release()


def test_different_folders_do_not_block_each_other(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    la, lb = locks.acquire_project(a), locks.acquire_project(b)
    la.release(), lb.release()


def test_a_second_window_waits_for_the_model_then_goes(tmp_path):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold_model, args=(str(config.CONFIG_DIR), ready, release), daemon=True)
    p.start()
    assert ready.wait(20)
    seen = []
    locks.set_wait_reporter(lambda waiting, owner: seen.append(waiting))
    threading.Timer(2.6, release.set).start()
    started = time.monotonic()
    try:
        with locks.model_guard(threading.Lock()):
            waited = time.monotonic() - started
    finally:
        locks.set_wait_reporter(None)
        p.join(10)
    assert waited >= 2.0
    assert seen == [True, False]  # told it was waiting, then that it stopped


def test_the_model_lock_gives_up_waiting_rather_than_hang(tmp_path, monkeypatch):
    holder = locks.FileLock(locks.locks_dir() / "model.lock")
    assert holder.acquire()
    other = locks.FileLock(locks.locks_dir() / "model.lock")
    monkeypatch.setattr(locks, "MODEL_LOCK_TIMEOUT", 0.3)
    try:
        # same process, so the OS lock is per file description: a second descriptor must be refused
        started = time.monotonic()
        with locks.model_guard(threading.Lock()):
            pass
        assert time.monotonic() - started < 5
    finally:
        holder.release()
    assert other.acquire() is True
    other.release()


def test_threads_in_one_process_still_take_turns():
    order, lock = [], threading.Lock()

    def work(n):
        with locks.model_guard(lock):
            order.append(("in", n))
            time.sleep(0.05)
            order.append(("out", n))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(order[i][0] == "in" and order[i + 1][0] == "out" and order[i][1] == order[i + 1][1] for i in range(0, 6, 2))


def test_atomic_write_leaves_no_temp_files_and_replaces_whole(tmp_path):
    f = tmp_path / "x.json"
    locks.atomic_write(f, "one")
    locks.atomic_write(f, "two")
    assert f.read_text() == "two" and [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_concurrent_trust_and_config_writes_lose_nothing(tmp_path):
    folders = [tmp_path / f"f{i}" for i in range(12)]
    for f in folders:
        f.mkdir()
    ts = [threading.Thread(target=trust.trust, args=(f,)) for f in folders]
    ts += [threading.Thread(target=config.save, args=({f"KEY_{i}": str(i)},)) for i in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(trust.trusted_folders()) >= 12
    saved = config.CONFIG_FILE.read_text()
    assert all(f"KEY_{i}={i}" in saved for i in range(12))
    if os.name != "nt":
        assert oct(config.CONFIG_FILE.stat().st_mode & 0o777) == "0o600"


def test_serve_tells_a_second_window_the_folder_is_already_open(tmp_path, capsys, monkeypatch):
    import io
    import sys

    import localforge.serve as serve_module

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    folder = tmp_path / "proj3"
    folder.mkdir()
    p = ctx.Process(target=_hold, args=(str(config.CONFIG_DIR), str(folder), ready, release), daemon=True)
    p.start()
    try:
        assert ready.wait(20)
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        serve_module.serve_stdio(folder, "claude-opus-5")
        event = json.loads(out.getvalue().splitlines()[0])
    finally:
        release.set()
        p.join(10)
    assert event["type"] == "project_busy" and event["pid"] == p.pid and "already open" in event["message"]
