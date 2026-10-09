"""The setup review's "Check my setup": a plain pass/warn/fail list (borrowed from Paperclip's
environment test on its connect step)."""
import io
import json
import time

import localforge.serve as serve_module
from localforge import delegate_target, model_fit, trust
from localforge.orchestrator import Conversation
from localforge.serve import StdioServer, _setup_checks


def _by_id(checks):
    return {c["id"]: c for c in checks}


def _patch(monkeypatch, *, ready=True, running=True, installed=None, auto=None, free=100.0):
    monkeypatch.setattr(serve_module, "_orchestrator_readiness", lambda m, p: (ready, "" if ready else "no API key"))
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: running)
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: dict(installed or {}))
    monkeypatch.setattr(serve_module.shutil, "which", lambda name: "/usr/bin/ollama")
    rows = {m: {"target": "auto", "description": "d", **(auto or {}).get(m, {})} for m in delegate_target.ALL_MODALITIES}
    monkeypatch.setattr(serve_module, "_advanced_model_snapshot", lambda: rows)
    from localforge.hardware import HardwareProfile

    monkeypatch.setattr(serve_module, "detect_hardware", lambda: HardwareProfile(os="Darwin", arch="arm64", cpu_cores=8, ram_gb=32, free_disk_gb=free, gpus=[]))


def test_everything_ready(tmp_path, monkeypatch):
    auto = {m: {"auto_model": "qwen2.5-coder:7b", "installed": True, "disk_gb": 4.7} for m in delegate_target.MODALITIES}
    _patch(monkeypatch, installed={"qwen2.5-coder:7b": 4.7}, auto=auto)
    checks = _by_id(_setup_checks("claude-opus-5", None, tmp_path))
    assert checks["planner"]["state"] == "ok" and checks["ollama"]["state"] == "ok"
    assert all(checks[f"writer-{m}"]["state"] == "ok" for m in delegate_target.MODALITIES)
    assert "disk" not in checks


def test_a_missing_key_fails_the_planner_check_and_points_at_the_planner_step(tmp_path, monkeypatch):
    _patch(monkeypatch, ready=False)
    c = _by_id(_setup_checks("claude-opus-5", None, tmp_path))["planner"]
    assert c["state"] == "fail" and c["fix"] == "planner" and "no API key" in c["detail"]


def test_a_writer_not_downloaded_yet_is_a_warning_with_its_size_and_disk_is_checked(tmp_path, monkeypatch):
    auto = {m: {"auto_model": "qwen2.5-coder:7b", "installed": False, "disk_gb": 4.7} for m in delegate_target.MODALITIES}
    _patch(monkeypatch, installed={}, auto=auto, free=5.0)
    checks = _by_id(_setup_checks("claude-opus-5", None, tmp_path))
    assert checks["writer-coding"]["state"] == "warn" and "4.7 GB" in checks["writer-coding"]["detail"] and checks["writer-coding"]["fix"] == "writers"
    assert checks["disk"]["state"] == "warn"


def test_ollama_not_running_fails_when_local_models_are_needed(tmp_path, monkeypatch):
    auto = {m: {"auto_model": "qwen2.5-coder:7b", "installed": True, "disk_gb": 4.7} for m in delegate_target.MODALITIES}
    _patch(monkeypatch, running=False, auto=auto)
    assert _by_id(_setup_checks("claude-opus-5", None, tmp_path))["ollama"]["state"] == "fail"


def test_the_request_is_answered_over_the_protocol(tmp_path, monkeypatch):
    _patch(monkeypatch)
    out = io.StringIO()
    (tmp_path / "scratch").mkdir()
    trust.trust(tmp_path)
    server = StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=lambda *a, **k: None, scratch_root=tmp_path / "scratch", conversation=Conversation())
    server.handle({"type": "setup_checks_request"})
    for _ in range(100):
        events = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
        if any(e["type"] == "setup_checks" for e in events):
            break
        time.sleep(0.05)
    ev = [e for e in events if e["type"] == "setup_checks"][-1]
    assert ev["checks"] and isinstance(ev["ok"], bool)
