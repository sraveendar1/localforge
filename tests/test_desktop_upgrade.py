"""/upgrade in the desktop app: check whether a better open-weight model fits this machine, then
(on request) download it and remove the one LocalForge installed that nothing uses any more."""
import io
import json
import threading

import pytest

import localforge.serve as serve_module
from localforge import model_fit, trust, upgrades
from localforge.hardware import HardwareProfile
from localforge.orchestrator import Conversation
from localforge.serve import StdioServer


def _hw():
    from localforge.hardware import GPU

    return HardwareProfile(os="Darwin", arch="arm64", cpu_cores=12, ram_gb=48, free_disk_gb=200.0,
                           gpus=[GPU(name="Apple M", vram_gb=36, backend="metal")], memory_bandwidth_gbps=273)


def _server(tmp_path):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    return StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=lambda *a, **k: None, scratch_root=tmp_path / "scratch", conversation=Conversation()), out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


@pytest.fixture
def fake(monkeypatch):
    state = {"installed": {"qwen2.5-coder:7b": 4.7}, "pulled": [], "deleted": [], "running": True, "fail": False}
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: state["running"])
    monkeypatch.setattr(serve_module.OllamaBackend, "list_installed", lambda self: [{"name": n, "size": int(g * 1e9)} for n, g in state["installed"].items()])

    def ensure(self, name, on_progress=None):
        if state["fail"]:
            raise RuntimeError("network down")
        state["pulled"].append(name)
        state["installed"][name] = 19.0

    monkeypatch.setattr(serve_module.OllamaBackend, "ensure_available", ensure)
    monkeypatch.setattr(serve_module.OllamaBackend, "delete", lambda self, name: (state["deleted"].append(name), state["installed"].pop(name, None)))
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: dict(state["installed"]))
    monkeypatch.setattr(serve_module, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(model_fit, "download_problem", lambda name: None)
    return state


def test_check_reports_a_better_model_and_what_it_would_remove(tmp_path, fake):
    upgrades.mark_managed("qwen2.5-coder:7b")
    server, out = _server(tmp_path)
    server._upgrade_check(say=True)
    plan = _events(out, "upgrade_plan")[-1]
    assert plan["ok"] and not plan["empty"]
    assert plan["upgrades"] and plan["upgrades"][0]["old"] == "qwen2.5-coder:7b"
    assert "qwen2.5-coder:7b" in plan["remove"]
    assert any("Upgrade plan" in e["text"] and "/upgrade now" in e["text"] for e in _events(out, "system_text"))


def test_apply_downloads_first_then_removes_what_localforge_installed(tmp_path, fake):
    upgrades.mark_managed("qwen2.5-coder:7b")
    server, out = _server(tmp_path)
    server._upgrade_apply()
    new = fake["pulled"]
    assert new and fake["deleted"] == ["qwen2.5-coder:7b"]
    result = _events(out, "upgrade_result")[-1]
    assert result["ok"] and "removed qwen2.5-coder:7b" in result["message"]


def test_a_failed_download_removes_nothing(tmp_path, fake):
    upgrades.mark_managed("qwen2.5-coder:7b")
    fake["fail"] = True
    server, out = _server(tmp_path)
    server._upgrade_apply()
    assert fake["deleted"] == []
    result = _events(out, "upgrade_result")[-1]
    assert not result["ok"] and "Nothing was removed" in result["message"]


def test_a_model_the_user_installed_themselves_is_never_removed(tmp_path, fake):
    server, out = _server(tmp_path)  # not marked as managed
    server._upgrade_apply()
    assert fake["deleted"] == []


def test_removal_waits_while_a_task_is_running(tmp_path, fake, monkeypatch):
    upgrades.mark_managed("qwen2.5-coder:7b")
    server, out = _server(tmp_path)
    monkeypatch.setattr(StdioServer, "busy", property(lambda self: True))
    server._upgrade_apply()
    assert fake["deleted"] == []
    assert "left qwen2.5-coder:7b in place" in _events(out, "upgrade_result")[-1]["message"]


def test_ollama_not_running_is_said_plainly(tmp_path, fake):
    fake["running"] = False
    server, out = _server(tmp_path)
    server._upgrade_check(say=True)
    assert not _events(out, "upgrade_plan")[-1]["ok"]
    assert any("Ollama isn't running" in e["text"] for e in _events(out, "system_text"))


def test_slash_upgrade_answers_in_the_chat_and_now_applies(tmp_path, fake):
    upgrades.mark_managed("qwen2.5-coder:7b")
    server, out = _server(tmp_path)
    server.handle_upgrade_command("")
    server.handle_upgrade_command("now")
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(timeout=5)
    texts = " ".join(e["text"] for e in _events(out, "system_text"))
    assert "Checking whether a better" in texts and "Upgrading local models" in texts
    assert fake["deleted"] == ["qwen2.5-coder:7b"]
