"""Downloading open-weight models from the desktop setup screen: a model that
isn't recommended for this machine is refused *before* anything is downloaded,
a good one downloads with progress and can be put to use, and a model typed in
by hand can be used once it's on disk."""
import io
import json

import pytest

import localforge.serve as serve_module
from localforge import delegate_target as dt
from localforge import model_fit, trust
from localforge.catalog import load_catalog
from localforge.hardware import HardwareProfile
from localforge.serve import StdioServer


def _hw(ram=16.0, disk=200.0):
    return HardwareProfile(os="Darwin", arch="arm64", cpu_cores=8, ram_gb=ram, free_disk_gb=disk, gpus=[])


def _server(tmp_path):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    server = StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=lambda *a, **k: None, scratch_root=tmp_path / "scratch",
                         conversation=__import__("localforge.orchestrator", fromlist=["x"]).Conversation())
    return server, out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    state = {"installed": {}, "pulled": [], "running": True}
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: state["running"])
    monkeypatch.setattr(serve_module.OllamaBackend, "list_installed",
                        lambda self: [{"name": n, "size": int(g * 1e9)} for n, g in state["installed"].items()])

    def ensure(self, name, on_progress=None):
        state["pulled"].append(name)
        if on_progress:
            on_progress({"status": "downloading", "total": 100, "completed": 40})
            on_progress({"status": "success"})
        state["installed"][name] = 4.0

    monkeypatch.setattr(serve_module.OllamaBackend, "ensure_available", ensure)
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: {n: g for n, g in state["installed"].items()})
    monkeypatch.setattr(serve_module.model_fit, "installed_sizes", lambda: {n: g for n, g in state["installed"].items()})
    monkeypatch.setattr(model_fit, "detect_hardware", lambda: _hw())
    monkeypatch.setattr("localforge.upgrades.mark_managed", lambda name: state.setdefault("managed", []).append(name))
    return state


def test_a_model_too_big_for_the_machine_is_never_downloaded(tmp_path, fake_ollama):
    server, out = _server(tmp_path)
    big = max((m for m in load_catalog() if m.runtime == "ollama"), key=lambda m: m.disk_gb)
    server._pull_model(big.name, None)
    result = _events(out, "pull_result")[-1]
    assert result["ok"] is False and result["blocked"] is True
    assert "isn't recommended for this machine" in result["message"]
    assert fake_ollama["pulled"] == []


def test_not_enough_disk_blocks_the_download(tmp_path, fake_ollama, monkeypatch):
    monkeypatch.setattr(model_fit, "detect_hardware", lambda: _hw(ram=64.0, disk=0.5))
    server, out = _server(tmp_path)
    small = min((m for m in load_catalog() if m.runtime == "ollama" and m.modality == "coding"), key=lambda m: m.disk_gb)
    server._pull_model(small.name, None)
    assert fake_ollama["pulled"] == []
    assert "disk" in _events(out, "pull_result")[-1]["message"]


def test_a_typed_model_whose_size_cannot_be_confirmed_is_refused(tmp_path, fake_ollama, monkeypatch):
    monkeypatch.setattr(model_fit, "registry_size_gb", lambda name, timeout=6.0: None)
    server, out = _server(tmp_path)
    server._pull_model("someone/mystery-model:7b", None)
    result = _events(out, "pull_result")[-1]
    assert result["ok"] is False and "confirm its size" in result["message"]
    assert fake_ollama["pulled"] == []


def test_a_typed_model_that_is_small_enough_downloads(tmp_path, fake_ollama, monkeypatch):
    monkeypatch.setattr(model_fit, "registry_size_gb", lambda name, timeout=6.0: 2.0)
    server, out = _server(tmp_path)
    server._pull_model("tinyllama", None)  # no tag: becomes :latest
    assert fake_ollama["pulled"] == ["tinyllama:latest"]
    assert _events(out, "pull_result")[-1]["ok"] is True
    assert _events(out, "pull_progress")[0]["status"] == "starting"
    assert any(e["completed"] == 40 and e["total"] == 100 for e in _events(out, "pull_progress"))
    assert fake_ollama["managed"] == ["tinyllama:latest"]


def test_a_downloaded_model_can_be_set_for_a_task_type(tmp_path, fake_ollama):
    server, out = _server(tmp_path)
    good = min((m for m in load_catalog() if m.runtime == "ollama" and m.modality == "coding"), key=lambda m: m.disk_gb)
    server._pull_model(good.name, {"modality": "coding"})
    assert dt.get("coding").model == good.name
    assert "set for coding" in _events(out, "pull_result")[-1]["message"]


def test_a_typed_installed_model_can_be_chosen_for_a_task(tmp_path, fake_ollama):
    fake_ollama["installed"]["tinyllama:latest"] = 0.6
    dt.apply("docs", "tinyllama:latest")
    assert dt.get("docs").model == "tinyllama:latest"


def test_an_uninstalled_typed_model_cannot_be_chosen_for_a_task(tmp_path, fake_ollama):
    with pytest.raises(dt.InvalidTarget, match="isn't installed"):
        dt.apply("docs", "tinyllama:latest")


def test_nonsense_names_and_a_stopped_ollama_are_refused(tmp_path, fake_ollama):
    server, out = _server(tmp_path)
    server._pull_model("rm -rf /", None)
    assert _events(out, "pull_result")[-1]["blocked"] is True
    fake_ollama["running"] = False
    server._pull_model("llama3.1:8b", None)
    assert "Ollama isn't running" in _events(out, "pull_result")[-1]["message"]
    assert fake_ollama["pulled"] == []


def test_the_setup_status_suggests_what_fits_and_greys_out_what_does_not(tmp_path, fake_ollama):
    status = serve_module._setup_status("claude-opus-5", None)
    suggested = status["ollama"]["suggested"]
    assert suggested
    assert any(s["problem"] is None for s in suggested)
    assert all(s["name"] not in fake_ollama["installed"] for s in suggested)


def test_finishing_setup_writes_the_project_file(tmp_path, fake_ollama):
    from localforge import project_models

    server, out = _server(tmp_path)
    assert project_models.source() == "defaults"
    server.handle({"type": "finish_project_setup"})
    assert project_models.source() == "project"
    assert _events(out, "advanced_model")[-1]["source"] == "project"


def test_auto_rows_say_which_model_they_mean(tmp_path, fake_ollama):
    snap = serve_module._advanced_model_snapshot()
    assert snap["coding"].get("auto_model")
    assert snap["coding"]["installed"] is False
