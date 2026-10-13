"""The Decision model ("System 1", like TypeSafe's Jev): a small local model that makes the pass/fail call."""
import io
import json

import pytest

import localforge.serve as serve_module
from localforge import config, model_fit, project_models, trust
from localforge.hardware import HardwareProfile
from localforge.orchestrator import Conversation
from localforge.serve import StdioServer
from localforge.tools import Dispatcher


def _hw():
    return HardwareProfile(os="Darwin", arch="arm64", cpu_cores=8, ram_gb=48, free_disk_gb=200.0, gpus=[])


def _server(tmp_path):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    return StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=lambda *a, **k: None, scratch_root=tmp_path / "scratch", conversation=Conversation()), out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


@pytest.fixture
def fake(monkeypatch):
    state = {"installed": {"gemma3:12b": 8.0, "tev1:0.8b": 0.6}}
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: True)
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: dict(state["installed"]))
    monkeypatch.setattr(model_fit, "download_problem", lambda name, *a, **k: None)
    monkeypatch.setattr(serve_module.OllamaBackend, "ensure_available", lambda self, name, on_progress=None: state["installed"].__setitem__(name, 0.6))
    return state


def test_a_pinned_decision_model_is_the_judge(monkeypatch):
    monkeypatch.setenv(config.DECISION_MODEL_ENV_VAR, "tev1:0.8b")
    d = Dispatcher(_hw(), installed={"gemma3:12b", "tev1:0.8b"})
    assert d.judge(exclude="gemma3:12b").name == "tev1:0.8b"


def test_the_writer_is_never_its_own_decision_model(monkeypatch):
    monkeypatch.setenv(config.DECISION_MODEL_ENV_VAR, "tev1:0.8b")
    d = Dispatcher(_hw(), installed={"gemma3:12b", "tev1:0.8b"})
    assert d.judge(exclude="tev1:0.8b").name != "tev1:0.8b"


def test_a_pinned_model_that_is_not_installed_falls_back_to_automatic(monkeypatch):
    monkeypatch.setenv(config.DECISION_MODEL_ENV_VAR, "tev1:0.8b")
    d = Dispatcher(_hw(), installed={"gemma3:12b"})
    assert d.judge(exclude="qwen2.5-coder:7b").name == "gemma3:12b"


def test_automatic_still_skips_a_paid_writer_but_a_chosen_decision_model_judges_it(monkeypatch):
    d = Dispatcher(_hw(), installed={"gemma3:12b", "tev1:0.8b"})
    d._last_delegate_cloud = True
    d._judge_check_calls = []
    monkeypatch.setattr(d, "judge", lambda exclude=None: d._judge_check_calls.append(1))  # would be asked only if not skipped
    assert d._judge_check("x", "y") is None and d._judge_check_calls == []  # automatic: skipped
    monkeypatch.setenv(config.DECISION_MODEL_ENV_VAR, "tev1:0.8b")
    d._judge_check("x", "y")
    assert d._judge_check_calls == [1]  # chosen on purpose: asked


def test_the_choice_is_saved_with_the_project_and_comes_back(tmp_path, monkeypatch):
    trust.trust(tmp_path)
    project_models.activate(tmp_path)
    project_models.save({config.DECISION_MODEL_ENV_VAR: "tev1:0.8b"})
    saved = json.loads(project_models.path(tmp_path).read_text())
    assert saved["decision"] == "tev1:0.8b"
    import os

    os.environ.pop(config.DECISION_MODEL_ENV_VAR, None)
    project_models.activate(tmp_path)
    assert project_models.snapshot()["decision"] == "tev1:0.8b"


def test_the_app_sets_it_from_what_is_installed(tmp_path, fake):
    server, out = _server(tmp_path)
    server._set_decision_model("tev1:0.8b")
    assert _events(out, "decision_model")[-1] == {"type": "decision_model", "model": "tev1:0.8b", "installed": True}
    assert _events(out, "decision_result")[-1]["ok"]
    server._set_decision_model("auto")
    assert _events(out, "decision_model")[-1]["model"] == "auto"


def test_a_model_that_is_not_installed_is_not_silently_pinned(tmp_path, fake, monkeypatch):
    monkeypatch.delenv(config.DECISION_MODEL_ENV_VAR, raising=False)
    server, out = _server(tmp_path)
    server._set_decision_model("nimble:7b")
    result = _events(out, "decision_result")[-1]
    assert not result["ok"] and "Download & use" in result["message"]
    assert Dispatcher._decision_pin() is None


def test_downloading_one_makes_it_the_decision_model(tmp_path, fake):
    server, out = _server(tmp_path)
    assert server._pull_model("nimble:7b", {"decision": True}, ctx="decision")
    assert Dispatcher._decision_pin() == "nimble:7b"
    assert "set as the Decision model" in _events(out, "pull_result")[-1]["message"]


def test_research_can_suggest_a_decision_model():
    from localforge import research

    out = research.verify(
        [{"name": "tev1:0.8b", "modality": "decision", "why": "Jev-style, tiny"}], _hw(), {"gemma3:12b"}, {"decision": "gemma3:12b"}, set(),
        registry_size=lambda n: 0.6, fit_problem=lambda n: None,
    )
    assert [(s.name, s.modality) for s in out.suggestions] == [("tev1:0.8b", "decision")]
