"""Live research: the orchestrator suggests models from the web, but nothing it says is trusted."""
import io
import json
import types

import pytest

import localforge.serve as serve_module
from localforge import model_fit, research, trust, upgrades
from localforge.hardware import GPU, HardwareProfile
from localforge.orchestrator import Conversation
from localforge.serve import StdioServer


def _hw():
    return HardwareProfile(os="Darwin", arch="arm64", cpu_cores=12, ram_gb=48, free_disk_gb=200.0,
                           gpus=[GPU(name="Apple M", vram_gb=36, backend="metal")], memory_bandwidth_gbps=273)


def _verify(cands, installed=("qwen2.5-coder:7b",), sizes=None, problems=None, managed=("qwen2.5-coder:7b",)):
    sizes = sizes if sizes is not None else {}
    problems = problems or {}
    return research.verify(
        cands, _hw(), set(installed), {"coding": "qwen2.5-coder:7b", "docs": "gemma3:12b", "general": "gemma3:12b"}, set(managed),
        registry_size=lambda n: sizes.get(n), fit_problem=lambda n: problems.get(n),
    )


def test_parse_reads_the_json_even_inside_prose_and_fences():
    reply = 'Here you go:\n```json\n{"candidates": [{"name": "a:1b", "modality": "coding", "why": "x"}]}\n```'
    assert research.parse(reply) == [{"name": "a:1b", "modality": "coding", "why": "x"}]
    assert research.parse("no json here") == []
    assert research.parse('{"candidates": "nope"}') == []


def test_a_real_fitting_model_is_suggested_with_what_it_would_replace():
    out = _verify([{"name": "Qwen3-Coder:30b", "modality": "coding", "why": "newer"}], sizes={"qwen3-coder:30b": 19.0})
    s = out.suggestions[0]
    assert (s.name, s.modality, s.size_gb, s.replaces, s.replaces_managed) == ("qwen3-coder:30b", "coding", 19.0, "qwen2.5-coder:7b", True)


def test_a_made_up_model_is_dropped_because_the_registry_has_no_manifest():
    out = _verify([{"name": "totally-real:99b", "modality": "coding", "why": "trust me"}], sizes={})
    assert not out.suggestions and "not found in the Ollama registry" in out.rejected[0]["reason"]


@pytest.mark.parametrize("name", ["deepseek-r1:14b", "qwen3-thinking:30b", "qwq:32b", "gpt-oss:20b"])
def test_reasoning_models_are_left_out(name):
    out = _verify([{"name": name, "modality": "general", "why": "smart"}], sizes={name: 10.0})
    assert not out.suggestions and "reasoning" in out.rejected[0]["reason"]


def test_a_model_that_does_not_fit_is_dropped_with_the_reason():
    out = _verify([{"name": "big:405b", "modality": "general", "why": "huge"}], sizes={"big:405b": 230.0}, problems={"big:405b": "needs 260 GB of memory"})
    assert not out.suggestions and out.rejected[0]["reason"] == "needs 260 GB of memory"


def test_bad_entries_and_duplicates_and_installed_models_are_ignored():
    out = _verify(
        [{"name": "noTag", "modality": "coding"}, {"name": "x:1b", "modality": "video"}, {"name": "qwen2.5-coder:7b", "modality": "coding"},
         {"name": "ok:7b", "modality": "docs", "why": "a"}, {"name": "ok:7b", "modality": "docs", "why": "again"}],
        sizes={"ok:7b": 4.0, "qwen2.5-coder:7b": 4.7},
    )
    assert [s.name for s in out.suggestions] == ["ok:7b"] and len(out.rejected) == 3


def test_nothing_usable_says_why():
    assert "usable list" in _verify([]).note
    assert "nothing new" in _verify([{"name": "nope:1b", "modality": "coding"}]).note


# --- through the desktop server ------------------------------------------------------------------


def _server(tmp_path, answer):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    run_fn = lambda *a, **k: types.SimpleNamespace(answer=answer, stats=types.SimpleNamespace())
    return StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=run_fn, scratch_root=tmp_path / "scratch", conversation=Conversation()), out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


@pytest.fixture
def fake(monkeypatch):
    state = {"installed": {"qwen2.5-coder:7b": 4.7}, "deleted": [], "pinned": {}}
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: True)
    monkeypatch.setattr(serve_module.OllamaBackend, "list_installed", lambda self: [{"name": n, "size": int(g * 1e9)} for n, g in state["installed"].items()])
    monkeypatch.setattr(serve_module.OllamaBackend, "ensure_available", lambda self, name, on_progress=None: state["installed"].__setitem__(name, 19.0))
    monkeypatch.setattr(serve_module.OllamaBackend, "delete", lambda self, name: (state["deleted"].append(name), state["installed"].pop(name, None)))
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: dict(state["installed"]))
    monkeypatch.setattr(model_fit, "download_problem", lambda name, *a, **k: None)
    monkeypatch.setattr(model_fit, "registry_size_gb", lambda name, timeout=6.0: {"qwen3-coder:30b": 19.0}.get(name))
    monkeypatch.setattr(serve_module, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(serve_module.StdioServer, "_record_usage", lambda self, stats: None)
    return state


def test_research_emits_only_verified_suggestions(tmp_path, fake):
    answer = json.dumps({"candidates": [{"name": "qwen3-coder:30b", "modality": "coding", "why": "newer"}, {"name": "ghost:7b", "modality": "docs", "why": "?"}]})
    upgrades.mark_managed("qwen2.5-coder:7b")
    server, out = _server(tmp_path, answer)
    server._research_models()
    result = _events(out, "research_result")[-1]
    assert result["ok"] and [s["name"] for s in result["suggestions"]] == ["qwen3-coder:30b"]
    assert result["suggestions"][0]["replaces"] == "qwen2.5-coder:7b" and result["suggestions"][0]["replaces_managed"]
    assert [r["name"] for r in result["rejected"]] == ["ghost:7b"]


def test_adopting_downloads_then_deletes_the_replaced_model_to_free_space(tmp_path, fake):
    upgrades.mark_managed("qwen2.5-coder:7b")
    server, out = _server(tmp_path, "{}")
    server._adopt_researched("qwen3-coder:30b", "coding", "qwen2.5-coder:7b", True)
    done = _events(out, "research_adopted")[-1]
    assert done["ok"] and fake["deleted"] == ["qwen2.5-coder:7b"] and "Removed qwen2.5-coder:7b" in done["message"]


def test_the_replaced_model_stays_unless_the_user_said_to_remove_it(tmp_path, fake):
    server, out = _server(tmp_path, "{}")
    server._adopt_researched("qwen3-coder:30b", "coding", "qwen2.5-coder:7b", False)
    assert fake["deleted"] == [] and "left installed" in _events(out, "research_adopted")[-1]["message"]


def test_the_replaced_model_stays_while_another_role_pins_it(tmp_path, fake, monkeypatch):
    from localforge import delegate_target as dt

    monkeypatch.setattr(dt, "get", lambda m: dt.DelegateTarget(kind="ollama", model="qwen2.5-coder:7b") if m == "docs" else dt.AUTO)
    server, out = _server(tmp_path, "{}")
    server._adopt_researched("qwen3-coder:30b", "coding", "qwen2.5-coder:7b", True)
    assert fake["deleted"] == [] and "docs role still uses it" in _events(out, "research_adopted")[-1]["message"]


def test_a_failed_download_adopts_nothing_and_removes_nothing(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(model_fit, "download_problem", lambda name, *a, **k: "needs 99 GB")
    server, out = _server(tmp_path, "{}")
    server._adopt_researched("qwen3-coder:30b", "coding", "qwen2.5-coder:7b", True)
    assert fake["deleted"] == [] and not _events(out, "research_adopted")[-1]["ok"]
