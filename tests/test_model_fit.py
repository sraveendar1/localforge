"""A model too big for this machine (memory, or the disk it would download to)
is refused with the reason, both when it's chosen and when a task would run
with it. Asked for: "if the model selected is too big for the machine, and
also from a storage point of view, say the issue and reject operations".
"""
import io
import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.model_fit as model_fit
import localforge.orchestrator as orch
from localforge import config, project_models, trust
from localforge import delegate_target as dt
from localforge.catalog import ModelEntry, fit_problem, load_catalog
from localforge.hardware import GPU, HardwareProfile
from localforge.serve import StdioServer

pytestmark = pytest.mark.real_model_fit  # these use the real check against hardware built here
runner = CliRunner()

MAC_16 = HardwareProfile(
    os="Darwin", arch="arm64", cpu_cores=8, ram_gb=16, free_disk_gb=200, gpus=[GPU(name="Apple", vram_gb=16, backend="metal")]
)


def _mac(free_disk=200.0, ram=16.0):
    return MAC_16.model_copy(update={"free_disk_gb": free_disk, "ram_gb": ram, "gpus": [GPU(name="Apple", vram_gb=ram, backend="metal")]})


@pytest.fixture
def machine(monkeypatch):
    """A 16 GB Mac, with nothing but qwen2.5-coder:7b on disk."""
    hw = _mac()
    monkeypatch.setattr(model_fit, "detect_hardware", lambda: hw)
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: {"qwen2.5-coder:7b": 4.7})
    return hw


def _entry(name):
    return next(m for m in load_catalog() if m.name == name)


# --- the check itself -------------------------------------------------------------


def test_a_model_that_fits_has_no_problem():
    assert fit_problem(_entry("qwen2.5-coder:7b"), _mac(), {"qwen2.5-coder:7b"}) is None


def test_too_big_for_memory_says_how_much_it_needs_and_what_there_is():
    why = fit_problem(_entry("qwen2.5-coder:32b"), _mac(), {"qwen2.5-coder:32b"})
    assert "needs about 24.5 GB of memory" in why and "16 GB of memory" in why and "12.0 GB" in why
    assert "isn't downloaded" not in why  # it's on disk: disk isn't the issue


def test_a_download_that_wont_fit_the_disk_says_so():
    why = fit_problem(_entry("qwen2.5-coder:14b"), _mac(free_disk=4.0), set())
    assert "isn't downloaded" in why and "about 9 GB" in why and "only 4 GB of disk is free" in why
    assert "memory" not in why  # it does fit in memory


def test_both_problems_are_reported_together():
    why = fit_problem(_entry("qwen2.5-coder:32b"), _mac(free_disk=4.0), set())
    assert "memory" in why and "only 4 GB of disk is free" in why


def test_an_installed_model_is_never_faulted_for_disk():
    assert fit_problem(_entry("qwen2.5-coder:14b"), _mac(free_disk=0.5, ram=32), {"qwen2.5-coder:14b"}) is None


def test_when_whats_on_disk_is_unknown_disk_isnt_judged():
    assert fit_problem(_entry("qwen2.5-coder:14b"), _mac(free_disk=0.5, ram=32), None) is None


def test_cloud_models_are_not_sized_against_the_machine():
    cloud = ModelEntry(name="gpt-5", modality="coding", runtime="api", provider="openai", min_vram_gb=0, min_ram_gb=0, disk_gb=0, quality_tier=0)
    assert fit_problem(cloud, _mac(ram=4, free_disk=0.1), set()) is None


def test_a_model_outside_the_catalog_is_sized_from_what_is_on_disk(machine):
    huge = model_fit.selection_problem("mystery:70b", machine, {"mystery:70b": 40.0})
    assert huge and "needs about" in huge
    assert model_fit.selection_problem("mystery:3b", machine, {"mystery:3b": 2.0}) is None


def test_a_model_we_know_nothing_about_is_left_alone(machine):
    assert model_fit.selection_problem("never-heard-of:1b", machine, {}) is None


# --- choosing one -----------------------------------------------------------------


def test_pinning_a_too_big_delegate_is_refused_and_nothing_is_saved(machine, tmp_path):
    with pytest.raises(dt.InvalidTarget) as info:
        dt.apply("coding", "qwen2.5-coder:32b")
    assert "can't run on this machine" in str(info.value) and "24.5 GB" in str(info.value) and "Nothing was changed" in str(info.value)
    assert dt.get("coding") == dt.AUTO


def test_pinning_a_model_whose_download_wont_fit_the_disk_is_refused(monkeypatch):
    hw = _mac(free_disk=3.0, ram=64)
    monkeypatch.setattr(model_fit, "detect_hardware", lambda: hw)
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: {})
    with pytest.raises(dt.InvalidTarget, match="only 3 GB of disk is free"):
        dt.apply("coding", "qwen2.5-coder:14b")


def test_pinning_one_that_fits_still_works(machine):
    assert dt.apply("coding", "qwen2.5-coder:7b").model == "qwen2.5-coder:7b"


def test_cli_refuses_a_too_big_delegate_with_the_reason(machine, monkeypatch):
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: set())
    monkeypatch.setattr(cli_module, "recommendations", lambda hw, installed=None: {})
    out = runner.invoke(cli_module.app, ["advanced-model", "coding", "qwen2.5-coder:32b"])
    assert out.exit_code == 1 and "can't run on this machine" in out.output and "24.5 GB" in out.output


def test_cli_model_refuses_a_too_big_orchestrator_and_saves_nothing(machine):
    with patch.object(cli_module, "_installed_model_names", return_value={"qwen2.5-coder:32b"}), patch.object(
        cli_module.cli_transport, "available", return_value=False
    ):
        out = runner.invoke(cli_module.app, ["model", "ollama/qwen2.5-coder:32b"])
    assert out.exit_code == 1 and "Can't use qwen2.5-coder:32b as the orchestrator" in out.output and "24.5 GB" in out.output
    assert config.FRONTIER_MODEL_ENV_VAR not in __import__("os").environ


def _server(folder):
    out = io.StringIO()
    scratch = folder.parent / (folder.name + "-scratch")
    scratch.mkdir(exist_ok=True)
    trust.trust(folder)
    return StdioServer(folder, "claude-opus-5", out=out, run_fn=lambda *a, **k: None, scratch_root=scratch, conversation=object()), out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


def test_desktop_refuses_a_too_big_orchestrator_and_keeps_the_current_one(machine, tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: {"qwen2.5-coder:32b"})
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda p: False)
    folder = tmp_path / "p"
    folder.mkdir()
    server, out = _server(folder)
    server.handle({"type": "set_model", "model": "ollama/qwen2.5-coder:32b"})
    assert "Can't use qwen2.5-coder:32b as the orchestrator" in _events(out, "error")[-1]["message"]
    assert server.frontier_model == "claude-opus-5"
    assert _events(out, "settings")[-1]["model"] == "claude-opus-5"  # the dropdown/chip goes back
    assert project_models.load(folder) is None


def test_desktop_refuses_a_too_big_delegate(machine, tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    server, out = _server(folder)
    server.handle({"type": "set_delegate_target", "modality": "coding", "target": "qwen2.5-coder:32b"})
    assert "can't run on this machine" in _events(out, "error")[-1]["message"]
    assert project_models.load(folder) is None


def test_the_pickers_are_told_which_models_cant_run_here(machine, tmp_path, monkeypatch):
    folder = tmp_path / "p"
    folder.mkdir()
    server, out = _server(folder)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: {"qwen2.5-coder:7b"})
    server.handle({"type": "delegate_options_request", "modality": "coding"})
    local = {m["name"]: m for m in _events(out, "delegate_options")[-1]["local"]}
    assert local["qwen2.5-coder:7b"]["problem"] is None
    assert "24.5 GB" in local["qwen2.5-coder:32b"]["problem"]
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda p: False)
    server._emit_orchestrator_options()
    opts = {o["id"]: o for o in _events(out, "orchestrator_options")[-1]["options"]}
    assert opts["ollama/qwen2.5-coder:7b"]["problem"] is None


# --- running with one already chosen ----------------------------------------------


def _saved_project(tmp_path, delegates=None, orchestrator=None):
    folder = tmp_path / "moved"
    (folder / ".localforge").mkdir(parents=True)
    trust.trust(folder)
    data = {}
    if delegates:
        data["delegates"] = delegates
    if orchestrator:
        data["orchestrator"] = orchestrator
    (folder / ".localforge" / "models.json").write_text(json.dumps(data))
    project_models.activate(folder)
    return folder


def test_a_task_is_rejected_before_anything_runs_when_a_chosen_delegate_cant_run_here(machine, tmp_path):
    """The project's settings came from a bigger machine."""
    _saved_project(tmp_path, delegates={"coding": "ollama:qwen2.5-coder:32b"})
    called = []
    with patch.object(orch, "completion", side_effect=lambda **kw: called.append(1)):
        with pytest.raises(model_fit.ModelNotUsable) as info:
            orch.run("build it", "claude-opus-5", hardware=machine)
    msg = str(info.value)
    assert "The coding model is set to qwen2.5-coder:32b" in msg and "can't run on this machine" in msg
    assert "/advanced-model coding auto" in msg and "Nothing was run" in msg
    assert called == []  # no paid call was made


def test_a_too_big_local_orchestrator_is_rejected_too(machine, tmp_path):
    with pytest.raises(model_fit.ModelNotUsable, match="The orchestrator is set to qwen2.5-coder:32b"):
        orch.run("hi", "ollama/qwen2.5-coder:32b", hardware=machine)


def test_every_problem_is_listed_not_just_the_first(machine, tmp_path):
    _saved_project(tmp_path, delegates={"coding": "ollama:qwen2.5-coder:32b", "docs": "ollama:qwen3-coder:30b"})
    problems = model_fit.preflight("claude-opus-5", machine)
    assert len(problems) == 2 and "coding" in problems[0] and "docs" in problems[1]


def test_auto_and_cloud_and_fitting_choices_pass_preflight(machine, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _saved_project(tmp_path, delegates={"coding": "ollama:qwen2.5-coder:7b", "docs": "api:anthropic:claude-haiku-4-5", "general": "auto"})
    assert model_fit.preflight("claude-opus-5", machine) == []


def test_the_terminal_prints_the_rejection_and_the_summary(machine, tmp_path, monkeypatch):
    folder = _saved_project(tmp_path, delegates={"coding": "ollama:qwen2.5-coder:32b"})
    monkeypatch.chdir(folder)
    monkeypatch.setattr(orch, "_installed_models", lambda: set())
    monkeypatch.setattr(orch, "detect_hardware", lambda: machine)
    out = runner.invoke(cli_module.app, ["run", "build it", "--model", "claude-opus-5", "--yes"])
    assert out.exit_code == 1
    text = " ".join(out.output.split())  # the terminal wraps long lines; ignore where
    assert "can't run on this machine" in text and "Nothing was run" in text and "Failed" in text


def test_desktop_reports_the_rejection_as_an_error_and_a_failed_summary(machine, tmp_path):
    folder = _saved_project(tmp_path, delegates={"coding": "ollama:qwen2.5-coder:32b"})
    out = io.StringIO()
    scratch = tmp_path / "s"
    scratch.mkdir()
    server = StdioServer(
        folder, "claude-opus-5", out=out, run_fn=lambda t, m, **k: orch.run(t, m, hardware=machine, **k),
        scratch_root=scratch, conversation=orch.Conversation(),
    )
    server._run_turn("build it")
    assert "can't run on this machine" in _events(out, "error")[-1]["message"]
    summary = _events(out, "task_summary")[-1]["summary"]
    assert summary["outcome"] == "failed" and "Nothing was run" in summary["error"]
