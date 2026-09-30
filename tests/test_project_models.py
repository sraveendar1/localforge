"""Each project keeps its own models in `.localforge/models.json`: opening a
folder uses them, changing a model there updates that file (and only that
one), and a folder that has none uses your defaults.

Asked for: "for every project, a local file that stores what the models are
set to, so if I change the project folder it references it, and if I change
it, the file gets updated."
"""
import io
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge import config, memory, project_models, trust
from localforge import delegate_target as dt
from localforge.serve import StdioServer

runner = CliRunner()


def _folder(tmp_path, name="proj"):
    folder = tmp_path / name
    folder.mkdir()
    trust.trust(folder)
    return folder


def _write(folder, data):
    d = folder / ".localforge"
    d.mkdir(exist_ok=True)
    (d / "models.json").write_text(json.dumps(data))


# --- the file ---------------------------------------------------------------------


def test_a_folder_with_no_file_uses_the_defaults(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "claude-opus-5")
    assert project_models.activate(folder) is False
    assert project_models.source() == "defaults"
    assert os.environ[config.FRONTIER_MODEL_ENV_VAR] == "claude-opus-5"


def test_the_first_change_writes_the_whole_set_into_the_project(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "claude-opus-5")
    monkeypatch.setenv(config.AUTH_METHOD_ENV_VAR, config.AUTH_API_KEY)
    monkeypatch.setenv(config.FRONTIER_PROVIDER_ENV_VAR, "anthropic")
    project_models.activate(folder)
    dt.apply("coding", "qwen2.5-coder:7b")
    saved = json.loads((folder / ".localforge" / "models.json").read_text())
    assert saved["orchestrator"] == {"model": "claude-opus-5", "auth_method": "api_key", "provider": "anthropic"}
    assert saved["delegates"] == {
        "coding": "ollama:qwen2.5-coder:7b", "docs": "auto", "general": "auto", "image": "auto", "video": "auto",
    }
    assert project_models.source() == "project"
    assert (folder / ".localforge" / ".gitignore").exists()  # git ignores it by itself, like the rest of .localforge


def test_a_change_updates_the_file_and_leaves_the_global_defaults_alone(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    project_models.activate(folder)
    dt.apply("coding", "qwen2.5-coder:7b")
    dt.apply("docs", "llama3.1:8b")
    dt.apply("coding", "auto")
    saved = json.loads((folder / ".localforge" / "models.json").read_text())
    assert saved["delegates"]["coding"] == "auto" and saved["delegates"]["docs"] == "ollama:llama3.1:8b"
    assert not config.CONFIG_FILE.exists() or "TARGET" not in config.CONFIG_FILE.read_text()


def test_reopening_the_folder_brings_its_models_back(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    project_models.activate(folder)
    dt.apply("coding", "qwen2.5-coder:7b")
    # a new process: environment gone, only the file remains
    for name in list(os.environ):
        if name.startswith("LOCALFORGE_"):
            os.environ.pop(name)  # (not monkeypatch.delenv: its undo would put back a value the code wrote)
    assert dt.get("coding") == dt.AUTO
    project_models.deactivate()
    assert project_models.activate(folder) is True
    assert dt.get("coding") == dt.DelegateTarget(kind="ollama", model="qwen2.5-coder:7b")


def test_two_projects_do_not_share_models(tmp_path, monkeypatch):
    a, b = _folder(tmp_path, "a"), _folder(tmp_path, "b")
    project_models.activate(a)
    dt.apply("coding", "qwen2.5-coder:7b")
    os.environ.pop("LOCALFORGE_CODING_TARGET", None)
    project_models.deactivate()
    project_models.activate(b)
    assert dt.get("coding") == dt.AUTO
    dt.apply("docs", "llama3.1:8b")
    assert json.loads((a / ".localforge" / "models.json").read_text())["delegates"]["docs"] == "auto"


def test_the_project_file_beats_the_global_default(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "claude-opus-5")
    _write(folder, {"orchestrator": {"model": "claude-sonnet-5", "auth_method": "api_key", "provider": "anthropic"}})
    project_models.activate(folder)
    assert os.environ[config.FRONTIER_MODEL_ENV_VAR] == "claude-sonnet-5"


def test_a_bad_file_is_ignored_not_fatal(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    (folder / ".localforge").mkdir()
    (folder / ".localforge" / "models.json").write_text("{not json")
    assert project_models.activate(folder) is False
    _write(folder, {"orchestrator": "nonsense", "delegates": {"coding": 5, "docs": "garbage", "nope": "auto"}})
    project_models.activate(folder)
    assert dt.get("docs") == dt.AUTO and dt.get("coding") == dt.AUTO


def test_without_an_active_project_a_change_is_a_global_default(tmp_path):
    project_models.deactivate()
    dt.apply("coding", "qwen2.5-coder:7b")
    assert "LOCALFORGE_CODING_TARGET=ollama:qwen2.5-coder:7b" in config.CONFIG_FILE.read_text()


def test_setup_sets_global_defaults_even_inside_a_project(tmp_path):
    folder = _folder(tmp_path)
    project_models.activate(folder)
    dt.apply("coding", "qwen2.5-coder:7b", scope="global")
    assert "LOCALFORGE_CODING_TARGET=ollama:qwen2.5-coder:7b" in config.CONFIG_FILE.read_text()
    assert not (folder / ".localforge" / "models.json").exists()


# --- the terminal -----------------------------------------------------------------


def test_cli_advanced_model_change_is_saved_for_the_folder_and_says_so(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    monkeypatch.chdir(folder)
    monkeypatch.setattr(cli_module, "detect_hardware", lambda: None, raising=False)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: set())
    monkeypatch.setattr(cli_module, "recommendations", lambda hw, installed=None: {})
    first = runner.invoke(cli_module.app, ["advanced-model"])
    assert "Using your defaults" in first.output
    ok = runner.invoke(cli_module.app, ["advanced-model", "coding", "qwen2.5-coder:7b"])
    assert ok.exit_code == 0 and ".localforge/models.json" in ok.output
    assert json.loads((folder / ".localforge" / "models.json").read_text())["delegates"]["coding"] == "ollama:qwen2.5-coder:7b"
    again = runner.invoke(cli_module.app, ["advanced-model"])
    assert "Saved for this project" in again.output and "qwen2.5-coder:7b" in again.output


def test_cli_opening_a_folder_applies_its_saved_models(tmp_path, monkeypatch):
    folder = _folder(tmp_path)
    _write(folder, {"delegates": {"docs": "ollama:llama3.1:8b"}})
    monkeypatch.chdir(folder)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: set())
    monkeypatch.setattr(cli_module, "recommendations", lambda hw, installed=None: {})
    out = runner.invoke(cli_module.app, ["advanced-model"]).output
    assert "llama3.1:8b" in out and "Saved for this project" in out


@pytest.mark.untrusted
def test_an_untrusted_folders_file_is_not_applied(tmp_path, monkeypatch):
    """A cloned repo must not be able to pick a paid model on the user's keys."""
    folder = tmp_path / "cloned"
    folder.mkdir()
    _write(folder, {"delegates": {"coding": "api:openai:gpt-image-1"}, "orchestrator": {"model": "gpt-5"}})
    monkeypatch.chdir(folder)
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: set())
    monkeypatch.setattr(cli_module, "recommendations", lambda hw, installed=None: {})
    out = runner.invoke(cli_module.app, ["advanced-model"]).output
    assert "gpt-image-1" not in out and "isn't trusted yet" in out
    assert os.environ.get(config.FRONTIER_MODEL_ENV_VAR) != "gpt-5"


def test_slash_model_in_the_terminal_saves_to_the_project_not_the_defaults(tmp_path, monkeypatch):
    from unittest.mock import patch

    folder = _folder(tmp_path)
    monkeypatch.chdir(folder)
    with patch.object(cli_module, "_installed_model_names", return_value={"gemma3:4b"}), patch.object(
        cli_module.cli_transport, "available", return_value=False
    ):
        assert runner.invoke(cli_module.app, ["model", "ollama/gemma3:4b"]).exit_code == 0
    assert project_models.load(folder)["orchestrator"]["model"] == "ollama/gemma3:4b"
    assert not config.CONFIG_FILE.exists() or "gemma3" not in config.CONFIG_FILE.read_text()


# --- the desktop backend ----------------------------------------------------------


def _server(folder, model="claude-opus-5", **kw):
    out = io.StringIO()
    scratch = folder.parent / (folder.name + "-scratch")
    scratch.mkdir(exist_ok=True)
    server = StdioServer(folder, model, out=out, run_fn=lambda *a, **k: None, scratch_root=scratch, conversation=object(), **kw)
    return server, out


def _events(out, kind):
    return [e for e in (json.loads(l) for l in out.getvalue().splitlines() if l.strip()) if e["type"] == kind]


def test_desktop_uses_the_projects_saved_orchestrator_when_the_folder_opens(tmp_path):
    folder = _folder(tmp_path)
    _write(folder, {"orchestrator": {"model": "claude-sonnet-5", "auth_method": "api_key", "provider": "anthropic"}})
    server, _ = _server(folder)
    assert server.frontier_model == "claude-sonnet-5"


def test_an_explicit_model_beats_the_projects(tmp_path):
    folder = _folder(tmp_path)
    _write(folder, {"orchestrator": {"model": "claude-sonnet-5", "auth_method": "api_key", "provider": "anthropic"}})
    server, _ = _server(folder, model="gpt-5", model_explicit=True)
    assert server.frontier_model == "gpt-5"


@pytest.mark.untrusted
def test_desktop_applies_the_projects_models_only_after_the_folder_is_trusted(tmp_path):
    folder = tmp_path / "cloned"
    folder.mkdir()
    _write(folder, {"orchestrator": {"model": "claude-sonnet-5", "auth_method": "api_key", "provider": "anthropic"},
                    "delegates": {"docs": "ollama:llama3.1:8b"}})
    server, out = _server(folder)
    assert server.frontier_model == "claude-opus-5" and dt.get("docs") == dt.AUTO
    server.handle({"type": "trust_response", "trust": True})
    assert server.frontier_model == "claude-sonnet-5" and dt.get("docs").model == "llama3.1:8b"
    assert _events(out, "settings")[-1]["model"] == "claude-sonnet-5"
    assert _events(out, "advanced_model")[-1]["source"] == "project"


def test_desktop_model_switch_updates_the_projects_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_module, "_installed_model_names", lambda o: {"gemma3:4b"})
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda p: False)
    folder = _folder(tmp_path)
    server, out = _server(folder)
    server.handle({"type": "set_model", "model": "ollama/gemma3:4b"})
    assert project_models.load(folder)["orchestrator"] == {"model": "ollama/gemma3:4b", "auth_method": "local", "provider": "local"}
    assert _events(out, "settings")[-1]["model"] == "ollama/gemma3:4b"


def test_a_typed_model_id_is_saved_with_the_auth_that_reaches_it(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(config.AUTH_METHOD_ENV_VAR, config.AUTH_CLI_LOGIN)  # a Claude login from before
    monkeypatch.setenv(config.FRONTIER_PROVIDER_ENV_VAR, "anthropic")
    monkeypatch.setattr(cli_module.cli_transport, "available", lambda p: True)
    folder = _folder(tmp_path)
    server, _ = _server(folder)
    server.handle({"type": "set_model", "model": "gpt-5"})
    assert project_models.load(folder)["orchestrator"] == {"model": "gpt-5", "auth_method": "api_key", "provider": "openai"}
    assert server.cli_provider is None  # gpt-5 must not run through the Claude login
    project_models.deactivate()
    project_models.activate(folder)  # and reopening the project keeps it that way
    assert cli_module._cli_provider_for(os.environ[config.FRONTIER_MODEL_ENV_VAR], explicit=False) is None


def test_desktop_delegate_change_updates_the_file_and_the_event_says_project(tmp_path):
    folder = _folder(tmp_path)
    server, out = _server(folder)
    server.handle({"type": "advanced_model_request"})
    assert _events(out, "advanced_model")[-1]["source"] == "defaults"
    server.handle({"type": "set_delegate_target", "modality": "coding", "target": "qwen2.5-coder:7b"})
    assert _events(out, "advanced_model")[-1]["source"] == "project"
    assert project_models.load(folder)["delegates"]["coding"] == "ollama:qwen2.5-coder:7b"
