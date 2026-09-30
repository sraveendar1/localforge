import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "real_local_menu: use the real local-orchestrator menu builder")
    config.addinivalue_line("markers", "untrusted: start the test with no trusted folders")
    config.addinivalue_line("markers", "real_model_fit: use the real hardware/disk lookup for the fit check")


@pytest.fixture(autouse=True)
def _restore_environment():
    """config.save() applies saved values to os.environ (so a session sees
    /setup and /model changes immediately). Undo that after each test so a
    setup run in one test can't change what the next one sees.
    """
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _fixed_local_orchestrator_menu(request, monkeypatch):
    """Setup's local-orchestrator menu is built from the real machine
    (installed Ollama models + hardware fit). Tests pick entries by number,
    so give them a fixed list instead of whatever this machine has -- except
    tests marked `real_local_menu`, which test that function itself.
    """
    from localforge import local_transport

    if request.node.get_closest_marker("real_local_menu"):
        return

    monkeypatch.setattr(local_transport, "orchestrator_choices", lambda hardware, installed: ["ollama/llama3.1:70b", "ollama/qwen2.5:72b"])


@pytest.fixture(autouse=True)
def _roomy_machine_for_model_fit(request, monkeypatch):
    """Choosing a local model is refused if it can't run on this machine
    (model_fit.py). Tests pick catalog models without caring which, so they
    get a big machine with nothing on disk to look up, instead of whatever the
    developer's laptop or CI happens to be -- except tests marked
    `real_model_fit`, which test the check against hardware they build."""
    import localforge.model_fit as model_fit
    from localforge.hardware import HardwareProfile

    if request.node.get_closest_marker("real_model_fit"):
        return
    big = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=32, ram_gb=256, free_disk_gb=4000, gpus=[])
    monkeypatch.setattr(model_fit, "detect_hardware", lambda: big)
    monkeypatch.setattr(model_fit, "installed_sizes", lambda: None)


@pytest.fixture(autouse=True)
def _no_active_project():
    """The project whose models file applies is module state (project_models.py);
    don't let one test's project leak into the next."""
    import localforge.project_models as project_models

    project_models.deactivate()
    yield
    project_models.deactivate()


@pytest.fixture(autouse=True)
def _isolated_config_dir(monkeypatch, tmp_path_factory):
    """Every test gets an empty config folder of its own. Without this, a test
    that doesn't isolate reads the developer's real ~/.config/localforge, and
    tests that reassign config.CONFIG_FILE leaked it into later tests.
    """
    from localforge import config

    cfg = tmp_path_factory.mktemp("localforge-config")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg)
    monkeypatch.setattr(config, "CONFIG_FILE", cfg / "config.env")
    monkeypatch.setenv("LOCALFORGE_CONFIG_DIR", str(cfg))


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch):
    """The interactive session's state (conversation, approvals, usage) is
    module-level in cli.py because the REPL runs commands in-process. Give
    every test a fresh one so nothing carries over."""
    import localforge.cli as cli_module

    session = cli_module._SessionState()
    monkeypatch.setattr(cli_module, "_session", session)
    monkeypatch.setattr(cli_module, "_session_usage", [])
    yield
    session.drop_scratchpad()  # never leave scratch folders in the real temp dir


@pytest.fixture(autouse=True)
def _own_cwd(monkeypatch, tmp_path_factory):
    """Run every test from a folder of its own. A command run with the repo as
    the working directory writes that project's `.localforge/` (usage history,
    the saved models) into the real checkout, and one test's saved models then
    applied to the next."""
    monkeypatch.chdir(tmp_path_factory.mktemp("project"))


@pytest.fixture(autouse=True)
def _trusted_cwd(request, _isolated_config_dir, _own_cwd):
    """Most tests call `run` and aren't about folder trust, so trust the
    current folder in this test's own (throwaway) config. Tests marked
    `untrusted` start with nothing trusted."""
    if request.node.get_closest_marker("untrusted"):
        return
    from pathlib import Path

    from localforge import trust

    trust.trust(Path.cwd())


@pytest.fixture(autouse=True)
def _scratch_in_tmp(monkeypatch, tmp_path_factory):
    """Scratchpads made during tests live under pytest's temp dir, not the
    real system temp folder."""
    from localforge import scratchpad

    base = tmp_path_factory.mktemp("scratch-base")
    monkeypatch.setattr(scratchpad, "base_dir", lambda: base)


@pytest.fixture(autouse=True)
def _no_real_gemini_lookup(monkeypatch):
    """With a real GEMINI_API_KEY in the developer's shell, /model and setup
    would ask Google for its model list mid-test. Tests that exercise that
    lookup patch httpx themselves."""
    from localforge import cli

    cli._gemini_cache.clear()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _fresh_local_windows():
    """local_transport caches each orchestrator model's window per process;
    a test that changes LOCALFORGE_LOCAL_CONTEXT must not see another's."""
    from localforge import local_transport

    local_transport._windows.clear()


@pytest.fixture(autouse=True)
def _fresh_trained_windows():
    """OllamaBackend caches each model's trained context length per process."""
    from localforge.backends import ollama

    ollama._trained_context.clear()
