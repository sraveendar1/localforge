import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "real_local_menu: use the real local-orchestrator menu builder")


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

    monkeypatch.setattr(cli_module, "_session", cli_module._SessionState())
    monkeypatch.setattr(cli_module, "_session_usage", [])
