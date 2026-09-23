import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge.backends.ollama import OllamaBackend

INSTALLED = [{"name": "qwen2.5-coder:14b", "size": 9_000_000_000, "modified_at": "2026-09-20T01:00:00Z"}]


def _make_handler(installed: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/api/version":
                self._json({"version": "0.0.0-fake"})
            elif self.path == "/api/tags":
                self._json({"models": installed})
            else:
                self.send_response(404)
                self.end_headers()

        def do_DELETE(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            installed[:] = [m for m in installed if m["name"] != body.get("name")]
            self.send_response(200)
            self.end_headers()

        def _json(self, data):
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


@pytest.fixture
def uninstall_env(tmp_path, monkeypatch):
    """Isolates every destructive side effect of `uninstall`: a fake Ollama
    server (never the real one), a temp config dir, a fake ~/.ollama under
    tmp_path (never the real home directory), and subprocess.run replaced
    with a recording mock so brew/uv are never actually invoked.
    """
    installed = [dict(m) for m in INSTALLED]
    server = HTTPServer(("localhost", 0), _make_handler(installed))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://localhost:{server.server_port}"

    fake_home = tmp_path / "home"
    fake_ollama_dir = fake_home / ".ollama"
    fake_ollama_dir.mkdir(parents=True)
    (fake_ollama_dir / "some-blob").write_text("data")

    monkeypatch.setenv("LOCALFORGE_CONFIG_DIR", str(tmp_path / "config"))
    cli_module.config.CONFIG_DIR = tmp_path / "config"
    cli_module.config.CONFIG_FILE = cli_module.config.CONFIG_DIR / "config.env"
    cli_module.config.save({"ANTHROPIC_API_KEY": "sk-fake"})

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0

    with (
        patch.object(cli_module, "OllamaBackend", lambda *a, **k: OllamaBackend(base_url=base_url)),
        patch.object(cli_module, "Path") as mock_path_cls,
        patch.object(cli_module.subprocess, "run", mock_run),
    ):
        mock_path_cls.home.return_value = fake_home
        yield {"installed": installed, "fake_ollama_dir": fake_ollama_dir, "mock_run": mock_run}

    server.shutdown()
    thread.join()


def test_uninstall_cancelled_by_default_removes_nothing(uninstall_env):
    result = CliRunner().invoke(cli_module.app, ["uninstall"], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    assert uninstall_env["fake_ollama_dir"].exists()
    assert cli_module.config.CONFIG_FILE.exists()
    assert len(uninstall_env["installed"]) == 1


def test_uninstall_without_purge_ollama_keeps_ollama_data(uninstall_env):
    # confirm main prompt yes, decline the separate Ollama-purge prompt
    result = CliRunner().invoke(cli_module.app, ["uninstall"], input="y\nn\n")
    assert result.exit_code == 0
    assert "Deleted model qwen2.5-coder:14b" in result.output
    assert len(uninstall_env["installed"]) == 0  # models are always deleted
    assert uninstall_env["fake_ollama_dir"].exists()  # but ~/.ollama is left alone
    assert not cli_module.config.CONFIG_FILE.exists()  # config is always removed
    # last subprocess.run call should be the self-uninstall, not a brew uninstall
    assert uninstall_env["mock_run"].call_args_list[-1].args[0] == ["uv", "tool", "uninstall", "localforge"]


def test_uninstall_with_purge_ollama_flag_removes_ollama_data(uninstall_env):
    result = CliRunner().invoke(cli_module.app, ["uninstall", "--purge-ollama", "--yes"])
    assert result.exit_code == 0
    assert not uninstall_env["fake_ollama_dir"].exists()
    assert "Removed Ollama and its data" in result.output


def test_uninstall_yes_flag_skips_only_main_confirmation_not_ollama_purge(uninstall_env):
    # --yes alone should still leave Ollama data alone (purge_ollama defaults False
    # and --yes short-circuits the *prompt* for it, not the value itself)
    result = CliRunner().invoke(cli_module.app, ["uninstall", "--yes"])
    assert result.exit_code == 0
    assert uninstall_env["fake_ollama_dir"].exists()
