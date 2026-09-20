import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge.backends.ollama import OllamaBackend

INSTALLED = [
    {"name": "qwen2.5-coder:14b", "size": 9_000_000_000, "modified_at": "2026-09-20T01:00:00Z"},
    {"name": "mistral-nemo:12b", "size": 7_000_000_000, "modified_at": "2026-09-20T01:05:00Z"},
]


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
            if self.path != "/api/delete":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            name = body.get("name")
            before = len(installed)
            installed[:] = [m for m in installed if m["name"] != name]
            self.send_response(200 if len(installed) < before else 404)
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
def fake_ollama():
    installed = [dict(m) for m in INSTALLED]
    server = HTTPServer(("localhost", 0), _make_handler(installed))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://localhost:{server.server_port}"
    with patch.object(cli_module, "OllamaBackend", lambda *a, **k: OllamaBackend(base_url=base_url)):
        yield installed
    server.shutdown()
    thread.join()


def test_installed_lists_models_with_total(fake_ollama):
    result = CliRunner().invoke(cli_module.app, ["installed"])
    assert result.exit_code == 0
    assert "qwen2.5-coder:14b" in result.output
    assert "mistral-nemo:12b" in result.output
    assert "Total: 14.90 GB" in result.output


def test_delete_by_name_removes_only_that_model(fake_ollama):
    result = CliRunner().invoke(cli_module.app, ["delete", "qwen2.5-coder:14b", "--yes"])
    assert result.exit_code == 0
    assert "Deleted qwen2.5-coder:14b" in result.output
    assert {m["name"] for m in fake_ollama} == {"mistral-nemo:12b"}


def test_delete_skips_unknown_model_names(fake_ollama):
    result = CliRunner().invoke(cli_module.app, ["delete", "not-a-real-model", "--yes"])
    assert result.exit_code == 0
    assert "Skipping" in result.output
    assert "Nothing queued" in result.output
    assert len(fake_ollama) == 2  # nothing deleted


def test_delete_interactive_queue_requires_confirmation(fake_ollama):
    result = CliRunner().invoke(cli_module.app, ["delete"], input="1\nn\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    assert len(fake_ollama) == 2  # declining the confirmation deletes nothing
