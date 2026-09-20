"""Ollama backend: serves text/coding/docs models via Ollama's local REST API."""

from __future__ import annotations

import json
from typing import Callable

import httpx

from localforge.backends.base import BackendResult

# Called with each raw progress event Ollama streams back while pulling, e.g.
# {"status": "pulling manifest"} or
# {"status": "downloading", "digest": "...", "total": 123, "completed": 45}.
ProgressCallback = Callable[[dict], None]

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaNotRunningError(RuntimeError):
    """Raised when Ollama's local server cannot be reached."""


class OllamaBackend:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = 300.0):
        self.base_url = base_url
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def is_running(self) -> bool:
        try:
            with self._client() as client:
                client.get("/api/version").raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def _installed_models(self, client: httpx.Client) -> set[str]:
        resp = client.get("/api/tags")
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}

    def list_installed(self) -> list[dict]:
        """Models actually pulled and present on disk, per Ollama -- not the
        static catalog. Each entry has at least "name", "size" (bytes), and
        "modified_at".
        """
        with self._client() as client:
            resp = client.get("/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])

    def delete(self, model_name: str) -> None:
        """Remove a pulled model from disk, freeing its space."""
        with self._client() as client:
            resp = client.request("DELETE", "/api/delete", json={"name": model_name})
            resp.raise_for_status()

    def ensure_available(self, model_name: str, on_progress: ProgressCallback | None = None) -> None:
        if not self.is_running():
            raise OllamaNotRunningError(
                "Ollama is not running. Install it from https://ollama.com and start it, "
                "then retry."
            )
        with self._client() as client:
            if model_name in self._installed_models(client):
                return
            with client.stream("POST", "/api/pull", json={"name": model_name}) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or on_progress is None:
                        continue
                    try:
                        on_progress(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def generate(self, model_name: str, prompt: str, **kwargs) -> BackendResult:
        with self._client() as client:
            resp = client.post(
                "/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False, **kwargs},
            )
            resp.raise_for_status()
            data = resp.json()
            # "eval_count" is Ollama's count of tokens it generated for this
            # response -- used for usage metrics, not for the API call itself.
            return {"type": "text", "content": data["response"], "tokens": data.get("eval_count", 0)}
