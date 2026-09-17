"""Ollama backend: serves text/coding/docs models via Ollama's local REST API."""

from __future__ import annotations

import httpx

from localforge.backends.base import BackendResult

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

    def ensure_available(self, model_name: str) -> None:
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
                for _ in resp.iter_lines():
                    pass  # drain the pull progress stream; CLI reports progress separately

    def generate(self, model_name: str, prompt: str, **kwargs) -> BackendResult:
        with self._client() as client:
            resp = client.post(
                "/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False, **kwargs},
            )
            resp.raise_for_status()
            return {"type": "text", "content": resp.json()["response"]}
