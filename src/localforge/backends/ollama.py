"""Ollama backend: serves text/coding/docs models via Ollama's local REST API."""

from __future__ import annotations

import json
import time
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


class LocalModelStuck(RuntimeError):
    """A generation was stopped: it looped, or ran far too long."""


# Guards for a streamed generation (see _generate_streaming).
MAX_OUTPUT_TOKENS = 8192  # a whole source file fits; an endless ramble doesn't
MAX_GENERATION_SECONDS = 900
# Loading a big model on a laptop can take minutes before the first token,
# so reads get more patience than the quick /api/tags calls.
GENERATE_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)


def _is_repeating(text: str, window: int = 120, times: int = 4) -> bool:
    """True when the output's tail is one chunk of text repeated over and
    over -- small models sometimes fall into this and never stop."""
    if len(text) < window * times:
        return False
    tail = text[-window:]
    if len(set(tail)) < 12:
        return False  # blank space or a divider line like "=====", not a loop
    return text[-window * times * 2 :].count(tail) >= times


class OllamaBackend:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = 300.0):
        self.base_url = base_url
        self.timeout = timeout

    def _client(self, timeout: float | httpx.Timeout | None = None) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout if timeout is not None else self.timeout)

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

    def generate(
        self, model_name: str, prompt: str, on_token: Callable[[str], None] | None = None, **kwargs
    ) -> BackendResult:
        """Run the model. With `on_token`, the reply is streamed and each chunk
        is handed over as Ollama produces it, so the user watches the local
        model work instead of staring at a spinner until it's done.
        """
        if on_token is not None:
            return self._generate_streaming(model_name, prompt, on_token, **kwargs)
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

    def _generate_streaming(self, model_name: str, prompt: str, on_token: Callable[[str], None], **kwargs) -> BackendResult:
        """Stream a generation, with guards so a stuck model can't hang the
        task: a cap on output length, detection of the model repeating
        itself, and an overall time limit. Each raises LocalModelStuck, which
        the dispatcher recovers from (another model, or a retry)."""
        parts: list[str] = []
        tokens = 0
        started = time.monotonic()
        options = {"num_predict": MAX_OUTPUT_TOKENS, **kwargs.pop("options", {})}
        with self._client(GENERATE_TIMEOUT) as client:
            with client.stream(
                "POST", "/api/generate", json={"model": model_name, "prompt": prompt, "stream": True, "options": options, **kwargs}
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("error"):
                        raise RuntimeError(f"{model_name}: {event['error']}")
                    chunk = event.get("response", "")
                    if chunk:
                        parts.append(chunk)
                        on_token(chunk)
                        if len(parts) % 50 == 0:  # cheap enough to check every 50 chunks
                            if _is_repeating("".join(parts)):
                                raise LocalModelStuck(
                                    f"{model_name} got stuck repeating the same text after ~{len(parts)} tokens; stopped it"
                                )
                            if time.monotonic() - started > MAX_GENERATION_SECONDS:
                                raise LocalModelStuck(
                                    f"{model_name} was still writing after {MAX_GENERATION_SECONDS // 60} minutes; stopped it"
                                )
                    if event.get("done"):
                        # only the final event carries the token count
                        tokens = event.get("eval_count", 0)
        return {"type": "text", "content": "".join(parts), "tokens": tokens}
