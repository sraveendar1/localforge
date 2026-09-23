"""Ollama backend: serves text/coding/docs models via Ollama's local REST API."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable

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
# A non-streamed call (memory, the project brief) wants a short reply.
MAX_QUIET_OUTPUT_TOKENS = 4096

# Context window. Ollama's default is only a few thousand tokens, and it
# silently drops whatever doesn't fit: a delegated task with a file attached
# lost its own instructions that way, and the model answered the tail of the
# file instead. So every call asks for a window sized to its prompt plus the
# reply, rounded up to a few fixed sizes -- Ollama reloads the model whenever
# num_ctx changes, and a reload per call would cost more than the room.
MIN_NUM_CTX = 8192
MAX_NUM_CTX_ENV_VAR = "LOCALFORGE_LOCAL_CONTEXT"
DEFAULT_MAX_NUM_CTX = 32768
CHARS_PER_TOKEN = 3  # code packs tighter than prose; erring small keeps us inside the window


def max_context() -> int:
    try:
        return max(MIN_NUM_CTX, int(os.environ.get(MAX_NUM_CTX_ENV_VAR, DEFAULT_MAX_NUM_CTX)))
    except ValueError:
        return DEFAULT_MAX_NUM_CTX


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def _cap(limit: int | None) -> int:
    return min(limit, max_context()) if limit else max_context()


def context_size(prompt_tokens: int, output_tokens: int, limit: int | None = None) -> int:
    """num_ctx for a call: the smallest of 8k, 16k, 32k, ... that holds the
    prompt and the reply, capped at `limit` (what this model can run with
    on this machine, catalog.context_limit) and max_context()."""
    cap = _cap(limit)
    needed, size = prompt_tokens + output_tokens, MIN_NUM_CTX
    while size < needed and size < cap:
        size *= 2
    return min(size, cap)


def reply_room(limit: int | None = None, output_tokens: int = MAX_OUTPUT_TOKENS) -> int:
    """Tokens kept for the reply in a window of `limit`: at most half of it,
    so a small window still has room for the prompt."""
    return min(output_tokens, _cap(limit) // 2)


def prompt_budget(limit: int | None = None, output_tokens: int = MAX_OUTPUT_TOKENS) -> int:
    """Characters of prompt that fit in a `limit` window beside the reply."""
    cap = _cap(limit)
    return max(0, cap - reply_room(limit, output_tokens)) * CHARS_PER_TOKEN


def _with_context(prompt: str, output_tokens: int, kwargs: dict) -> dict:
    limit = kwargs.pop("context_limit", None)
    options = {"num_predict": output_tokens, **kwargs.pop("options", {})}
    prompt_tokens = estimate_tokens(prompt)
    options.setdefault("num_ctx", context_size(prompt_tokens, options["num_predict"], limit))
    # The reply must fit in what's left of the window.
    options["num_predict"] = max(256, min(options["num_predict"], options["num_ctx"] - prompt_tokens))
    return options


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
        options = _with_context(prompt, MAX_QUIET_OUTPUT_TOKENS, kwargs)
        with self._client(GENERATE_TIMEOUT) as client:
            resp = client.post(
                "/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False, "options": options, **kwargs},
            )
            resp.raise_for_status()
            data = resp.json()
            # "eval_count" is Ollama's count of tokens it generated for this
            # response -- used for usage metrics, not for the API call itself.
            return {
                "type": "text",
                "content": data["response"],
                "tokens": data.get("eval_count", 0),
                "truncated": data.get("done_reason") == "length",
            }

    def _generate_streaming(self, model_name: str, prompt: str, on_token: Callable[[str], None], **kwargs) -> BackendResult:
        """Stream a generation, with guards so a stuck model can't hang the
        task: a cap on output length, detection of the model repeating
        itself, and an overall time limit. Each raises LocalModelStuck, which
        the dispatcher recovers from (another model, or a retry)."""
        parts: list[str] = []
        tokens, truncated = 0, False
        started = time.monotonic()
        options = _with_context(prompt, MAX_OUTPUT_TOKENS, kwargs)
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
                        # "length": it hit num_predict and stopped mid-output
                        truncated = event.get("done_reason") == "length"
        return {"type": "text", "content": "".join(parts), "tokens": tokens, "truncated": truncated}
