"""Orchestrate with an open-weight model on this machine, via Ollama.

Chosen with a model id like `ollama/qwen2.5:7b` (setup's "local" provider,
`/model`, or `run --model`). Talks to Ollama directly and never goes near a
provider CLI or API key -- a stale Claude-login setting once made a "local"
orchestrator still call `claude` and fail on the Claude account's limits.

Uses the same JSON decision protocol as the CLI transport (the tool list
and conversation rendered into one prompt, a strict JSON reply parsed back)
rather than native tool calling, because many open-weight models (Gemma,
among others) have no tool-calling template in Ollama at all. Ollama's
`format: "json"` keeps the reply parseable even for small models.
"""

from __future__ import annotations

import re

import httpx

from localforge import cli_transport
from localforge.backends.ollama import OLLAMA_BASE_URL, OllamaBackend

PREFIXES = ("ollama/", "ollama_chat/")
# Ollama's default context (2-4k tokens) would silently cut off the tool list
# and history; the orchestrator prompt needs far more.
NUM_CTX = 16384
TIMEOUT = 600.0
# Below this, open-weight models tend to misread the task as orchestrators
# (seen live: gemma3:4b answered "hi" by creating a README).
MIN_RELIABLE_BILLIONS = 7


def parameter_billions(frontier_model: str) -> float | None:
    """Size from an Ollama tag like `gemma3:4b` or `qwen2.5:0.5b`; None if the
    tag doesn't say (e.g. `llama3:latest`)."""
    match = re.search(r"[:\-](\d+(?:\.\d+)?)b\b", model_name(frontier_model).lower())
    return float(match.group(1)) if match else None


def orchestrator_choices(hardware, installed: set[str]) -> list[str]:
    """Local orchestrator options for THIS machine, as `ollama/<tag>` ids:
    what's already in Ollama first (no download), then text models from the
    catalog that fit the hardware, best first. The old fixed list offered
    only 70B-class models, ~40 GB downloads no laptop can run.
    """
    from localforge.catalog import candidates, load_catalog

    catalog = load_catalog()
    ids = [f"ollama/{name}" for name in sorted(installed)]
    fitting = []
    for modality in ("general", "coding", "docs"):
        fitting += [m for m in candidates(modality, hardware, catalog, installed) if m.runtime == "ollama"]
    for entry in sorted(fitting, key=lambda m: (-m.quality_tier, m.disk_gb)):
        model_id = f"ollama/{entry.name}"
        if model_id not in ids:
            ids.append(model_id)
    return ids


class LocalOrchestratorError(RuntimeError):
    """The local orchestrator model couldn't be reached or run."""


def model_name(frontier_model: str) -> str:
    for prefix in PREFIXES:
        if frontier_model.startswith(prefix):
            return frontier_model[len(prefix) :]
    return frontier_model


def complete(frontier_model: str, messages: list[dict], tools: list[dict]) -> cli_transport.CLIResponse:
    name = model_name(frontier_model)
    backend = OllamaBackend()
    if not backend.is_running():
        raise LocalOrchestratorError("Ollama is not running — start it (the Ollama app, or `ollama serve`) and try again.")
    try:
        backend.ensure_available(name)
    except Exception as exc:  # noqa: BLE001 - reported with a clear next step
        raise LocalOrchestratorError(f"Could not get {name} from Ollama: {exc}") from exc

    prompt = cli_transport._render_prompt(messages, tools)
    try:
        with httpx.Client(base_url=OLLAMA_BASE_URL, timeout=TIMEOUT) as client:
            resp = client.post(
                "/api/chat",
                json={
                    "model": name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"num_ctx": NUM_CTX},
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise LocalOrchestratorError(f"{name} failed: {exc}") from exc

    raw = str((data.get("message") or {}).get("content") or "")
    usage = cli_transport._Usage(
        prompt_tokens=int(data.get("prompt_eval_count") or 0),
        completion_tokens=int(data.get("eval_count") or 0),
    )
    return cli_transport.CLIResponse(
        choices=[cli_transport._Choice(message=cli_transport.message_from_reply(raw))],
        usage=usage,
        local=True,
    )
