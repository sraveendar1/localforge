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

import json
import re

import httpx

from localforge import cli_transport, content_blocks
from localforge.answer_stream import AnswerStreamer
from localforge.backends.ollama import (
    CHARS_PER_TOKEN,
    GENERATE_LOCK,
    MAX_NUM_CTX_ENV_VAR,  # noqa: F401 - re-exported: the documented setting lives with the orchestrator
    MIN_NUM_CTX,
    OLLAMA_BASE_URL,
    OllamaBackend,
    error_from,
    max_context,
)

PREFIXES = ("ollama/", "ollama_chat/")
# Ollama's default context (2-4k tokens) would silently cut off the tool list
# and history; the orchestrator prompt needs far more.
# Ollama silently drops whatever doesn't fit the context window, and the
# model then reads a spliced-together prompt as if it were real (reported:
# read_file results "coming back corrupted", line 1535 spliced into line
# 1276, so the orchestrator re-read the same files and refused to write).
# So: size the window to the prompt, and if the prompt is bigger than the
# model can hold, trim it here -- visibly -- instead.
# (The window sizes are shared with delegated calls; see backends/ollama.py.)
RESERVED_TOKENS = 1024  # room for the reply
TIMEOUT = 600.0
# Below this, open-weight models tend to misread the task as orchestrators
# (seen live: gemma3:4b answered "hi" by creating a README).
MIN_RELIABLE_BILLIONS = 7


def parameter_billions(frontier_model: str) -> float | None:
    """Size from an Ollama tag like `gemma3:4b` or `qwen2.5:0.5b`; None if the
    tag doesn't say (e.g. `llama3:latest`)."""
    match = re.search(r"[:\-](\d+(?:\.\d+)?)b\b", model_name(frontier_model).lower())
    return float(match.group(1)) if match else None


# Vision-capable Ollama model families -- checked against Ollama's own
# published model cards (ollama.com/library/<name>) at the time this was
# written, but NOT re-verified live against the registry the way
# catalog.py's own entries are (see catalog.py's "Sizes were checked
# against Ollama's registry on 2026-09-23" note): this sandbox's network
# policy blocks ollama.com outright, so there was no way to confirm current
# tags/sizes here. These aren't in catalog_data.yaml at all -- they're
# orchestrator-only (picked via /model, never a delegate/worker choice for
# the coding/docs/general modalities), matched by base family name so any
# installed tag (llava:7b, llava:13b, llava:34b, ...) counts.
VISION_CAPABLE_FAMILIES = {"llava", "bakllava", "llava-llama3", "llava-phi3", "moondream", "llama3.2-vision"}


def supports_vision(frontier_model: str) -> bool:
    """Whether this local orchestrator model can actually see an attached
    image, so orchestrator.run() can refuse up front rather than silently
    dropping it or sending Ollama an `images` field a text-only model
    ignores."""
    family = model_name(frontier_model).split(":")[0].lower()
    return family in VISION_CAPABLE_FAMILIES


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


def _stream_chat(client: httpx.Client, body: dict, streamer: AnswerStreamer) -> tuple[str, dict]:
    """(full reply, final event) from a streamed /api/chat, feeding each
    chunk to the answer streamer as it arrives."""
    parts: list[str] = []
    final: dict = {}
    with client.stream("POST", "/api/chat", json=body) as resp:
        if (error := error_from(resp, body["model"])) is not None:
            raise LocalOrchestratorError(str(error))
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise LocalOrchestratorError(str(event["error"]))
            chunk = str((event.get("message") or {}).get("content") or "")
            if chunk:
                parts.append(chunk)
                streamer.feed(chunk)
            if event.get("done"):
                final = event  # carries the token counts
    return "".join(parts), final


def model_name(frontier_model: str) -> str:
    for prefix in PREFIXES:
        if frontier_model.startswith(prefix):
            return frontier_model[len(prefix) :]
    return frontier_model


_windows: dict[str, int | None] = {}


def context_window(name: str) -> int:
    """The window this orchestrator model gets: what it can run with on this
    machine while leaving room for everything else (catalog.context_limit),
    for a model localforge knows; max_context() otherwise."""
    from localforge.catalog import context_limit, load_catalog
    from localforge.hardware import detect_hardware

    if name not in _windows:  # hardware doesn't change mid-session; don't re-detect every turn
        try:
            entry = next((e for e in load_catalog() if e.name == name), None)
            _windows[name] = context_limit(entry, detect_hardware()) if entry is not None else None
        except Exception:  # noqa: BLE001 - sizing is best-effort; the default cap still protects the prompt
            _windows[name] = None
    limit = _windows[name]
    # Never more than the model was trained for: Ollama would silently cap
    # the window there and cut the prompt (a local orchestrator got 32k
    # whatever the model was; an 8k-trained model lost most of it).
    trained = OllamaBackend().trained_context(name)
    return min(x for x in (limit, trained, max_context()) if x)


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def fit_to_context(messages: list[dict], render, budget_tokens: int) -> tuple[str, int]:
    """(prompt, trimmed count) for a prompt that fits `budget_tokens`.

    Oldest tool results are shortened first (they're the bulk: file reads,
    command output), and the model is told that happened, so it can read
    something again instead of trusting a half-truncated copy.
    """
    prompt = render(messages)
    if _estimate_tokens(prompt) <= budget_tokens:
        return prompt, 0
    trimmed, working = 0, [dict(m) for m in messages]
    for message in working:
        if _estimate_tokens(render(working)) <= budget_tokens:
            break
        content = str(message.get("content") or "")
        if message.get("role") == "tool" and len(content) > 200:
            message["content"] = content[:200] + f"\n[... {len(content) - 200} characters trimmed to fit this model's context]"
            trimmed += 1
    while _estimate_tokens(render(working)) > budget_tokens and len(working) > 2:
        # still too big: drop the oldest turn (after the system message)
        del working[1]
        trimmed += 1
    prompt = render(working)
    prompt, clamped = _clamp(prompt, budget_tokens)
    trimmed += clamped
    if trimmed:
        prompt += (
            f"\n\n[localforge: {trimmed} earlier item(s) were trimmed so this prompt fits the model's context. "
            "Anything you need in full, fetch again (read_file with an offset, or a narrower search).]"
        )
    return prompt, trimmed


def _clamp(prompt: str, budget_tokens: int) -> tuple[str, int]:
    """Last resort: one message can be bigger than the whole window on its
    own. Cut its middle, keeping the start (the instructions and tools) and
    the end (the newest turn) -- and say so, rather than letting Ollama cut
    it invisibly."""
    limit = budget_tokens * CHARS_PER_TOKEN
    if len(prompt) <= limit:
        return prompt, 0
    marker = "\n[... middle cut to fit this model's context ...]\n"
    room = limit - len(marker)
    head = room // 3
    return prompt[:head] + marker + prompt[len(prompt) - (room - head) :], 1


def complete(frontier_model: str, messages: list[dict], tools: list[dict], on_text=None) -> cli_transport.CLIResponse:
    """One orchestration turn on a local model. With `on_text`, the reply is
    streamed and the final answer's text is passed on as it's written."""
    name = model_name(frontier_model)
    backend = OllamaBackend()
    if not backend.is_running():
        raise LocalOrchestratorError("Ollama is not running — start it (the Ollama app, or `ollama serve`) and try again.")
    try:
        backend.ensure_available(name)
    except Exception as exc:  # noqa: BLE001 - reported with a clear next step
        raise LocalOrchestratorError(f"Could not get {name} from Ollama: {exc}") from exc

    window = context_window(name)
    budget = window - RESERVED_TOKENS
    prompt, trimmed = fit_to_context(messages, lambda msgs: cli_transport._render_prompt(msgs, tools), budget)
    # Ask for exactly as much context as this prompt needs (capped): too
    # small silently corrupts it, too large wastes memory on the user's machine.
    num_ctx = min(window, max(MIN_NUM_CTX, _estimate_tokens(prompt) + RESERVED_TOKENS))
    user_message = {"role": "user", "content": prompt}
    image = content_blocks.extract_image(messages)
    if image:
        # Ollama's chat API takes bare base64 in its own `images` field,
        # separate from the flattened text prompt built above (orchestrator.
        # _check_image_support() already refused up front if `name` isn't
        # vision-capable, so reaching here means it is).
        user_message["images"] = [image["data"]]
    body = {
        "model": name,
        "messages": [user_message],
        "stream": on_text is not None,
        "format": "json",
        "options": {"num_ctx": num_ctx},
    }
    try:
        # Shared with backends.ollama.OllamaBackend.generate() (see
        # GENERATE_LOCK's comment there): a local orchestrator's own turn is
        # just as vulnerable to a concurrent delegate/side-question call
        # evicting its model mid-stream as a delegated generate() call is.
        with GENERATE_LOCK, httpx.Client(base_url=OLLAMA_BASE_URL, timeout=TIMEOUT) as client:
            if on_text is None:
                resp = client.post("/api/chat", json=body)
                if (error := error_from(resp, name)) is not None:
                    raise LocalOrchestratorError(str(error))
                data = resp.json()
                raw = str((data.get("message") or {}).get("content") or "")
            else:
                raw, data = _stream_chat(client, body, AnswerStreamer(on_text))
    except httpx.HTTPError as exc:
        raise LocalOrchestratorError(f"{name} failed: {exc}") from exc

    usage = cli_transport._Usage(
        prompt_tokens=int(data.get("prompt_eval_count") or 0),
        completion_tokens=int(data.get("eval_count") or 0),
    )
    return cli_transport.CLIResponse(
        choices=[cli_transport._Choice(message=cli_transport.message_from_reply(raw))],
        usage=usage,
        local=True,
    )
