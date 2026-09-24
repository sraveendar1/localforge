"""Orchestrate through a provider's own logged-in CLI instead of an API key.

`orchestrator.run()` needs one thing from a frontier model: given the
conversation plus tool schemas, either ask for tool calls or give a final
answer. The Messages API provides that natively via `tool_calls`. A headless
CLI (`claude -p`, `codex exec`, `gemini -p`) only prints text, so this module
asks for a strict JSON reply and parses it back into the *same shape* the
orchestrator already consumes -- which keeps the loop, its memory management,
retry handling and usage metrics identical across both transports.

Deliberately generic: it knows only "a command that takes a prompt and prints
text", configured per provider in `config.FRONTIER_CLI_AUTH`. That avoids
tying the orchestrator to any one vendor's SDK.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from localforge import config
from localforge.answer_stream import AnswerStreamer


class CLINotAvailableError(RuntimeError):
    """The provider's CLI isn't installed, or isn't logged in."""


class UsageLimitError(CLINotAvailableError):
    """The account hit a usage/spend limit. Carries when it resets, if the
    provider said -- a task can then wait and carry on instead of dying."""

    def __init__(self, message: str, provider: str = "", reset_at: datetime | None = None):
        super().__init__(message)
        self.provider = provider
        self.reset_at = reset_at


LIMIT_MARKERS = (
    "usage limit",
    "spend limit",
    "rate limit",
    "rate_limit",
    "quota",
    "credit balance",
    "out of credits",
    "limit reached",
    "exceeded your",
    "too many requests",
)
_RESET = re.compile(r"reset[s]?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*(?:\(([^)]+)\))?", re.I)


def looks_like_limit(text: str) -> bool:
    return any(marker in (text or "").lower() for marker in LIMIT_MARKERS)


def parse_reset_time(text: str, now: datetime | None = None) -> datetime | None:
    """When the provider says the limit resets, e.g. "your session limit
    resets 11:10pm (Europe/Lisbon)". Returns a local-time datetime in the
    future, or None when the message doesn't say (a monthly spend limit
    usually doesn't -- and waiting for one would be pointless)."""
    match = _RESET.search(text or "")
    if not match:
        return None
    hour, minute, meridiem, zone = match.group(1), match.group(2), match.group(3), match.group(4)
    hour, minute = int(hour), int(minute or 0)
    if meridiem:
        meridiem = meridiem.replace(".", "").lower()
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    tz = None
    if zone:
        try:
            tz = ZoneInfo(zone.strip())
        except Exception:  # noqa: BLE001 - an unknown zone name just means "assume local"
            tz = None
    now = now or datetime.now(tz or timezone.utc).astimezone()
    reference = now.astimezone(tz) if tz else now
    reset = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= reference:
        reset += timedelta(days=1)  # the time given is tomorrow
    return reset.astimezone(now.tzinfo)


# --- response objects shaped like the LiteLLM ones the orchestrator expects ---


@dataclass
class _Function:
    name: str
    arguments: str  # JSON string, matching the API's own encoding


@dataclass
class _ToolCall:
    id: str
    function: _Function


@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None

    def model_dump(self) -> dict:
        # Rendered back into the transcript; tool_calls are replayed as text
        # because a CLI has no native assistant-tool_call message type.
        if self.tool_calls:
            calls = ", ".join(f"{c.function.name}({c.function.arguments})" for c in self.tool_calls)
            return {"role": "assistant", "content": f"[requested: {calls}]"}
        return {"role": "assistant", "content": self.content or ""}


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _Choice:
    message: _Message


@dataclass
class CLIResponse:
    choices: list[_Choice]
    usage: _Usage = field(default_factory=_Usage)
    # What this call *would* have cost on pay-per-token billing. Under CLI
    # login it's drawn from the account's subscription instead, so callers
    # must not present it as money separately billed.
    notional_cost_usd: float = 0.0
    # True when an open-weight model on this machine produced it: no
    # subscription, no billing, nothing notional to report.
    local: bool = False


def _spec(provider: str) -> dict:
    spec = config.FRONTIER_CLI_AUTH.get(provider)
    if spec is None:
        raise CLINotAvailableError(f"No CLI login is defined for provider {provider!r}.")
    return spec


def available(provider: str) -> bool:
    """Is the provider's CLI installed on PATH?"""
    try:
        return shutil.which(_spec(provider)["command"]) is not None
    except CLINotAvailableError:
        return False


def probe(provider: str, timeout: float = 60.0) -> tuple[bool, str]:
    """(works, why not). Asks the CLI a trivial prompt; a failure's own
    message is kept, because "not logged in" and "hit your usage limit" need
    different fixes. Costs a negligible amount of the account's usage.
    """
    if not available(provider):
        return False, "not installed"
    spec = _spec(provider)
    try:
        proc = subprocess.run(
            [spec["command"], *spec["headless_args"], *spec.get("isolation_args", []), "Reply with exactly: OK"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, f"no answer within {timeout:.0f}s"
    except (subprocess.SubprocessError, OSError) as exc:
        return False, str(exc)
    if proc.returncode == 0 and "OK" in proc.stdout:
        return True, ""
    return False, _failure_detail(proc)


def logged_in(provider: str, timeout: float = 60.0) -> bool:
    """Whether the CLI answers a trivial prompt, i.e. is usable right now."""
    return probe(provider, timeout)[0]


def requirements_message(provider: str) -> str:
    """Actionable text for when CLI login isn't usable yet."""
    spec = _spec(provider)
    return (
        f"The {spec['command']!r} CLI is required for {provider} CLI login. "
        f"Install it ({spec['install_hint']}), then {spec['login_hint']}."
    )


def _signature(tool: dict) -> str:
    """`name(path, offset?, limit?)` from a JSON-schema tool definition."""
    params = tool["function"].get("parameters") or {}
    required = set(params.get("required") or [])
    names = [n if n in required else f"{n}?" for n in (params.get("properties") or {})]
    return f"{tool['function']['name']}({', '.join(names)})"


_PROTOCOL = (
    "If the user's latest message doesn't ask you to build, change or look into something "
    "(a greeting, a question, a chat), answer it directly with final_answer and use no tools.\n"
    "Reply with ONLY a single JSON object, no prose and no code fence.\n"
    'To use tools (one or more): {"tool_calls": [{"name": "<tool>", "arguments": {"<arg>": <value>, ...}}]}\n'
    "Independent steps (reading several files, several searches) go in ONE reply as several tool_calls: "
    "every reply re-sends the whole conversation, so fewer replies cost less.\n"
    'When finished:  {"final_answer": "<your reply to the user>"}'
)


def _render_parts(messages: list[dict], tools: list[dict]) -> tuple[str, str]:
    """(instructions, conversation): the system message, tool list and reply
    protocol -- the same on every step of a task -- and the transcript, which
    only grows at the end. Kept apart so a CLI that takes its own system
    prompt can get the first as that (see complete())."""
    tool_lines = []
    for t in tools:
        tool_lines.append(f"- {_signature(t)}: {t['function']['description']}")
        for name, prop in ((t["function"].get("parameters") or {}).get("properties") or {}).items():
            if prop.get("description"):
                tool_lines.append(f"    {name}: {prop['description']}")

    system = [m for m in messages if m.get("role") == "system"]
    transcript = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        content = m.get("content") or ""
        transcript.append(f"[{'tool result' if role == 'tool' else role}]\n{content}")

    instructions = (
        (str(system[0].get("content")) + "\n\n" if system else "")
        + "Decide the next step. The tools below are the only ones you have.\n\n"
        "Available tools (arguments marked ? are optional):\n"
        + ("\n".join(tool_lines) if tool_lines else "(none)")
    )
    return instructions, "Conversation so far:\n" + "\n\n".join(transcript)


def _render_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Flatten the conversation + tool schemas into one prompt, asking for a
    strict JSON reply we can parse back into tool calls with arguments.
    """
    instructions, conversation = _render_parts(messages, tools)
    return instructions + "\n\n" + conversation + "\n\n" + _PROTOCOL


def _render_split(messages: list[dict], tools: list[dict]) -> tuple[str, str]:
    """(system prompt, prompt) for a CLI that takes a system prompt of its
    own. Replacing claude's default one matters: measured live, `claude -p`
    sent ~8,100 tokens of Claude Code's own instructions with every step --
    irrelevant here, with its tools off -- against ~1,960 with ours (a
    one-word reply cost 7x as much). The fixed part also stays the same all
    task long, so the CLI can cache it."""
    instructions, conversation = _render_parts(messages, tools)
    return (
        instructions + "\n\n" + _PROTOCOL,
        conversation + "\n\nDecide the next step. Reply with ONLY a single JSON object, as your instructions describe.",
    )


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of a CLI reply, tolerating code fences or
    surrounding prose (models add them even when told not to).
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _read_usage(blob: dict | None) -> _Usage:
    """Token counts, tolerant of each CLI's naming.

    Claude reports input_tokens/output_tokens; Gemini nests its counts under
    `stats`; others use prompt/completion naming. Unknown shapes yield zeros
    rather than wrong numbers -- never invent usage we didn't actually read.
    """
    if not isinstance(blob, dict):
        return _Usage()
    # Gemini nests the real counters a level down (stats -> models/tokens).
    for nested in ("tokens", "usage", "models"):
        if isinstance(blob.get(nested), dict):
            inner = blob[nested]
            if any(k in inner for k in ("input_tokens", "prompt_tokens", "input", "prompt")):
                blob = inner
                break

    def pick(*names: str) -> int:
        for n in names:
            v = blob.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    # Claude reports cached input apart from input_tokens (a first step
    # showed input 9 and cache writes 8,099); it's all input the model read.
    cached = pick("cache_creation_input_tokens") + pick("cache_read_input_tokens")
    return _Usage(
        prompt_tokens=pick("input_tokens", "prompt_tokens", "input", "prompt") + cached,
        completion_tokens=pick("output_tokens", "completion_tokens", "output", "candidates"),
    )


def _unwrap_envelope(stdout: str, spec: dict) -> tuple[str, _Usage, float]:
    """Pull (reply text, usage, notional cost) out of a CLI's machine output.

    Two shapes exist in the wild: a single JSON object (claude, gemini) and a
    JSON Lines event stream (codex), where the reply is the last
    `agent_message` item rather than a top-level key.
    """
    if spec.get("envelope") == "jsonl":
        text, usage, cost = "", _Usage(), 0.0
        for line in stdout.splitlines():
            event = _extract_json(line)
            if not isinstance(event, dict):
                continue
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = str(item.get(spec["result_key"], "") or text)
            if isinstance(event.get(spec.get("usage_key", "usage")), dict):
                usage = _read_usage(event[spec.get("usage_key", "usage")])
            if isinstance(event.get("total_cost_usd"), (int, float)):
                cost = float(event["total_cost_usd"])
        # No agent_message found -> fall back to the raw text.
        return (text or stdout), usage, cost

    envelope = _extract_json(stdout)
    if isinstance(envelope, dict) and spec["result_key"] in envelope:
        return (
            str(envelope.get(spec["result_key"]) or ""),
            _model_usage(envelope.get("modelUsage")) or _read_usage(envelope.get(spec.get("usage_key", "usage"))),
            float(envelope.get("total_cost_usd", 0.0) or 0.0),
        )
    return stdout, _Usage(), 0.0


def _model_usage(per_model) -> _Usage | None:
    """Every model a step used, added up. claude's top-level `usage` covers
    only the main model: a step whose built-in advisor tool ran Opus showed
    Haiku's tokens while Opus was 75% of the cost (seen live)."""
    if not isinstance(per_model, dict) or not per_model:
        return None
    prompt = completion = 0
    for counts in per_model.values():
        if not isinstance(counts, dict):
            continue
        prompt += sum(int(counts.get(k) or 0) for k in ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens"))
        completion += int(counts.get("outputTokens") or 0)
    return _Usage(prompt_tokens=prompt, completion_tokens=completion)


def message_from_reply(raw: str) -> _Message:
    """Turn a model's JSON decision ({"tool_calls": [...]} or
    {"final_answer": ...}) into the LiteLLM-shaped message the orchestrator
    consumes. Shared with the local (Ollama) orchestrator transport.
    """
    decision = _extract_json(raw)
    if decision is None:
        # Couldn't parse a decision -- treat the text as the final answer
        # rather than failing the whole run.
        message = _Message(content=raw, tool_calls=None)
    elif decision.get("tool_calls"):
        calls = []
        for i, call in enumerate(decision["tool_calls"]):
            name = call.get("name")
            if not name:
                continue
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                # some replies double-encode the arguments object
                arguments = _extract_json(arguments) or {"instructions": arguments}
            if not isinstance(arguments, dict):
                arguments = {}
            if "instructions" in call and "instructions" not in arguments:
                arguments["instructions"] = call["instructions"]  # older flat form
            calls.append(_ToolCall(id=f"cli_call_{i}", function=_Function(name=name, arguments=json.dumps(arguments))))
        message = _Message(content=None, tool_calls=calls or None)
        if not calls:
            message = _Message(content=raw, tool_calls=None)
    else:
        message = _Message(content=decision.get("final_answer") or raw, tool_calls=None)

    return message


def _failure_detail(proc) -> str:
    """The useful part of a failed run: a JSON envelope's own error/result
    text if there is one, else stderr/stdout -- not the first 300 chars of a
    usage blob, which is what hid the cause of an intermittent exit 1.
    """
    envelope = _extract_json(proc.stdout or "")
    if isinstance(envelope, dict):
        for key in ("error", "result", "message", "subtype"):
            if envelope.get(key):
                return str(envelope[key])[:500]
    return (proc.stderr or proc.stdout or "no output").strip()[:500]


def _run_streaming(cmd: list[str], stdin_text: str, cwd: str, timeout: float, on_text) -> SimpleNamespace:
    """Run a CLI that emits stream-json events (one JSON object per line),
    passing each text delta to an AnswerStreamer as it arrives. Returns a
    proc-like object whose stdout is the final `result` event, so the normal
    envelope parsing applies unchanged.
    """
    streamer = AnswerStreamer(on_text)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, bufsize=1
    )
    timed_out = threading.Event()

    def _watchdog() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, _watchdog)
    timer.start()
    # Write the prompt from a thread: a large prompt could fill the pipe
    # while the CLI is already writing events back.
    writer = threading.Thread(target=lambda: (proc.stdin.write(stdin_text), proc.stdin.close()), daemon=True)
    writer.start()
    result_event: dict | None = None
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "stream_event":
                delta = (event.get("event") or {}).get("delta") or {}
                if delta.get("type") == "text_delta":
                    streamer.feed(delta.get("text") or "")
            elif event.get("type") == "result":
                result_event = event
        stderr = proc.stderr.read()
        proc.wait()
    except BaseException:
        proc.kill()  # e.g. /stop mid-answer: don't leave `claude` running
        raise
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise subprocess.TimeoutExpired(cmd, timeout)
    stdout = json.dumps(result_event) if result_event is not None else "".join(lines)
    return SimpleNamespace(returncode=proc.returncode, stdout=stdout, stderr=stderr)


def complete(
    provider: str,
    messages: list[dict],
    tools: list[dict],
    timeout: float = 600.0,
    model: str | None = None,
    on_text=None,
) -> CLIResponse:
    """Run one orchestration turn through the provider's logged-in CLI, on
    `model` if given (otherwise the CLI's own default model). With `on_text`
    and a CLI that can stream, the final answer's text is passed on as it's
    written."""
    spec = _spec(provider)
    if not available(provider):
        raise CLINotAvailableError(requirements_message(provider))

    streaming = on_text is not None and bool(spec.get("stream_args"))
    output_args = spec["stream_args"] if streaming else spec.get("json_args", [])
    cmd = [spec["command"], *spec["headless_args"], *spec.get("isolation_args", []), *output_args]
    if spec.get("system_prompt_flag"):
        system_prompt, prompt = _render_split(messages, tools)
        cmd += [spec["system_prompt_flag"], system_prompt]
    else:
        prompt = _render_prompt(messages, tools)
    if model and spec.get("model_flag"):
        cmd += [spec["model_flag"], model.removeprefix(f"{provider}/")]  # gemini/x is LiteLLM's name; the CLI wants x
    if spec.get("effort_flag"):
        effort = os.environ.get(config.ORCHESTRATOR_EFFORT_ENV_VAR) or config.DEFAULT_ORCHESTRATOR_EFFORT
        cmd += [spec["effort_flag"], effort]
    if spec.get("prompt_via_stdin"):
        # A session's prompt grows every turn; stdin carries any size (a
        # 340k-char prompt verified live), where one argv element is fragile.
        # Closing stdin after writing is also what avoids `claude -p` hanging
        # on an open, never-written stdin.
        stdin_text = prompt
    else:
        cmd.append(prompt)
        stdin_text = ""
    try:
        # An empty scratch cwd: even a CLI whose tools can't be switched off
        # sees nothing of the folder localforge was started from.
        with tempfile.TemporaryDirectory(prefix="localforge-orchestrator-") as scratch:
            if streaming:
                proc = _run_streaming(cmd, stdin_text, scratch, timeout, on_text)
            else:
                proc = subprocess.run(
                    cmd, input=stdin_text, capture_output=True, text=True, timeout=timeout, check=False, cwd=scratch
                )
    except subprocess.TimeoutExpired as exc:
        raise CLINotAvailableError(f"{spec['command']} timed out after {timeout}s") from exc
    except OSError as exc:
        raise CLINotAvailableError(f"Could not run {spec['command']}: {exc}") from exc

    if proc.returncode != 0:
        detail = _failure_detail(proc)
        if looks_like_limit(detail):
            raise UsageLimitError(f"{spec['command']}: {detail}", provider=provider, reset_at=parse_reset_time(detail))
        raise CLINotAvailableError(f"{spec['command']} exited {proc.returncode}: {detail}")

    raw, usage, notional_cost = _unwrap_envelope(proc.stdout.strip(), spec)

    message = message_from_reply(raw)
    return CLIResponse(choices=[_Choice(message=message)], usage=usage, notional_cost_usd=notional_cost)
