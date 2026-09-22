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
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from localforge import config


class CLINotAvailableError(RuntimeError):
    """The provider's CLI isn't installed, or isn't logged in."""


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


def logged_in(provider: str, timeout: float = 60.0) -> bool:
    """Probe whether the CLI answers a trivial prompt, i.e. is authenticated.
    Costs a negligible amount of the account's own subscription usage.
    """
    if not available(provider):
        return False
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
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0 and "OK" in proc.stdout


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


def _render_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Flatten the conversation + tool schemas into one prompt, asking for a
    strict JSON reply we can parse back into tool calls with arguments.
    """
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

    return (
        (str(system[0].get("content")) + "\n\n" if system else "")
        + "Decide the next step. The tools below are the only ones you have.\n\n"
        "Available tools (arguments marked ? are optional):\n"
        + ("\n".join(tool_lines) if tool_lines else "(none)")
        + "\n\nConversation so far:\n"
        + "\n\n".join(transcript)
        + "\n\nReply with ONLY a single JSON object, no prose and no code fence.\n"
        'To use tools (one or more): {"tool_calls": [{"name": "<tool>", "arguments": {"<arg>": <value>, ...}}]}\n'
        'When finished:  {"final_answer": "<your reply to the user>"}'
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

    return _Usage(
        prompt_tokens=pick("input_tokens", "prompt_tokens", "input", "prompt"),
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
            _read_usage(envelope.get(spec.get("usage_key", "usage"))),
            float(envelope.get("total_cost_usd", 0.0) or 0.0),
        )
    return stdout, _Usage(), 0.0


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


def complete(provider: str, messages: list[dict], tools: list[dict], timeout: float = 600.0) -> CLIResponse:
    """Run one orchestration turn through the provider's logged-in CLI."""
    spec = _spec(provider)
    if not available(provider):
        raise CLINotAvailableError(requirements_message(provider))

    prompt = _render_prompt(messages, tools)
    cmd = [spec["command"], *spec["headless_args"], *spec.get("isolation_args", []), *spec.get("json_args", [])]
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
            proc = subprocess.run(
                cmd, input=stdin_text, capture_output=True, text=True, timeout=timeout, check=False, cwd=scratch
            )
    except subprocess.TimeoutExpired as exc:
        raise CLINotAvailableError(f"{spec['command']} timed out after {timeout}s") from exc
    except OSError as exc:
        raise CLINotAvailableError(f"Could not run {spec['command']}: {exc}") from exc

    if proc.returncode != 0:
        raise CLINotAvailableError(f"{spec['command']} exited {proc.returncode}: {_failure_detail(proc)}")

    raw, usage, notional_cost = _unwrap_envelope(proc.stdout.strip(), spec)

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

    return CLIResponse(choices=[_Choice(message=message)], usage=usage, notional_cost_usd=notional_cost)
