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
            [spec["command"], *spec["headless_args"], "Reply with exactly: OK"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
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


def _render_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Flatten the conversation + tool schemas into one prompt, asking for a
    strict JSON reply we can parse back into tool calls.
    """
    tool_lines = [
        f"- {t['function']['name']}: {t['function']['description']}" for t in tools
    ]
    transcript = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        transcript.append(f"[{role}]\n{content}")

    return (
        "You are orchestrating a task. Decide the next step.\n\n"
        "Available tools (each takes a single string field `instructions`):\n"
        + ("\n".join(tool_lines) if tool_lines else "(none)")
        + "\n\nConversation so far:\n"
        + "\n\n".join(transcript)
        + "\n\nReply with ONLY a single JSON object, no prose and no code fence.\n"
        'To delegate: {"tool_calls": [{"name": "<tool>", "instructions": "<what to do>"}]}\n'
        'When finished:  {"final_answer": "<your complete answer>"}'
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


def complete(provider: str, messages: list[dict], tools: list[dict], timeout: float = 600.0) -> CLIResponse:
    """Run one orchestration turn through the provider's logged-in CLI."""
    spec = _spec(provider)
    if not available(provider):
        raise CLINotAvailableError(requirements_message(provider))

    cmd = [spec["command"], *spec["headless_args"], *spec.get("json_args", []), _render_prompt(messages, tools)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise CLINotAvailableError(f"{spec['command']} timed out after {timeout}s") from exc
    except OSError as exc:
        raise CLINotAvailableError(f"Could not run {spec['command']}: {exc}") from exc

    if proc.returncode != 0:
        raise CLINotAvailableError(
            f"{spec['command']} exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:300]}"
        )

    usage = _Usage()
    notional_cost = 0.0
    raw = proc.stdout.strip()

    envelope = _extract_json(raw)
    if envelope and spec["result_key"] in envelope:
        # Structured envelope: pull real usage out, then parse the reply text.
        u = envelope.get("usage") or {}
        usage = _Usage(
            prompt_tokens=int(u.get("input_tokens", 0) or 0),
            completion_tokens=int(u.get("output_tokens", 0) or 0),
        )
        notional_cost = float(envelope.get("total_cost_usd", 0.0) or 0.0)
        raw = str(envelope.get(spec["result_key"]) or "")

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
            instructions = call.get("instructions") or call.get("arguments", {}).get("instructions", "")
            calls.append(
                _ToolCall(id=f"cli_call_{i}", function=_Function(name=name, arguments=json.dumps({"instructions": instructions})))
            )
        message = _Message(content=None, tool_calls=calls or None)
        if not calls:
            message = _Message(content=raw, tool_calls=None)
    else:
        message = _Message(content=decision.get("final_answer") or raw, tool_calls=None)

    return CLIResponse(choices=[_Choice(message=message)], usage=usage, notional_cost_usd=notional_cost)
