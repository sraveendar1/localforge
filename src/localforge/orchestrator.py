"""The plan -> delegate -> collect loop. The frontier model (any provider
LiteLLM supports) drives the plan; this loop mechanically executes whatever
tool calls it requests against local models and feeds results back, until
the frontier model stops requesting tools.
"""

from __future__ import annotations

import json

from litellm import completion

from localforge.hardware import HardwareProfile, detect_hardware
from localforge.tools import Dispatcher, build_tool_schemas

MAX_ROUNDS = 25


def run(task: str, frontier_model: str, hardware: HardwareProfile | None = None) -> str:
    """Run `task` to completion, delegating subtasks to local models.

    `frontier_model` is any LiteLLM model string, e.g. "claude-opus-5",
    "gpt-5", or "ollama/llama3.1:70b" if you want to self-host the
    orchestrator too.
    """
    hardware = hardware if hardware is not None else detect_hardware()
    dispatcher = Dispatcher(hardware)
    tools = build_tool_schemas()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an orchestrator. Break the user's request into subtasks and "
                "delegate each one to the appropriate tool. Do not do the work yourself; "
                "delegate it, then combine the results into a final answer."
            ),
        },
        {"role": "user", "content": task},
    ]

    for _ in range(MAX_ROUNDS):
        response = completion(model=frontier_model, messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message.model_dump())

        if not message.tool_calls:
            return message.content or ""

        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            try:
                result = dispatcher.dispatch(call.function.name, args["instructions"])
            except Exception as exc:  # noqa: BLE001 - surfaced to the orchestrator model, not swallowed
                result = f"Error running {call.function.name}: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    raise RuntimeError(f"Orchestration did not converge within {MAX_ROUNDS} rounds.")
