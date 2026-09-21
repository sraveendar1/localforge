"""The plan -> delegate -> collect loop. The frontier model (any provider
LiteLLM supports) drives the plan; this loop mechanically executes whatever
tool calls it requests against local models and feeds results back, until
the frontier model stops requesting tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import litellm
from litellm import completion

from localforge import cli_transport
from localforge.hardware import HardwareProfile, detect_hardware
from localforge.tools import DelegateCallback, Dispatcher, build_tool_schemas


@dataclass
class RunStats:
    """Usage metrics for one `run()` call, to show what delegating to local
    models actually saved versus sending everything through the frontier
    model's API.
    """

    frontier_prompt_tokens: int = 0
    frontier_completion_tokens: int = 0
    frontier_cost_usd: float = 0.0
    local_tokens_generated: int = 0
    # True when the frontier model ran through its own logged-in CLI, so
    # frontier_cost_usd is what pay-per-token billing *would* have cost, not
    # money charged separately -- it came out of the account's subscription.
    frontier_via_subscription: bool = False

    @property
    def frontier_total_tokens(self) -> int:
        return self.frontier_prompt_tokens + self.frontier_completion_tokens


@dataclass
class RunResult:
    answer: str
    stats: RunStats = field(default_factory=RunStats)


class OrchestrationError(RuntimeError):
    """Raised when the loop doesn't converge within MAX_ROUNDS. Carries the
    usage stats accumulated up to that point, so a caller can still show the
    user what was actually spent instead of losing that entirely -- a
    non-convergent task still burns real frontier tokens/cost and local
    compute along the way.
    """

    def __init__(self, message: str, stats: RunStats):
        super().__init__(message)
        self.stats = stats


MAX_ROUNDS = 25

# How many of the most recent tool results to keep in full. Once a task runs
# long enough to accumulate more than this many, older ones are collapsed to
# a short placeholder in place -- otherwise a long task's full history
# (including large generated files) gets resent to the frontier model every
# single round, growing context size and cost unboundedly. The frontier
# model still sees that it made each call (the assistant's tool_calls
# message is never touched), just not the full old result content.
KEEP_RECENT_TOOL_RESULTS = 4


def _collapse_old_tool_results(messages: list[dict], tool_message_indices: list[int]) -> None:
    excess = len(tool_message_indices) - KEEP_RECENT_TOOL_RESULTS
    if excess <= 0:
        return
    for idx in tool_message_indices[:excess]:
        content = messages[idx]["content"]
        if content.startswith("[superseded:"):
            continue  # already collapsed on a previous round
        messages[idx]["content"] = (
            f"[superseded: earlier result, {len(content)} chars -- no longer kept in full in context]"
        )


def _record_frontier_usage(response, stats: RunStats) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None:
        stats.frontier_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        stats.frontier_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    # A CLI-transport response reports what it *would* have cost; an API
    # response has to be priced by LiteLLM. Checked by type, not by probing
    # for an attribute -- duck-typing here misfires on anything that
    # auto-creates attributes (mocks, proxies) and silently mis-bills.
    if isinstance(response, cli_transport.CLIResponse):
        stats.frontier_cost_usd += response.notional_cost_usd or 0.0
        stats.frontier_via_subscription = True
        return

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:  # noqa: BLE001 - cost is a nice-to-have metric, never worth failing the run over
        cost = 0.0
    stats.frontier_cost_usd += cost or 0.0


def run(
    task: str,
    frontier_model: str,
    hardware: HardwareProfile | None = None,
    on_delegate: DelegateCallback | None = None,
    cli_provider: str | None = None,
) -> RunResult:
    """Run `task` to completion, delegating subtasks to local models.

    `frontier_model` is any LiteLLM model string, e.g. "claude-opus-5",
    "gpt-5", or "ollama/llama3.1:70b" if you want to self-host the
    orchestrator too. `on_delegate`, if given, is called with
    (modality, ModelEntry) right before each subtask is handed to a local
    model, so the caller can show the user what's doing the work.

    `cli_provider`, if given (e.g. "anthropic"), routes the frontier turns
    through that provider's own logged-in CLI instead of an API key -- see
    cli_transport. The loop below is identical either way; only the call
    that produces a response differs.

    Returns a `RunResult` with the final answer and usage metrics (frontier
    tokens/cost actually spent, and tokens local models generated instead --
    the latter never touched the frontier API at all).
    """
    hardware = hardware if hardware is not None else detect_hardware()
    dispatcher = Dispatcher(hardware)
    tools = build_tool_schemas(hardware, dispatcher.catalog)
    stats = RunStats()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an orchestrator. Break the user's request into subtasks and "
                "delegate each one to the appropriate tool. Do not do the work yourself; "
                "delegate it, then combine the results into a final answer.\n\n"
                "Local models occasionally produce bad results: empty output, a refusal, "
                "something far too short for what was asked, or content that doesn't "
                "actually satisfy the subtask. Do not accept a delegated result at face "
                "value -- check it against what you asked for before using it. If a "
                "result is prefixed with [WARNING: ...], that is an automated flag that "
                "something looked wrong; treat it with extra scrutiny. If a result is "
                "clearly bad, delegate that subtask again with more specific or simpler "
                "instructions rather than passing the bad result through. If you've "
                "retried and still can't get a usable result, say so plainly in your "
                "final answer instead of presenting a broken result as if it were fine."
            ),
        },
        {"role": "user", "content": task},
    ]

    tool_message_indices: list[int] = []

    for _ in range(MAX_ROUNDS):
        if cli_provider:
            response = cli_transport.complete(cli_provider, messages, tools)
        else:
            response = completion(model=frontier_model, messages=messages, tools=tools)
        _record_frontier_usage(response, stats)
        message = response.choices[0].message
        messages.append(message.model_dump())

        if not message.tool_calls:
            stats.local_tokens_generated = dispatcher.local_tokens_generated
            return RunResult(answer=message.content or "", stats=stats)

        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            try:
                result = dispatcher.dispatch(call.function.name, args["instructions"], on_delegate=on_delegate)
            except Exception as exc:  # noqa: BLE001 - surfaced to the orchestrator model, not swallowed
                result = f"Error running {call.function.name}: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
            tool_message_indices.append(len(messages) - 1)

        _collapse_old_tool_results(messages, tool_message_indices)

    stats.local_tokens_generated = dispatcher.local_tokens_generated
    raise OrchestrationError(f"Orchestration did not converge within {MAX_ROUNDS} rounds.", stats)
