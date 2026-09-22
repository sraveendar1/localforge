"""The plan -> delegate -> collect loop. The frontier model (any provider
LiteLLM supports) drives the plan; this loop mechanically executes whatever
tool calls it requests against local models and feeds results back, until
the frontier model stops requesting tools.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import litellm
from litellm import completion

from localforge import cli_transport, local_transport, memory
from localforge.backends.ollama import OllamaBackend
from localforge.hardware import HardwareProfile, detect_hardware
from localforge.tools import ActivityHooks, DelegateCallback, Dispatcher, build_tool_schemas
from localforge.workspace import Workspace


def _installed_models() -> set[str] | None:
    """Ollama tags on disk, or None if Ollama can't be asked (the dispatcher
    then falls back to plain best-fit selection).
    """
    try:
        return {m["name"] for m in OllamaBackend().list_installed()}
    except Exception:  # noqa: BLE001 - a nice-to-have preference, never a reason to fail the run
        return None


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


MAX_ROUNDS = 40

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


# An answer that only announces work ("I will create...", "Let me write...").
_PROMISE = re.compile(r"\b(I will|I'll|I am going to|I'm going to|Let me)\s+(now\s+)?(create|write|add|make|build|update|edit|run|fix|generate|implement|set up|delegate)\b", re.I)


def _canonical_args(arguments: str | None) -> str:
    """Tool-call arguments in a stable form, so the same call is recognised
    regardless of key order or whitespace."""
    try:
        return json.dumps(json.loads(arguments or "{}"), sort_keys=True)
    except (TypeError, ValueError):
        return str(arguments)


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
        if not response.local:
            stats.frontier_cost_usd += response.notional_cost_usd or 0.0
            stats.frontier_via_subscription = True
        return

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:  # noqa: BLE001 - cost is a nice-to-have metric, never worth failing the run over
        cost = 0.0
    stats.frontier_cost_usd += cost or 0.0


SYSTEM_PROMPT = """You are the orchestrator inside localforge, a coding harness in the user's terminal, working in their project folder. You plan, investigate and review; local open-weight models running on the user's machine write the code and docs.

How to work:
- Do only what the user asked. If the message is a greeting, a question, or a chat, just answer it -- don't create, change or run anything unless they asked for that.
- Investigate before changing anything: list_files, search and read_file show you the project. Never guess at code you haven't read.
- To create a file or change code, call delegate_coding_task (or delegate_docs_task) with a `path`. The local model writes that file's complete new contents; localforge shows the user a diff and asks before saving. The local model sees ONLY your instructions plus the current file contents -- no other files, no conversation, no internet. So put everything it needs into `instructions`: the goal, the exact interfaces and names from other files, conventions, and any facts you looked up.
- edit_file is only for small fix-ups (a few lines), e.g. correcting a local model's mistake. Don't write whole files or features yourself.
- run_command runs shell commands in the project (git clone, tests, installs, builds); the user approves each one. Run the tests after changes when the project has them.
- The local models have no internet access. When the task needs anything current or external -- library docs, API details, versions, a URL the user mentioned -- use web_search and fetch_url yourself and pass the relevant facts along.
- For any task with more than two steps, call update_todos first with your plan, and update it as steps complete.
- This is an ongoing conversation: the user's earlier messages and your earlier work are above, and older turns may be condensed into a session-memory note.
- If the user declines a change or command, don't retry it unchanged; ask or adjust.
- Finish with a short summary of what you changed (files, commands run, results) and anything left to do.

Local models occasionally produce bad results: empty output, a refusal, something far too short for what was asked, or content that doesn't actually satisfy the subtask. Do not accept a delegated result at face value -- check it against what you asked for before using it (read_file the written file when it matters). If a result is prefixed with [WARNING: ...], that is an automated flag that something looked wrong; treat it with extra scrutiny. If a result is clearly bad, delegate that subtask again with more specific or simpler instructions rather than passing the bad result through. If you've retried and still can't get a usable result, say so plainly in your final answer instead of presenting a broken result as if it were fine."""


@dataclass
class Conversation:
    """One session's history, kept across the user's messages so the
    orchestrator has context the way Claude Code does. `memory` is the
    local-model-maintained summary of turns that were compacted away.
    """

    messages: list[dict] = field(default_factory=list)
    tool_indices: list[int] = field(default_factory=list)
    memory: str = ""
    project_snapshot: str = ""

    def system_message(self) -> dict:
        content = SYSTEM_PROMPT
        if self.project_snapshot:
            content += "\n\n" + self.project_snapshot
        if self.memory:
            content += "\n\nSession memory (condensed earlier turns, kept by a local model):\n" + self.memory
        return {"role": "system", "content": content}

    def chars(self) -> int:
        return sum(len(str(m.get("content") or "")) for m in self.messages)

    def reindex_tools(self) -> None:
        self.tool_indices = [i for i, m in enumerate(self.messages) if m.get("role") == "tool"]


def run(
    task: str,
    frontier_model: str,
    hardware: HardwareProfile | None = None,
    on_delegate: DelegateCallback | None = None,
    cli_provider: str | None = None,
    hooks: ActivityHooks | None = None,
    conversation: Conversation | None = None,
    workspace: Workspace | None = None,
) -> RunResult:
    """Run one user message to completion: the frontier model investigates,
    delegates writing to local models, and answers.

    `frontier_model` is any LiteLLM model string, e.g. "claude-opus-5",
    "gpt-5", or "ollama/llama3.1:70b" to self-host the orchestrator too.

    `cli_provider`, if given (e.g. "anthropic"), routes the frontier turns
    through that provider's own logged-in CLI instead of an API key -- see
    cli_transport. The loop below is identical either way.

    `conversation` carries history across calls (the interactive session
    passes the same one every time); without it each call starts fresh.
    `workspace` is the project folder the file/command tools act on.
    `hooks` lets the caller show activity live.

    Returns a `RunResult` with the final answer and usage metrics (frontier
    tokens/cost actually spent, and tokens local models generated instead --
    the latter never touched the frontier API at all).
    """
    hardware = hardware if hardware is not None else detect_hardware()
    hooks = hooks or ActivityHooks()
    conversation = conversation if conversation is not None else Conversation()
    installed = _installed_models()
    dispatcher = Dispatcher(hardware, installed=installed, hooks=hooks, workspace=workspace)
    tools = build_tool_schemas(hardware, dispatcher.catalog, installed)
    stats = RunStats()

    if workspace is not None and not conversation.project_snapshot:
        conversation.project_snapshot = workspace.snapshot()
    if conversation.chars() > memory.COMPACT_AT_CHARS:
        memory.compact(conversation, dispatcher, hooks)
    if not conversation.messages:
        conversation.messages.append(conversation.system_message())
    else:
        conversation.messages[0] = conversation.system_message()
    conversation.messages.append({"role": "user", "content": task})
    messages = conversation.messages

    def _finish() -> None:
        stats.local_tokens_generated = dispatcher.local_tokens_generated

    # Identical tool calls already made in this task -> their result. Small
    # local orchestrators loop, re-issuing the same call after it succeeded
    # (seen live: one file delegated three times); re-running it wastes a
    # local generation and, for writes, another approval prompt.
    done_calls: dict[tuple[str, str], str] = {}
    used_tools = nudged = False

    for round_number in range(1, MAX_ROUNDS + 1):
        if hooks.on_frontier is not None:
            hooks.on_frontier(round_number)
        if frontier_model.startswith(local_transport.PREFIXES):
            # An open-weight orchestrator on this machine: straight to Ollama,
            # never through a provider CLI, whatever else is configured.
            response = local_transport.complete(frontier_model, messages, tools)
        elif cli_provider:
            response = cli_transport.complete(cli_provider, messages, tools)
        else:
            response = completion(model=frontier_model, messages=messages, tools=tools)
        _record_frontier_usage(response, stats)
        message = response.choices[0].message
        messages.append(message.model_dump())

        if not message.tool_calls:
            if not used_tools and not nudged and _PROMISE.search(message.content or ""):
                # "Okay, I will create hello.py." with no tool call, then
                # nothing -- small local orchestrators do this (seen live with
                # gemma3:4b). Ask once to actually do it, or to just answer.
                nudged = True
                messages.append(
                    {
                        "role": "user",
                        "content": "You said what you'll do but didn't call any tool, so nothing happened. "
                        "Do it now with the tools, or if no action is needed, give your final answer.",
                    }
                )
                continue
            _finish()
            return RunResult(answer=message.content or "", stats=stats)
        used_tools = True

        for call in message.tool_calls:
            key = (call.function.name, _canonical_args(call.function.arguments))
            if key in done_calls and call.function.name != "update_todos":
                result = (
                    f"You already made this exact {call.function.name} call in this task; it was not run again. "
                    f"Its result was:\n{done_calls[key][:1500]}\n\nDon't repeat it: do the next step, or give your final answer."
                )
            else:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = dispatcher.dispatch(call.function.name, args, on_delegate=on_delegate)
                except Exception as exc:  # noqa: BLE001 - surfaced to the orchestrator model, not swallowed
                    result = f"Error running {call.function.name}: {exc}"
                done_calls[key] = result
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            conversation.tool_indices.append(len(messages) - 1)

        _collapse_old_tool_results(messages, conversation.tool_indices)

    _finish()
    raise OrchestrationError(f"Orchestration did not converge within {MAX_ROUNDS} rounds.", stats)
