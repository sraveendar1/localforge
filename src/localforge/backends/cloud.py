"""Delegate backends that aren't local: an "advanced" per-modality override
lets coding/docs/general delegate to a paid cloud model instead of the
usual free local Ollama pick -- either metered (an API key, the same
LiteLLM call the orchestrator's own API-key path already makes) or via a
CLI subscription login (reusing `cli_transport`'s existing subprocess/
JSON-protocol machinery, the same one the orchestrator uses under CLI
login). See `delegate_target.py` for the override itself and
`tools.Dispatcher.resolve()` for how it becomes a `ModelEntry` these
backends can run.

Both backends are one-shot: no tool calling, no multi-turn planning -- just
"here are the instructions, write the content" -- which is exactly the
delegate contract every local backend already fulfills. `BackendResult`'s
`cost_usd`/`notional_cost_usd` (see backends/base.py) are how a cloud
delegate's real (or subscription-notional) spend reaches usage accounting,
mirroring `RunStats.frontier_cost_usd`/`frontier_via_subscription`.
"""

from __future__ import annotations

import os
from typing import Callable

import litellm

from localforge import cli_transport, config
from localforge.backends.base import BackendResult


class CloudProviderNotConfigured(RuntimeError):
    """The provider a cloud delegate target names has no usable API
    key/CLI login -- raised instead of letting the underlying call fail
    with a confusing, unrelated error."""


class CloudApiBackend:
    """Runs a delegate task via a plain LiteLLM completion, billed like any
    other API call -- the same mechanism the orchestrator's own API-key
    transport uses, just for a single one-shot generation instead of a
    planning loop."""

    def ensure_available(self, model_name: str, on_progress: object = None, provider: str | None = None) -> None:
        env_var = config.FRONTIER_PROVIDERS.get(provider or "")
        if env_var and not os.environ.get(env_var):
            raise CloudProviderNotConfigured(
                f"No {env_var} is set, so {provider} can't be used as a delegate. "
                "Set it (e.g. via `localforge setup`) or pick a different target with `localforge local-model`."
            )

    def generate(
        self, model_name: str, prompt: str, on_token: Callable[[str], None] | None = None,
        provider: str | None = None, **kwargs,
    ) -> BackendResult:
        model_id = config.litellm_model_id(model_name)
        response = litellm.completion(model=model_id, messages=[{"role": "user", "content": prompt}])
        message = response.choices[0].message
        content = message.content or ""
        if on_token is not None and content:
            on_token(content)  # LiteLLM's non-streaming call returns the whole reply at once
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - an unpriced/unknown model shouldn't fail the delegation
            cost = 0.0
        return {"type": "text", "content": content, "tokens": tokens, "cost_usd": cost, "notional_cost_usd": 0.0}


class CloudCliBackend:
    """Runs a delegate task through a provider's logged-in CLI (the same
    subscription the orchestrator itself may already be using), reusing
    cli_transport.complete() with an empty tool list -- there's nothing to
    call, just instructions to follow and content to write, exactly the
    JSON-decision "final_answer" path local/CLI orchestrators already use."""

    def ensure_available(self, model_name: str, on_progress: object = None, provider: str | None = None) -> None:
        if provider and not cli_transport.available(provider):
            raise CloudProviderNotConfigured(cli_transport.requirements_message(provider))

    def generate(
        self, model_name: str, prompt: str, on_token: Callable[[str], None] | None = None,
        provider: str | None = None, **kwargs,
    ) -> BackendResult:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Proceed."},
        ]
        response = cli_transport.complete(provider, messages, tools=[], model=model_name, on_text=on_token)
        message = response.choices[0].message
        content = message.content or ""
        return {
            "type": "text",
            "content": content,
            "tokens": response.usage.completion_tokens,
            "cost_usd": 0.0,
            "notional_cost_usd": response.notional_cost_usd,
        }
