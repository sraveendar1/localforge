"""Where a delegated subtask (coding/docs/general) actually runs.

Until now `delegate_coding_task`/`delegate_docs_task`/`delegate_general_task`
always meant "the best-fitting local Ollama model" (see `catalog.best_match()`
and `tools.Dispatcher.resolve()`) -- automatic, and free. This module adds an
"advanced" override, per modality: keep automatic, pin a specific local
model, or route that modality to a paid cloud model instead -- either
metered (an API key, the same LiteLLM call the orchestrator's own API path
already makes) or via a CLI subscription login (reusing `cli_transport`'s
existing subprocess/JSON-protocol machinery). See `backends/cloud.py` for
where a cloud target actually runs, and `tools.Dispatcher.resolve()` for how
a target is turned into something `dispatch()` can call.

Stored globally (one set of choices for the whole machine, like the
orchestrator model itself) rather than per-project: which local models are
installed, and which API keys/CLI logins exist, don't vary by folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from localforge import config

MODALITIES = ("coding", "docs", "general")

TARGET_ENV_VARS = {
    "coding": "LOCALFORGE_CODING_TARGET",
    "docs": "LOCALFORGE_DOCS_TARGET",
    "general": "LOCALFORGE_GENERAL_TARGET",
}


@dataclass(frozen=True)
class DelegateTarget:
    kind: str  # "auto" | "ollama" | "api" | "cli"
    provider: str | None = None  # "anthropic" | "openai" | "gemini" -- only for api/cli
    model: str | None = None  # an Ollama tag (ollama), or the provider's model id (api/cli)

    @property
    def is_cloud(self) -> bool:
        return self.kind in ("api", "cli")


AUTO = DelegateTarget(kind="auto")


def parse(raw: str | None) -> DelegateTarget:
    """Parse a stored target string. Anything unset, "auto", or otherwise
    unrecognized (a stale value from an older localforge, hand-edited
    config, etc.) is treated as automatic rather than raising -- a bad
    saved value should degrade to today's behavior, never break delegation
    outright."""
    if not raw or raw == "auto":
        return AUTO
    kind, _, rest = raw.partition(":")
    if kind == "ollama" and rest:
        return DelegateTarget(kind="ollama", model=rest)
    if kind in ("api", "cli") and rest:
        provider, _, model = rest.partition(":")
        if provider and model:
            return DelegateTarget(kind=kind, provider=provider, model=model)
    return AUTO


def render(target: DelegateTarget) -> str:
    if target.kind == "auto":
        return "auto"
    if target.kind == "ollama":
        return f"ollama:{target.model}"
    return f"{target.kind}:{target.provider}:{target.model}"


def get(modality: str) -> DelegateTarget:
    return parse(os.environ.get(TARGET_ENV_VARS[modality]))


def get_all() -> dict[str, DelegateTarget]:
    return {modality: get(modality) for modality in MODALITIES}


def set_target(modality: str, target: DelegateTarget) -> None:
    config.save({TARGET_ENV_VARS[modality]: render(target)})


def clear(modality: str) -> None:
    set_target(modality, AUTO)


def describe(target: DelegateTarget) -> str:
    """One line for a human, e.g. in `localforge local-model` or the GUI."""
    if target.kind == "auto":
        return "auto (best-fitting installed local model)"
    if target.kind == "ollama":
        return f"{target.model} (local, via Ollama)"
    via = "your API key" if target.kind == "api" else "your CLI login"
    return f"{target.model} (cloud, {target.provider}, via {via})"
