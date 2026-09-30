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

Kept per project (see project_models.py: `<project>/.localforge/models.json`),
falling back to your global defaults (`localforge setup`) for a project that
hasn't saved its own. Which local models are installed and which API keys or
CLI logins exist don't vary by folder, but which of them a project should use
does -- and a model too big for the machine is refused when chosen (see
model_fit.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from localforge import config

# Text work: written by a local model by default, or a pinned/cloud one.
MODALITIES = ("coding", "docs", "general")
# Generated media. Image works only through a paid cloud image model -- an API
# key, or OpenAI's Codex CLI login -- (there is no local image backend yet, so
# "auto" means off: the orchestrator is simply not offered the tool). Video isn't built at all -- it has a row so the
# choice will have somewhere to live, and can't be set yet.
GENERATIVE = ("image", "video")
ALL_MODALITIES = MODALITIES + GENERATIVE

TARGET_ENV_VARS = {
    "coding": "LOCALFORGE_CODING_TARGET",
    "docs": "LOCALFORGE_DOCS_TARGET",
    "general": "LOCALFORGE_GENERAL_TARGET",
    "image": "LOCALFORGE_IMAGE_TARGET",
    "video": "LOCALFORGE_VIDEO_TARGET",
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


def set_target(modality: str, target: DelegateTarget, scope: str = "project") -> None:
    """Save a choice: to the active project's models file (see
    project_models.py) by default, or -- scope="global" -- to the defaults
    every project starts from (what `localforge setup` sets)."""
    from localforge import project_models

    project_models.save({TARGET_ENV_VARS[modality]: render(target)}, scope)


def clear(modality: str, scope: str = "project") -> None:
    set_target(modality, AUTO, scope)


class InvalidTarget(ValueError):
    """A target string that would fail on first actual use -- an unknown
    modality, a local model not in the catalog, or a cloud provider with no
    usable API key/CLI login. Carries a human-readable reason so both the
    CLI (localforge advanced-model) and the desktop app (its "Change" picker
    and its /advanced-model chat command) can show the same message without
    duplicating this validation."""


def apply(modality: str, value: str, scope: str = "project") -> DelegateTarget:
    """Parse, validate, and persist a target from a raw string: 'auto', a
    local catalog model name, or api:<provider>:<model> / cli:<provider>:
    <model> for a cloud target. Raises InvalidTarget rather than silently
    accepting something that would break the next delegation -- including a local
    model too big for this machine's memory or disk. `scope` is where it is
    saved: the active project (default) or the global defaults (setup)."""
    if modality not in ALL_MODALITIES:
        raise InvalidTarget(f"Unknown task type {modality!r}. Use coding, docs, general, image, or video.")
    value = value.strip()
    if modality == "video":
        if value.lower() == "auto":
            clear(modality, scope)
            return AUTO
        raise InvalidTarget("Video generation isn't available yet, so there's nothing to choose for it.")
    if value.lower() == "auto":
        clear(modality, scope)
        return AUTO
    if modality == "image":
        return _apply_image(value, scope)
    if value.startswith("api:") or value.startswith("cli:"):
        target = parse(value)
        if target is AUTO:
            raise InvalidTarget(
                f"Couldn't parse {value!r}. Expected api:<provider>:<model> or cli:<provider>:<model>, "
                "e.g. api:anthropic:claude-haiku-4-5."
            )
        if target.kind == "api":
            env_var = config.FRONTIER_PROVIDERS.get(target.provider)
            if not env_var or not os.environ.get(env_var):
                raise InvalidTarget(
                    f"No API key set for {target.provider}. Run `localforge setup` to add one, "
                    "or use a local model instead."
                )
        else:
            from localforge import cli_transport  # local: avoids a module-load-order dependency

            if not cli_transport.available(target.provider):
                raise InvalidTarget(cli_transport.requirements_message(target.provider))
        set_target(modality, target, scope)
        return target
    from localforge.catalog import load_catalog  # local: avoids a module-load-order dependency

    match = next((m for m in load_catalog() if m.modality == modality and m.name == value), None)
    if match is None:
        raise InvalidTarget(
            f"{value!r} isn't a {modality} model in the catalog. "
            "Run `localforge catalog` to see options, or pass 'auto'/an api:.../cli:... target."
        )
    from localforge import model_fit  # local: model_fit imports this module

    if why := model_fit.selection_problem(value):
        raise InvalidTarget(
            f"{value} can't run on this machine for {modality}: {why}. Nothing was changed. "
            "Pick a smaller model, or 'auto' to let localforge choose one that fits."
        )
    target = DelegateTarget(kind="ollama", model=value)
    set_target(modality, target, scope)
    return target


def _apply_image(value: str, scope: str = "project") -> DelegateTarget:
    """An image target is a paid cloud image model through an API key
    (api:openai:gpt-image-1), or an image tool inside a provider's own CLI
    login -- only OpenAI's Codex has one (cli:openai:codex-image). A local
    model has no image backend yet."""
    target = parse(value)
    if target.kind not in ("api", "cli"):
        raise InvalidTarget(
            f"{value!r} can't generate images. Use api:<provider>:<model> with an image model, e.g. "
            "api:openai:gpt-image-1, cli:openai:codex-image to use your ChatGPT login through Codex, "
            "or 'auto' to turn image generation off."
        )
    if target.kind == "cli":
        models = config.IMAGE_CLI_CHOICES.get(target.provider)
        if models is None:
            raise InvalidTarget(
                f"{target.provider}'s CLI login can't generate images"
                + (" (Claude has no image model)" if target.provider == "anthropic" else "")
                + ". Only OpenAI's Codex login can: cli:openai:codex-image. Or use an API key with an image model."
            )
        if target.model not in models:
            raise InvalidTarget(f"{target.model!r} isn't a known {target.provider} CLI image option. Use: " + ", ".join(models) + ".")
        from localforge import cli_transport  # local: avoids a module-load-order dependency

        if not cli_transport.available(target.provider):
            raise InvalidTarget(cli_transport.requirements_message(target.provider))
        set_target("image", target, scope)
        return target
    models = config.IMAGE_MODEL_CHOICES.get(target.provider)
    if models is None:
        raise InvalidTarget(
            f"{target.provider} has no image generation here. Providers with image models: "
            + ", ".join(sorted(config.IMAGE_MODEL_CHOICES)) + "."
        )
    if target.model not in models:
        raise InvalidTarget(f"{target.model!r} isn't a known {target.provider} image model. Choose one of: " + ", ".join(models) + ".")
    env_var = config.FRONTIER_PROVIDERS.get(target.provider)
    if not env_var or not os.environ.get(env_var):
        raise InvalidTarget(f"No API key set for {target.provider}. Run `localforge setup` to add one.")
    set_target("image", target, scope)
    return target


def describe_modality(modality: str, target: DelegateTarget | None = None) -> str:
    """describe() for any task type, with the two generated ones worded for
    what "auto" really means for them."""
    target = target if target is not None else get(modality)
    if modality == "video":
        return "not available yet"
    if modality == "image" and target.kind == "auto":
        return "off -- no image model chosen (pick a paid one in Advanced)"
    return describe(target)


def describe(target: DelegateTarget) -> str:
    """One line for a human, e.g. in `localforge advanced-model` or the GUI."""
    if target.kind == "auto":
        return "auto (best-fitting installed local model)"
    if target.kind == "ollama":
        return f"{target.model} (local, via Ollama)"
    via = "your API key" if target.kind == "api" else "your CLI login"
    return f"{target.model} (cloud, {target.provider}, via {via})"
