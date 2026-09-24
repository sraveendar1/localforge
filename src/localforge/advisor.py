"""Lets the frontier model choose which local models to download, instead of
the deterministic `catalog.best_match()` heuristic. The frontier model only
ever sees, and can only pick from, catalog entries that already pass the
hardware/disk fit check (`catalog.candidates()`) -- it cannot invent a model
we have no backend for. If the call fails or returns something invalid, we
fall back to the deterministic highest-quality-tier pick.
"""

from __future__ import annotations

import json

from litellm import completion

from localforge.catalog import (
    INTERNAL_MODALITIES,
    ModelEntry,
    balanced_pick,
    candidates,
    load_catalog,
    running_gb,
    share_text_model,
    tokens_per_second,
)
from localforge.hardware import HardwareProfile


def _best_choice(entries: list[ModelEntry], installed: set[str] | None, hardware: HardwareProfile) -> ModelEntry:
    """Deterministic fallback: an already-installed fitting model beats a
    higher-tier one that would need a fresh download, and otherwise the
    balanced pick (same rules as catalog.best_match).
    """
    if installed:
        already_have = [e for e in entries if e.name in installed]
        if already_have:
            return max(already_have, key=lambda e: e.quality_tier)
    return balanced_pick(entries, hardware)


def _ask_via_cli(cli_provider: str, prompt: str, choosable: dict[str, list[ModelEntry]], model: str | None = None) -> dict:
    """Same question, asked through a logged-in CLI instead of an API key.

    The CLI has no native tool-calling/enum constraint, so we ask for plain
    JSON and let the caller validate the names against `choosable` -- which
    it already does anyway as a safety net against a hallucinated model.
    """
    from localforge import cli_transport

    options = {m: [e.name for e in es] for m, es in choosable.items()}
    messages = [
        {
            "role": "user",
            "content": (
                prompt
                + "\n\nReply with ONLY a JSON object mapping each modality to one "
                "model name from its list, e.g. "
                + json.dumps({m: names[0] for m, names in options.items()})
            ),
        }
    ]
    response = cli_transport.complete(cli_provider, messages, tools=[], model=model)
    content = response.choices[0].message.content or ""
    return cli_transport._extract_json(content) or {}


def recommend_models(
    hardware: HardwareProfile,
    frontier_model: str,
    catalog: list[ModelEntry] | None = None,
    cli_provider: str | None = None,
    installed: set[str] | None = None,
) -> dict[str, ModelEntry | None]:
    """Ask `frontier_model` to pick the best local model per modality for
    `hardware`. Returns one entry per modality present in the catalog (None
    if nothing fits that modality at all).

    `installed` is the set of Ollama tags already on disk. Each candidate is
    labelled with it so the frontier model can weigh "no download needed"
    against a quality bump, and the deterministic fallback prefers installed
    models outright.
    """
    installed = installed or set()
    catalog = catalog if catalog is not None else load_catalog()
    # INTERNAL_MODALITIES (e.g. "judge") are resolved directly by
    # tools.Dispatcher, never through setup/the advisor: there's exactly one
    # candidate anyway, so asking the frontier model to "pick" would just
    # spend a tool-call round trip deciding nothing.
    modalities = sorted({m.modality for m in catalog} - INTERNAL_MODALITIES)

    per_modality = {modality: candidates(modality, hardware, catalog, installed) for modality in modalities}
    choosable = {modality: entries for modality, entries in per_modality.items() if entries}

    result: dict[str, ModelEntry | None] = {modality: None for modality in modalities if modality not in choosable}
    if not choosable:
        return result

    tool = {
        "type": "function",
        "function": {
            "name": "select_models",
            "description": (
                "Pick the single best local model for each task modality, "
                "given this machine's hardware. Every candidate listed "
                "already fits the machine's RAM/VRAM/disk space."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    modality: {
                        "type": "string",
                        "enum": [e.name for e in entries],
                        "description": f"Best {modality} model for this hardware.",
                    }
                    for modality, entries in choosable.items()
                },
                "required": list(choosable.keys()),
            },
        },
    }

    prompt = (
        "Here is a machine's hardware profile and, per task modality, the "
        "local models that fit it. Call select_models to choose the best "
        "one per modality. Aim for balance: the best quality that still runs "
        "at a comfortable speed (est_tokens_per_second of 15 or more) and "
        "leaves the machine usable. Every candidate already fits in memory "
        "with room kept for the OS and apps (running_gb includes its context "
        "cache). If two different models can't be in memory together, "
        "prefer using one model for several modalities over constant "
        "reloading.\n\n"
        "Candidates marked \"installed\": true are ALREADY on this machine and "
        "need no download at all. Strongly prefer an installed model unless a "
        "not-installed one is clearly better for the task -- a multi-GB "
        "download for a marginal quality gain is a bad trade.\n\n"
        f"Hardware: {hardware.model_dump_json()}\n\n"
        "Candidates: "
        + json.dumps(
            {
                m: [
                    {
                        **e.model_dump(),
                        "installed": e.name in installed,
                        "running_gb": round(running_gb(e), 1),
                        "est_tokens_per_second": round(tokens_per_second(e, hardware)),
                    }
                    for e in es
                ]
                for m, es in choosable.items()
            }
        )
    )

    try:
        if cli_provider:
            args = _ask_via_cli(cli_provider, prompt, choosable, model=frontier_model)
        else:
            response = completion(
                model=frontier_model,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": "select_models"}},
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    except Exception:  # noqa: BLE001 - any failure here falls back to the deterministic heuristic
        args = {}

    fell_back = False
    for modality, entries in choosable.items():
        picked_name = args.get(modality)
        match = next((e for e in entries if e.name == picked_name), None)
        if match is None:
            result[modality] = _best_choice(entries, installed, hardware)
            fell_back = True
        else:
            result[modality] = match
    # The deterministic fallback shares one text model on a small machine,
    # as catalog.recommendations() does; a frontier model's own picks are kept.
    return share_text_model(result, hardware, installed) if fell_back else result
