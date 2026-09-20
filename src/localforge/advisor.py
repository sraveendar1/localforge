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

from localforge.catalog import ModelEntry, candidates, load_catalog
from localforge.hardware import HardwareProfile


def _best_quality(entries: list[ModelEntry]) -> ModelEntry:
    return max(entries, key=lambda e: e.quality_tier)


def recommend_models(
    hardware: HardwareProfile,
    frontier_model: str,
    catalog: list[ModelEntry] | None = None,
) -> dict[str, ModelEntry | None]:
    """Ask `frontier_model` to pick the best local model per modality for
    `hardware`. Returns one entry per modality present in the catalog (None
    if nothing fits that modality at all).
    """
    catalog = catalog if catalog is not None else load_catalog()
    modalities = sorted({m.modality for m in catalog})

    per_modality = {modality: candidates(modality, hardware, catalog) for modality in modalities}
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
        "one per modality -- weigh quality_tier against how much RAM/VRAM/"
        "disk headroom each choice leaves for actually running it "
        "alongside everything else on the machine.\n\n"
        f"Hardware: {hardware.model_dump_json()}\n\n"
        "Candidates: "
        + json.dumps({m: [e.model_dump() for e in es] for m, es in choosable.items()})
    )

    try:
        response = completion(
            model=frontier_model,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "select_models"}},
        )
        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    except Exception:  # noqa: BLE001 - any failure here falls back to the deterministic heuristic
        args = {}

    for modality, entries in choosable.items():
        picked_name = args.get(modality)
        match = next((e for e in entries if e.name == picked_name), None)
        result[modality] = match or _best_quality(entries)

    return result
