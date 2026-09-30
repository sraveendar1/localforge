"""Refuse a model that can't run here, and say why.

Reported: "if the model selected is too big for the machine, and also from a
storage point of view, say the issue and reject operations". Two moments:

- **Choosing** one (`/model`, `/advanced-model`, the Models section): the
  choice is refused with the reason and nothing is saved -- `selection_problem`.
- **Running** with one already chosen -- a project's saved models copied to a
  smaller machine, a disk that has since filled -- `preflight` checks the
  whole configuration before a task spends anything, and `orchestrator.run()`
  raises `ModelNotUsable` if a model can't run.

Only *local* models are judged (they're sized against this machine's memory and
disk); a cloud model has no such limit. A local model that isn't in the catalog
is sized from what Ollama says is on disk; one that's neither is left alone,
since there's nothing to judge it by.
"""
from __future__ import annotations

from localforge import delegate_target
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import ModelEntry, fit_problem, load_catalog
from localforge.hardware import HardwareProfile, detect_hardware


class ModelNotUsable(RuntimeError):
    """A chosen model can't run on this machine. The message says which one,
    why, and how to change it."""


def installed_sizes() -> dict[str, float] | None:
    """{Ollama tag: size in GB} for what's on disk, or None if Ollama can't
    be asked (not running): then disk and unknown models aren't judged."""
    try:
        return {m["name"]: m.get("size", 0) / 1e9 for m in OllamaBackend().list_installed()}
    except Exception:  # noqa: BLE001 - "can't tell" must never block a choice
        return None


def entry_for(name: str, sizes: dict[str, float] | None, modality: str = "general") -> ModelEntry | None:
    """The catalog entry for a local model, or one built from its size on disk
    when it isn't in the catalog. None when there's nothing to size it by."""
    for entry in load_catalog():
        if entry.name == name:
            return entry
    if sizes and name in sizes and sizes[name] > 0:
        return ModelEntry(
            name=name, modality=modality, runtime="ollama", min_vram_gb=0, min_ram_gb=0,
            disk_gb=sizes[name], quality_tier=0,
        )
    return None


def selection_problem(
    name: str,
    hardware: HardwareProfile | None = None,
    sizes: dict[str, float] | None = None,
    *,
    fetch_sizes: bool = True,
) -> str | None:
    """Why the local model `name` can't be used here, or None. `name` is a
    bare Ollama tag (strip any `ollama/` prefix first)."""
    sizes = sizes if sizes is not None or not fetch_sizes else installed_sizes()
    entry = entry_for(name, sizes)
    if entry is None:
        return None
    hardware = hardware or detect_hardware()
    installed = set(sizes) if sizes is not None else None
    return fit_problem(entry, hardware, installed)


def refusal(setting: str, name: str, problem: str, fix: str) -> str:
    return f"{setting} is set to {name}, but it can't run on this machine: {problem}. {fix}"


def preflight(orchestrator_model: str, hardware: HardwareProfile | None = None) -> list[str]:
    """Every chosen local model that can't run here, as messages. Empty when
    everything is usable (or can't be judged). Checked before a task starts."""
    from localforge import local_transport

    sizes = installed_sizes()
    hardware = hardware or detect_hardware()
    problems: list[str] = []
    if orchestrator_model.startswith(local_transport.PREFIXES):
        name = local_transport.model_name(orchestrator_model)
        if why := selection_problem(name, hardware, sizes, fetch_sizes=False):
            problems.append(refusal("The orchestrator", name, why, "Pick another with `/model`."))
    for modality in delegate_target.MODALITIES:
        target = delegate_target.get(modality)
        if target.kind != "ollama":
            continue  # auto already picks something that fits; cloud has no limit here
        if why := selection_problem(target.model, hardware, sizes, fetch_sizes=False):
            problems.append(
                refusal(f"The {modality} model", target.model, why, f"Change it with `/advanced-model {modality} auto`.")
            )
    return problems
