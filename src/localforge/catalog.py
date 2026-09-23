"""Static rules + curated catalog: picks the best-fitting local model for a
task's modality given the detected hardware. See catalog_data.yaml for the
model list.

"Fits" means the model can run *while the machine stays usable*, not just
that its file is smaller than RAM (reported: pick the right balance and
keep system performance in mind). A running model needs its weights, a
context cache that grows with the context window localforge asks for, and
some overhead -- and the OS and the user's apps need room too. Among models
that fit, the pick is balanced: the best quality that still generates at a
comfortable speed (MIN_TOKENS_PER_SECOND), estimated from the machine's
memory bandwidth, since generating each token reads the model's active
weights once.
"""

from __future__ import annotations

from importlib import resources

import yaml
from pydantic import BaseModel

from localforge.hardware import HardwareProfile


class ModelEntry(BaseModel):
    name: str
    modality: str
    runtime: str
    min_vram_gb: float
    min_ram_gb: float
    disk_gb: float  # approximate download size
    quality_tier: int
    # Context (KV) cache per 8k tokens of context window, GB, from the
    # model's architecture. 0 = unknown: estimated from its size.
    kv_gb_per_8k: float = 0.0
    # Weights read per generated token, GB: the whole model for a dense one,
    # far less for a mixture-of-experts. None = the whole download.
    active_gb: float | None = None


class NoFittingModelError(RuntimeError):
    """Raised when no catalog entry for a modality fits the detected hardware."""


def load_catalog() -> list[ModelEntry]:
    data = yaml.safe_load(resources.files("localforge").joinpath("catalog_data.yaml").read_text())
    return [ModelEntry(**m) for m in data["models"]]


# --- memory and speed -----------------------------------------------------------

# Kept free for the OS and the user's own apps: the larger of these.
SYSTEM_RESERVE_GB = 4.0
SYSTEM_RESERVE_FRACTION = 0.25
GPU_USABLE_FRACTION = 0.9  # drivers and the display need some VRAM
RUNTIME_OVERHEAD_GB = 0.5  # Ollama's own buffers per loaded model
WEIGHTS_IN_MEMORY = 1.1  # a loaded model is a little bigger than its download
# Balanced: the best model that still writes at least this fast. Below it a
# file takes minutes and the machine is busy the whole time.
MIN_TOKENS_PER_SECOND = 15
BANDWIDTH_EFFICIENCY = 0.75  # real generation reaches ~75% of peak bandwidth
# Typical bandwidth (GB/s) when the exact chip isn't known.
TYPICAL_BANDWIDTH = {"cuda": 300.0, "metal": 100.0, "rocm": 300.0, "cpu": 50.0}


def _context_sizes() -> list[int]:
    """Context windows a model may be given, largest first: max_context()
    halved down to the smallest window localforge uses."""
    from localforge.backends.ollama import MIN_NUM_CTX, max_context

    sizes, size = [], max_context()
    while size >= MIN_NUM_CTX:
        sizes.append(size)
        size //= 2
    return sizes or [MIN_NUM_CTX]


def running_gb(entry: ModelEntry, context_tokens: int | None = None) -> float:
    """Memory the model takes while running with a `context_tokens` window
    (Ollama allocates the whole window up front). Defaults to the smallest
    window, i.e. the least the model can run with."""
    context_tokens = context_tokens if context_tokens is not None else _context_sizes()[-1]
    kv_per_8k = entry.kv_gb_per_8k or entry.disk_gb * 0.1
    return entry.disk_gb * WEIGHTS_IN_MEMORY + kv_per_8k * context_tokens / 8192 + RUNTIME_OVERHEAD_GB


def ram_budget_gb(hw: HardwareProfile) -> float:
    """RAM a model may use while the OS and the user's apps keep theirs."""
    return max(0.0, hw.ram_gb - max(SYSTEM_RESERVE_GB, hw.ram_gb * SYSTEM_RESERVE_FRACTION))


def resident_capacity_gb(hw: HardwareProfile) -> float:
    """Memory all loaded models together may use: unified memory on Apple
    Silicon (the GPU's share of the same RAM), otherwise VRAM plus spare RAM
    -- a model that doesn't fit on the GPU runs from system RAM."""
    if any(g.backend == "metal" for g in hw.gpus):
        return min(hw.total_vram_gb * GPU_USABLE_FRACTION, ram_budget_gb(hw))
    return hw.total_vram_gb * GPU_USABLE_FRACTION + ram_budget_gb(hw)


def _placement(entry: ModelEntry, hw: HardwareProfile, context_tokens: int | None = None) -> str | None:
    """Where the model would run: "gpu", "cpu", or None if it doesn't fit.
    Apple Silicon's GPU shares RAM, so both limits apply there. A model
    split between VRAM and RAM runs at CPU speed, so it counts as "cpu"."""
    need = running_gb(entry, context_tokens)
    metal = any(g.backend == "metal" for g in hw.gpus)
    if hw.gpus and need <= hw.total_vram_gb * GPU_USABLE_FRACTION and (not metal or need <= ram_budget_gb(hw)):
        return "gpu"
    if not metal and need <= ram_budget_gb(hw):
        return "cpu"
    return None


def context_limit(entry: ModelEntry, hw: HardwareProfile) -> int | None:
    """The largest context window this model can run with here while the
    machine stays usable, or None if it can't run even with the smallest.
    A 12B model on a 16 GB Mac gets 8k rather than being ruled out -- or
    being given 32k and pushing the machine into swap."""
    for size in _context_sizes():
        if _placement(entry, hw, size) is not None:
            return size
    return None


def tokens_per_second(entry: ModelEntry, hw: HardwareProfile) -> float:
    """Rough generation speed: each token reads the active weights once."""
    placement = _placement(entry, hw)
    if placement == "gpu":
        bandwidth = hw.memory_bandwidth_gbps or TYPICAL_BANDWIDTH.get(hw.gpus[0].backend, 100.0)
    else:
        metal = any(g.backend == "metal" for g in hw.gpus)
        bandwidth = (hw.memory_bandwidth_gbps if metal or not hw.gpus else None) or TYPICAL_BANDWIDTH["cpu"]
    active = entry.active_gb or entry.disk_gb
    return bandwidth * BANDWIDTH_EFFICIENCY / max(active, 0.1)


def balanced_pick(entries: list[ModelEntry], hw: HardwareProfile) -> ModelEntry:
    """The best quality among models fast enough to be comfortable; if none
    is, the fastest one. Ties go to the smaller model."""
    fast = [e for e in entries if tokens_per_second(e, hw) >= MIN_TOKENS_PER_SECOND]
    if fast:
        return max(fast, key=lambda e: (e.quality_tier, context_limit(e, hw) or 0, -e.disk_gb))
    return max(entries, key=lambda e: tokens_per_second(e, hw))


# --- fitting --------------------------------------------------------------------


def _fits(entry: ModelEntry, hw: HardwareProfile, installed: set[str] | None = None) -> bool:
    if hw.ram_gb < entry.min_ram_gb:
        return False
    # Free disk only matters for a download. An installed model needs none --
    # counting its size again would reject it right after it was pulled
    # (seen on a 16 GB Mac left with 5.8 GB free by a 9 GB model).
    already_on_disk = installed is not None and entry.name in installed
    if not already_on_disk and hw.free_disk_gb < entry.disk_gb:
        return False
    if entry.runtime != "ollama":
        # Image/video runtimes aren't sized by running_gb(); keep the plain minimums.
        return entry.min_vram_gb == 0 or hw.total_vram_gb >= entry.min_vram_gb
    return _placement(entry, hw) is not None


def candidates(
    modality: str,
    hardware: HardwareProfile,
    catalog: list[ModelEntry] | None = None,
    installed: set[str] | None = None,
) -> list[ModelEntry]:
    """All catalog models for `modality` that fit `hardware` (RAM/VRAM, and
    free disk unless the model is already in `installed`).
    """
    catalog = catalog if catalog is not None else load_catalog()
    return [m for m in catalog if m.modality == modality and _fits(m, hardware, installed)]


def best_match(
    modality: str,
    hardware: HardwareProfile,
    catalog: list[ModelEntry] | None = None,
    installed: set[str] | None = None,
) -> ModelEntry:
    """Return the best model for `modality` that fits `hardware`.

    `installed`, if given, is the set of exact Ollama tag names already
    present on disk (see `OllamaBackend.list_installed()`). If any
    already-installed model fits, it's preferred over the theoretically
    "best" catalog entry -- reusing what's already there needs no download
    at all, whereas the highest quality_tier pick might trigger a
    multi-GB pull for a marginal quality difference. Falls back to the
    highest quality_tier fitting entry, installed or not, exactly as
    before, when nothing installed fits (or `installed` isn't given).
    """
    fitting = candidates(modality, hardware, catalog, installed)
    if not fitting:
        raise NoFittingModelError(
            f"No catalog model for modality={modality!r} fits this machine "
            f"(RAM={hardware.ram_gb}GB, VRAM={hardware.total_vram_gb}GB, "
            f"free disk={hardware.free_disk_gb}GB). Try a smaller quality tier, "
            "free up disk space, or add more hardware."
        )
    if installed:
        already_have = [m for m in fitting if m.name in installed]
        if already_have:
            return max(already_have, key=lambda m: m.quality_tier)
    return balanced_pick(fitting, hardware)


def recommendations(
    hardware: HardwareProfile,
    catalog: list[ModelEntry] | None = None,
    installed: set[str] | None = None,
) -> dict[str, ModelEntry | None]:
    """Best-fit model per modality present in the catalog, or None if nothing fits.
    See `best_match()` for what `installed` does.
    """
    catalog = catalog if catalog is not None else load_catalog()
    modalities = {m.modality for m in catalog}
    result: dict[str, ModelEntry | None] = {}
    for modality in sorted(modalities):
        try:
            result[modality] = best_match(modality, hardware, catalog, installed=installed)
        except NoFittingModelError:
            result[modality] = None
    return share_text_model(result, hardware, installed)


TEXT_MODALITIES = ("coding", "docs", "general")


def share_text_model(
    picks: dict[str, ModelEntry | None], hw: HardwareProfile, installed: set[str] | None = None
) -> dict[str, ModelEntry | None]:
    """On a machine that can't hold two of the picked text models at once,
    use one model for all text work: the coding pick (a coder writes docs
    and summaries well enough). Otherwise every switch between coding and
    docs unloads one model and reloads the other, and the downloads double
    for little gain. Picks already on disk are left alone."""
    coder = picks.get("coding")
    if coder is None:
        return picks
    budget = resident_capacity_gb(hw)
    shared = dict(picks)
    for modality in TEXT_MODALITIES:
        other = picks.get(modality)
        if modality == "coding" or other is None or other.name == coder.name or other.name in (installed or ()):
            continue
        if running_gb(coder) + running_gb(other) > budget:
            shared[modality] = coder
    return shared
