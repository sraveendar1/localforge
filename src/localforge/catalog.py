"""Static rules + curated catalog: picks the best-fitting local model for a
task's modality given the detected hardware. See catalog_data.yaml for the
model list.
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


class NoFittingModelError(RuntimeError):
    """Raised when no catalog entry for a modality fits the detected hardware."""


def load_catalog() -> list[ModelEntry]:
    data = yaml.safe_load(resources.files("localforge").joinpath("catalog_data.yaml").read_text())
    return [ModelEntry(**m) for m in data["models"]]


def _fits(entry: ModelEntry, hw: HardwareProfile) -> bool:
    if hw.ram_gb < entry.min_ram_gb:
        return False
    if hw.free_disk_gb < entry.disk_gb:
        return False
    if entry.min_vram_gb == 0:
        return True  # CPU-runnable
    return hw.total_vram_gb >= entry.min_vram_gb


def candidates(modality: str, hardware: HardwareProfile, catalog: list[ModelEntry] | None = None) -> list[ModelEntry]:
    """All catalog models for `modality` that fit `hardware` (RAM/VRAM/disk)."""
    catalog = catalog if catalog is not None else load_catalog()
    return [m for m in catalog if m.modality == modality and _fits(m, hardware)]


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
    fitting = candidates(modality, hardware, catalog)
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
    return max(fitting, key=lambda m: m.quality_tier)


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
    return result
