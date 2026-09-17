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
    quality_tier: int


class NoFittingModelError(RuntimeError):
    """Raised when no catalog entry for a modality fits the detected hardware."""


def load_catalog() -> list[ModelEntry]:
    data = yaml.safe_load(resources.files("localforge").joinpath("catalog_data.yaml").read_text())
    return [ModelEntry(**m) for m in data["models"]]


def _fits(entry: ModelEntry, hw: HardwareProfile) -> bool:
    if hw.ram_gb < entry.min_ram_gb:
        return False
    if entry.min_vram_gb == 0:
        return True  # CPU-runnable
    return hw.total_vram_gb >= entry.min_vram_gb


def best_match(modality: str, hardware: HardwareProfile, catalog: list[ModelEntry] | None = None) -> ModelEntry:
    """Return the highest quality-tier model for `modality` that fits `hardware`."""
    catalog = catalog if catalog is not None else load_catalog()
    candidates = [m for m in catalog if m.modality == modality and _fits(m, hardware)]
    if not candidates:
        raise NoFittingModelError(
            f"No catalog model for modality={modality!r} fits this machine "
            f"(RAM={hardware.ram_gb}GB, VRAM={hardware.total_vram_gb}GB). "
            "Try a smaller quality tier or add more hardware."
        )
    return max(candidates, key=lambda m: m.quality_tier)


def recommendations(hardware: HardwareProfile, catalog: list[ModelEntry] | None = None) -> dict[str, ModelEntry | None]:
    """Best-fit model per modality present in the catalog, or None if nothing fits."""
    catalog = catalog if catalog is not None else load_catalog()
    modalities = {m.modality for m in catalog}
    result: dict[str, ModelEntry | None] = {}
    for modality in sorted(modalities):
        try:
            result[modality] = best_match(modality, hardware, catalog)
        except NoFittingModelError:
            result[modality] = None
    return result
