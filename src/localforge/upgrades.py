"""Keeping installed local models current (asked for: "when localforge
already has open-weight models installed, it doesn't check if there's a
better one ... replace it, like an upgrade, and remove the older ones to
free space").

Installed models win over better ones at run time on purpose -- a task
should never stop for a surprise multi-GB download. So upgrading is its own
step: `plan()` compares what's on disk with the balanced pick for this
machine (catalog.recommendations, which already weighs memory, speed and
free disk), and an upgrade is a model that is a strictly higher tier and
fits. The new model is downloaded first; the old one is removed only after
that, only when nothing uses it any more, and only if localforge installed
it (managed_models.json). A model the user pulled themselves is never
removed without them saying so in /upgrade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from localforge import config
from localforge.catalog import ModelEntry, load_catalog, recommendations
from localforge.hardware import HardwareProfile

TEXT_MODALITIES = ("coding", "docs", "general")
# Free disk kept after a download, so an upgrade never fills the disk.
DISK_MARGIN_GB = 3.0
# How the user wants upgrades handled: "always" (in the background, with a
# one-line note), "never", or unset (ask the first time one is found).
AUTO_UPGRADE_ENV_VAR = "LOCALFORGE_AUTO_UPGRADE"


def _managed_file():
    return config.CONFIG_DIR / "managed_models.json"


# --- which models localforge installed ------------------------------------------


def managed() -> set[str]:
    try:
        data = json.loads(_managed_file().read_text())
        return {str(name) for name in data} if isinstance(data, list) else set()
    except (OSError, ValueError):
        return set()


def _save(names: set[str]) -> None:
    path = _managed_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(names), indent=1) + "\n")


def mark_managed(name: str) -> None:
    """Record that localforge downloaded `name` (so it may remove it later)."""
    names = managed()
    if name not in names:
        _save(names | {name})


def unmark(name: str) -> None:
    names = managed()
    if name in names:
        _save(names - {name})


# --- the plan -------------------------------------------------------------------


@dataclass
class Upgrade:
    old: str
    new: ModelEntry
    modalities: list[str]


@dataclass
class Plan:
    upgrades: list[Upgrade] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)  # installed by localforge, no longer used
    unused_own: list[str] = field(default_factory=list)  # the user's own, no longer used: only mentioned
    download_gb: float = 0.0
    freed_gb: float = 0.0
    blocked: str | None = None  # why the upgrades can't happen now (disk space)

    @property
    def empty(self) -> bool:
        return not self.upgrades and not self.remove

    def summary(self) -> list[str]:
        lines = []
        for u in self.upgrades:
            lines.append(f"{', '.join(u.modalities)}: {u.old} → {u.new.name} (~{u.new.disk_gb:g} GB download)")
        for name in self.remove:
            lines.append(f"remove {name} (no longer used)")
        return lines


def plan(
    hw: HardwareProfile,
    installed: dict[str, int],
    catalog: list[ModelEntry] | None = None,
    keep: set[str] | frozenset[str] = frozenset(),
    managed_names: set[str] | None = None,
) -> Plan:
    """What upgrading would do here. `installed` maps Ollama tags on disk to
    their size in bytes; `keep` names models that must stay (the local
    orchestrator); `managed_names` defaults to what localforge installed."""
    catalog = catalog if catalog is not None else load_catalog()
    managed_names = managed() if managed_names is None else managed_names
    on_disk = set(installed)
    current = recommendations(hw, catalog, installed=on_disk)
    ideal = recommendations(hw, catalog)  # ignoring what's installed: what we'd pick fresh

    result = Plan()
    by_new: dict[str, Upgrade] = {}
    final: dict[str, str] = {}
    for modality in TEXT_MODALITIES:
        cur, best = current.get(modality), ideal.get(modality)
        if cur is None:
            continue
        final[modality] = cur.name
        old = cur.name
        if cur.name not in on_disk:
            # Nothing installed fits this any more (a 14b kept from before the
            # memory-aware rules, on a 16 GB Mac): replacing it is an upgrade too.
            # With nothing installed at all, it's setup's job instead.
            misfits = [e.name for e in catalog if e.modality == modality and e.name in on_disk]
            if not misfits:
                continue
            old, best = misfits[0], cur
        if best is not None and best.name != old and (best is cur or best.quality_tier > cur.quality_tier) and best.name not in on_disk:
            upgrade = by_new.setdefault(best.name, Upgrade(old=old, new=best, modalities=[]))
            upgrade.modalities.append(modality)
            final[modality] = best.name

    result.upgrades = list(by_new.values())
    result.download_gb = sum(u.new.disk_gb for u in result.upgrades)
    if result.upgrades and hw.free_disk_gb < result.download_gb + DISK_MARGIN_GB:
        result.blocked = (
            f"needs {result.download_gb:g} GB free (plus {DISK_MARGIN_GB:g} GB to spare) and this disk has "
            f"{hw.free_disk_gb:g} GB"
        )
        result.upgrades, result.download_gb = [], 0.0
        final = {m: e.name for m, e in current.items() if e is not None and m in TEXT_MODALITIES}

    used = set(final.values()) | set(keep)
    catalog_names = {e.name for e in catalog if e.runtime == "ollama"}
    for name in sorted(on_disk & catalog_names - used):
        (result.remove if name in managed_names else result.unused_own).append(name)
    result.freed_gb = round(sum(installed.get(n, 0) for n in result.remove) / 1e9, 1)
    return result


def local_orchestrator() -> set[str]:
    """The saved local orchestrator's Ollama tag, if there is one: never removed."""
    import os

    model = os.environ.get(config.FRONTIER_MODEL_ENV_VAR) or ""
    for prefix in ("ollama/", "ollama_chat/"):
        if model.startswith(prefix):
            return {model[len(prefix) :]}
    return set()
