"""Detects local hardware so the catalog can pick models that will actually run."""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

import psutil
from pydantic import BaseModel


class GPU(BaseModel):
    name: str
    vram_gb: float
    backend: str  # "cuda", "metal", "rocm"


class HardwareProfile(BaseModel):
    os: str
    arch: str
    cpu_cores: int
    ram_gb: float
    free_disk_gb: float
    gpus: list[GPU]
    # GB/s the GPU (or, without one, the CPU) reads memory at: generation
    # speed is bound by it. None = unknown; catalog.py then assumes a typical
    # figure for the kind of machine.
    memory_bandwidth_gbps: float | None = None

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0


def _detect_nvidia_gpus() -> list[GPU]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return []

    gpus = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, mem_mib = (part.strip() for part in line.split(","))
        gpus.append(GPU(name=name, vram_gb=round(float(mem_mib) / 1024, 1), backend="cuda"))
    return gpus


def _detect_apple_silicon_gpu(ram_gb: float) -> list[GPU]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        brand = "Apple Silicon"
    # Apple Silicon uses unified memory: the GPU can address (most of) system RAM.
    return [GPU(name=brand or "Apple Silicon GPU", vram_gb=round(ram_gb * 0.75, 1), backend="metal")]


# Apple's published unified-memory bandwidth per chip (GB/s). Binned Max
# chips are slower; the lower figure is used so estimates err on the slow side.
_APPLE_BANDWIDTH = {
    "M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 300, "M3 Ultra": 819,
    "M4": 120, "M4 Pro": 273, "M4 Max": 410,
    "M5": 153,
}


def apple_bandwidth(brand: str) -> float | None:
    """e.g. "Apple M4 Pro" -> 273. None for a chip not in the table."""
    import re

    match = re.search(r"\b(M\d+)(?:\s+(Pro|Max|Ultra))?\b", brand)
    if not match:
        return None
    chip = match.group(1) + (f" {match.group(2)}" if match.group(2) else "")
    return _APPLE_BANDWIDTH.get(chip) or _APPLE_BANDWIDTH.get(match.group(1))


def _detect_free_disk_gb() -> float:
    # Where Ollama/model downloads actually land, not just wherever the CLI runs.
    usage = shutil.disk_usage(Path.home())
    return round(usage.free / (1024**3), 1)


def detect_hardware() -> HardwareProfile:
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    gpus = _detect_nvidia_gpus() or _detect_apple_silicon_gpu(ram_gb)
    bandwidth = apple_bandwidth(gpus[0].name) if gpus and gpus[0].backend == "metal" else None
    return HardwareProfile(
        os=platform.system(),
        arch=platform.machine(),
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1,
        ram_gb=ram_gb,
        free_disk_gb=_detect_free_disk_gb(),
        gpus=gpus,
        memory_bandwidth_gbps=bandwidth,
    )
