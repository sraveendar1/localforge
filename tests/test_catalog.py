from localforge.catalog import ModelEntry, NoFittingModelError, best_match, recommendations
from localforge.hardware import GPU, HardwareProfile

CATALOG = [
    ModelEntry(name="small-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, quality_tier=1),
    ModelEntry(name="big-coder", modality="coding", runtime="ollama", min_vram_gb=20, min_ram_gb=32, quality_tier=3),
    ModelEntry(name="mid-coder", modality="coding", runtime="ollama", min_vram_gb=8, min_ram_gb=16, quality_tier=2),
    ModelEntry(name="only-docs", modality="docs", runtime="ollama", min_vram_gb=6, min_ram_gb=8, quality_tier=1),
]


def _hw(ram_gb: float, vram_gb: float = 0) -> HardwareProfile:
    gpus = [GPU(name="test-gpu", vram_gb=vram_gb, backend="cuda")] if vram_gb else []
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=ram_gb, gpus=gpus)


def test_best_match_picks_highest_tier_that_fits():
    hw = _hw(ram_gb=32, vram_gb=24)
    assert best_match("coding", hw, CATALOG).name == "big-coder"


def test_best_match_falls_back_to_lower_tier_on_weak_hardware():
    hw = _hw(ram_gb=16, vram_gb=8)
    assert best_match("coding", hw, CATALOG).name == "mid-coder"


def test_best_match_cpu_only_still_gets_a_model():
    hw = _hw(ram_gb=8, vram_gb=0)
    assert best_match("coding", hw, CATALOG).name == "small-coder"


def test_best_match_raises_when_nothing_fits():
    hw = _hw(ram_gb=2, vram_gb=0)
    try:
        best_match("coding", hw, CATALOG)
        assert False, "expected NoFittingModelError"
    except NoFittingModelError:
        pass


def test_recommendations_covers_every_modality():
    hw = _hw(ram_gb=32, vram_gb=24)
    recs = recommendations(hw, CATALOG)
    assert recs["coding"].name == "big-coder"
    assert recs["docs"].name == "only-docs"
