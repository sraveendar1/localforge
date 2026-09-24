from localforge.catalog import ModelEntry, NoFittingModelError, best_match, candidates, recommendations
from localforge.hardware import GPU, HardwareProfile

CATALOG = [
    ModelEntry(name="small-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=2, quality_tier=1),
    ModelEntry(name="big-coder", modality="coding", runtime="ollama", min_vram_gb=20, min_ram_gb=32, disk_gb=20, quality_tier=3),
    ModelEntry(name="mid-coder", modality="coding", runtime="ollama", min_vram_gb=8, min_ram_gb=16, disk_gb=9, quality_tier=2),
    ModelEntry(name="only-docs", modality="docs", runtime="ollama", min_vram_gb=6, min_ram_gb=8, disk_gb=5, quality_tier=1),
]


def _hw(ram_gb: float, vram_gb: float = 0, free_disk_gb: float = 100) -> HardwareProfile:
    gpus = [GPU(name="test-gpu", vram_gb=vram_gb, backend="cuda")] if vram_gb else []
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=8, ram_gb=ram_gb, free_disk_gb=free_disk_gb, gpus=gpus
    )


def _big_gpu() -> HardwareProfile:
    """A machine where even the 20 GB model fits on the GPU and runs fast."""
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=16, ram_gb=64, free_disk_gb=100,
        gpus=[GPU(name="big-gpu", vram_gb=48, backend="cuda")], memory_bandwidth_gbps=1000,
    )


def test_best_match_picks_highest_tier_that_fits():
    assert best_match("coding", _big_gpu(), CATALOG).name == "big-coder"


def test_best_match_falls_back_to_lower_tier_on_weak_hardware():
    # mid-coder (9 GB) doesn't fit an 8 GB GPU; from system RAM it would crawl
    # (~4 tok/s), so the balanced pick is the small model that runs on the GPU.
    hw = _hw(ram_gb=16, vram_gb=8)
    assert best_match("coding", hw, CATALOG).name == "small-coder"


def test_best_match_cpu_only_still_gets_a_model():
    hw = _hw(ram_gb=8, vram_gb=0)
    assert best_match("coding", hw, CATALOG).name == "small-coder"


def test_best_match_raises_when_nothing_fits():
    hw = _hw(ram_gb=2, vram_gb=0)
    try:
        best_match("coding", hw, CATALOG)
        raise AssertionError("expected NoFittingModelError")
    except NoFittingModelError:
        pass


def test_best_match_excludes_models_with_insufficient_disk():
    hw = _big_gpu().model_copy(update={"free_disk_gb": 10})  # too little disk for big-coder (20GB)
    assert best_match("coding", hw, CATALOG).name == "mid-coder"


def test_candidates_returns_all_fitting_models_not_just_the_best():
    hw = _big_gpu()
    names = {m.name for m in candidates("coding", hw, CATALOG)}
    assert names == {"small-coder", "mid-coder", "big-coder"}


def test_recommendations_covers_every_modality():
    hw = _big_gpu()
    recs = recommendations(hw, CATALOG)
    assert recs["coding"].name == "big-coder"
    assert recs["docs"].name == "only-docs"


def test_recommendations_never_includes_internal_modalities():
    """The judge model (tools.Dispatcher.judge()) is resolved directly, not
    through recommendations() -- setup's plan+pull loop and /upgrade both
    build their to-do list straight from this dict, so a modality left in
    here would get silently offered and auto-downloaded, breaking "never
    downloaded automatically" for the judge feature."""
    catalog_with_judge = CATALOG + [
        ModelEntry(name="judge-model", modality="judge", runtime="ollama", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)
    ]
    recs = recommendations(_big_gpu(), catalog_with_judge)
    assert "judge" not in recs
