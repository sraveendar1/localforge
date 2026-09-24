from localforge.advisor import recommend_models
from localforge.catalog import ModelEntry
from localforge.hardware import GPU, HardwareProfile

CATALOG = [
    ModelEntry(name="small-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=8, disk_gb=2, quality_tier=1),
    ModelEntry(name="big-coder", modality="coding", runtime="ollama", min_vram_gb=20, min_ram_gb=32, disk_gb=20, quality_tier=3),
    ModelEntry(name="only-docs", modality="docs", runtime="ollama", min_vram_gb=6, min_ram_gb=8, disk_gb=5, quality_tier=1),
]


def _hw(ram_gb: float, vram_gb: float = 0, free_disk_gb: float = 100) -> HardwareProfile:
    gpus = [GPU(name="test-gpu", vram_gb=vram_gb, backend="cuda")] if vram_gb else []
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=8, ram_gb=ram_gb, free_disk_gb=free_disk_gb, gpus=gpus
    )


def test_recommend_models_falls_back_to_best_quality_when_frontier_call_fails():
    # No real API key/network in tests, so the litellm.completion() call inside
    # recommend_models is expected to fail -- verifying the safety-net fallback
    # (deterministic highest quality_tier per modality) kicks in instead of raising.
    hw = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=16, ram_gb=64, free_disk_gb=100,
        gpus=[GPU(name="big-gpu", vram_gb=48, backend="cuda")], memory_bandwidth_gbps=1000,
    )
    recs = recommend_models(hw, "claude-opus-5", CATALOG)
    assert recs["coding"].name == "big-coder"
    assert recs["docs"].name == "only-docs"


def test_recommend_models_never_returns_a_non_fitting_model():
    hw = _hw(ram_gb=16, vram_gb=8)  # too little for big-coder
    recs = recommend_models(hw, "claude-opus-5", CATALOG)
    assert recs["coding"].name == "small-coder"


def test_recommend_models_returns_none_for_modality_nothing_fits():
    hw = _hw(ram_gb=2, vram_gb=0, free_disk_gb=1)
    recs = recommend_models(hw, "claude-opus-5", CATALOG)
    assert recs["coding"] is None
    assert recs["docs"] is None


def test_recommend_models_never_asks_the_frontier_model_about_the_judge():
    """There's exactly one judge candidate anyway, so asking the frontier
    model to "pick" would just spend a tool-call round trip on setup
    deciding nothing -- and it should never be offered for download here
    regardless (see catalog.INTERNAL_MODALITIES)."""
    catalog_with_judge = CATALOG + [
        ModelEntry(name="judge-model", modality="judge", runtime="ollama", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)
    ]
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=16, ram_gb=64, free_disk_gb=100, gpus=[])
    recs = recommend_models(hw, "claude-opus-5", catalog_with_judge)
    assert "judge" not in recs
