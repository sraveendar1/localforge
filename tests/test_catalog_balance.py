"""Model selection that keeps the machine usable (asked for: "the right
balance, and keep system performance in mind").

The old rule was "RAM >= a minimum", which picked a 9 GB 14b coder *and* a
7 GB docs model for a 16 GB Mac: together, plus their context caches, more
than the machine has, so it swapped. Now a model fits only if it can run
beside the OS and the user's apps, gets the largest context window that
allows, and among those the pick is the best that still runs comfortably.
"""

from unittest.mock import patch

import pytest

from localforge.backends import ollama
from localforge.catalog import (
    MIN_TOKENS_PER_SECOND,
    context_limit,
    load_catalog,
    recommendations,
    running_gb,
    tokens_per_second,
)
from localforge.hardware import GPU, HardwareProfile, apple_bandwidth
from localforge.tools import Dispatcher


def _mac(ram, chip, bandwidth):
    return HardwareProfile(
        os="Darwin", arch="arm64", cpu_cores=10, ram_gb=ram, free_disk_gb=100,
        gpus=[GPU(name=f"Apple {chip}", vram_gb=ram * 0.75, backend="metal")], memory_bandwidth_gbps=bandwidth,
    )


M4_16 = _mac(16, "M4", 120)  # the reporter's machine
M1_8 = _mac(8, "M1", 68)
M4_PRO_48 = _mac(48, "M4 Pro", 273)
CPU_16 = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=16, free_disk_gb=200, gpus=[])
RTX_4090 = HardwareProfile(
    os="Linux", arch="x86_64", cpu_cores=16, ram_gb=64, free_disk_gb=500,
    gpus=[GPU(name="RTX 4090", vram_gb=24, backend="cuda")], memory_bandwidth_gbps=1000,
)


def _picks(hw, installed=None):
    return {m: e.name for m, e in recommendations(hw, load_catalog(), installed=installed).items() if e and m in ("coding", "docs", "general")}


def test_a_16gb_mac_gets_one_model_that_leaves_it_usable():
    picks = _picks(M4_16)
    assert picks == {"coding": "qwen2.5-coder:7b", "docs": "qwen2.5-coder:7b", "general": "qwen2.5-coder:7b"}
    entry = next(e for e in load_catalog() if e.name == "qwen2.5-coder:7b")
    assert context_limit(entry, M4_16) == 32768  # and it gets the full context window
    assert tokens_per_second(entry, M4_16) >= MIN_TOKENS_PER_SECOND


def test_the_14b_no_longer_counts_as_fitting_a_16gb_mac():
    catalog = load_catalog()
    coder14 = next(e for e in catalog if e.name == "qwen2.5-coder:14b")
    assert context_limit(coder14, M4_16) is None  # even at 8k it leaves the system too little


def test_a_model_that_only_fits_with_a_small_window_gets_that_window():
    nemo = next(e for e in load_catalog() if e.name == "mistral-nemo:12b")
    assert context_limit(nemo, M4_16) == 8192  # runs, with 8k, instead of being ruled out or swapping at 32k
    assert running_gb(nemo, 32768) > 10.8


def test_an_8gb_mac_gets_a_small_model():
    assert set(_picks(M1_8).values()) == {"qwen2.5-coder:3b"}


def test_a_big_mac_gets_the_mixture_of_experts_coder_and_separate_models():
    picks = _picks(M4_PRO_48)
    assert picks["coding"] == "qwen3-coder:30b"
    assert picks["docs"] != picks["coding"]  # room for more than one model at once


def test_a_cpu_only_machine_gets_models_fast_enough_to_use():
    for name in _picks(CPU_16).values():
        entry = next(e for e in load_catalog() if e.name == name)
        assert tokens_per_second(entry, CPU_16) >= MIN_TOKENS_PER_SECOND


def test_a_discrete_gpu_with_plenty_of_ram_keeps_separate_models():
    picks = _picks(RTX_4090)
    assert len(set(picks.values())) > 1  # VRAM plus spare RAM holds more than one


def test_installed_models_are_never_swapped_for_the_shared_one():
    installed = {"qwen2.5-coder:7b", "llama3.1:8b"}
    picks = _picks(M4_16, installed)
    assert picks["docs"] == "llama3.1:8b"  # already on disk: reused, not replaced


def test_the_catalog_has_no_cloud_only_or_thinking_models():
    names = {e.name for e in load_catalog()}
    assert "kimi-k2:latest" not in names  # 0 bytes to download: cloud-only in Ollama
    assert not any(n.startswith(("qwen3:", "qwen3.5:", "gpt-oss:")) for n in names)
    text = [e for e in load_catalog() if e.runtime == "ollama"]
    assert all(e.kv_gb_per_8k > 0 for e in text)  # every text model's context cost is known


@pytest.mark.parametrize(
    "brand, expected",
    [("Apple M4", 120), ("Apple M4 Pro", 273), ("Apple M1 Max", 400), ("Apple M3 Ultra", 819), ("Apple M9", None), ("Intel", None)],
)
def test_apple_bandwidth(brand, expected):
    assert apple_bandwidth(brand) == expected


# --- the window reaches Ollama -------------------------------------------------


def test_the_dispatcher_gives_each_model_its_own_window():
    calls = []

    class Stub:
        def ensure_available(self, name, on_progress=None):
            pass

        def generate(self, name, prompt, on_token=None, **kw):
            calls.append(kw)
            return {"type": "text", "content": "fine output " * 5, "tokens": 5}

    nemo = next(e for e in load_catalog() if e.name == "mistral-nemo:12b")
    d = Dispatcher(M4_16, catalog=[nemo.model_copy(update={"runtime": "stub"})])
    with patch.dict("localforge.tools.BACKENDS", {"stub": Stub()}):
        d.dispatch("delegate_docs_task", {"instructions": "write a changelog entry"})
    assert calls[0]["context_limit"] == 8192
    assert d.prompt_chars("docs") == (8192 - 4096) * ollama.CHARS_PER_TOKEN  # half kept for the reply


def test_a_small_window_still_leaves_room_for_the_reply():
    options = ollama._with_context("x" * 9000, ollama.MAX_OUTPUT_TOKENS, {"context_limit": 8192})
    assert options["num_ctx"] == 8192
    assert options["num_predict"] <= 8192 - ollama.estimate_tokens("x" * 9000)


def test_setup_plan_lists_a_shared_model_once():
    from rich.console import Console

    import localforge.cli as cli_module

    recs = recommendations(M4_16, load_catalog())
    console = Console(record=True, width=200)
    with patch.object(cli_module, "console", console):
        cli_module._print_model_plan(recs, set(), M4_16)
    text = console.export_text()
    assert text.count("qwen2.5-coder:7b") == 1
    assert "qwen2.5-coder:7b (coding, docs, general, ~4.7 GB)" in text
