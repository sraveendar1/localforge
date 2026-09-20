from unittest.mock import patch

from localforge.catalog import ModelEntry
from localforge.hardware import GPU, HardwareProfile
from localforge.tools import TASK_MODALITIES, Dispatcher

CATALOG = [
    ModelEntry(name="small-coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=8, disk_gb=2, quality_tier=1),
]


class StubBackend:
    def ensure_available(self, model_name, on_progress=None):
        pass

    def generate(self, model_name, prompt, **kwargs):
        return {"type": "text", "content": f"handled by {model_name}", "tokens": 42}


def _hw() -> HardwareProfile:
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def test_dispatch_calls_on_delegate_with_modality_and_resolved_entry():
    dispatcher = Dispatcher(_hw(), catalog=CATALOG)
    calls = []

    with patch.dict("localforge.tools.BACKENDS", {"stub": StubBackend()}):
        result = dispatcher.dispatch(
            TASK_MODALITIES["coding"]["tool_name"],
            "do the thing",
            on_delegate=lambda modality, entry: calls.append((modality, entry.name)),
        )

    assert calls == [("coding", "small-coder")]
    assert result == "handled by small-coder"


def test_dispatch_works_without_on_delegate():
    dispatcher = Dispatcher(_hw(), catalog=CATALOG)
    with patch.dict("localforge.tools.BACKENDS", {"stub": StubBackend()}):
        result = dispatcher.dispatch(TASK_MODALITIES["coding"]["tool_name"], "do the thing")
    assert result == "handled by small-coder"


def test_dispatch_accumulates_local_tokens_across_calls():
    dispatcher = Dispatcher(_hw(), catalog=CATALOG)
    with patch.dict("localforge.tools.BACKENDS", {"stub": StubBackend()}):
        dispatcher.dispatch(TASK_MODALITIES["coding"]["tool_name"], "first")
        dispatcher.dispatch(TASK_MODALITIES["coding"]["tool_name"], "second")
    assert dispatcher.local_tokens_generated == 84  # 42 tokens per call, twice
