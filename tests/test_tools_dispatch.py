from unittest.mock import patch

from localforge.catalog import ModelEntry
from localforge.hardware import GPU, HardwareProfile
from localforge.tools import TASK_MODALITIES, Dispatcher, _looks_suspect

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


TWO_TIER_CATALOG = [
    ModelEntry(name="small-coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=8, disk_gb=2, quality_tier=1),
    ModelEntry(name="big-coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=8, disk_gb=2, quality_tier=2),
]


class PerModelBackend:
    """Returns different (possibly bad) output depending on which model was
    asked, and reports a fixed token count per call regardless.
    """

    def __init__(self, responses: dict[str, str], tokens_per_call: int = 10):
        self.responses = responses
        self.tokens_per_call = tokens_per_call
        self.calls: list[str] = []

    def ensure_available(self, model_name, on_progress=None):
        pass

    def generate(self, model_name, prompt, **kwargs):
        self.calls.append(model_name)
        return {"type": "text", "content": self.responses[model_name], "tokens": self.tokens_per_call}


def test_dispatch_retries_a_different_model_on_empty_output_and_returns_the_good_result():
    # resolve() always picks the highest quality_tier fitting model first, so
    # the first attempt is "big-coder"; the retry falls back to the only
    # alternative, "small-coder".
    backend = PerModelBackend({"big-coder": "", "small-coder": "def solved(): return 42"})
    dispatcher = Dispatcher(_hw(), catalog=TWO_TIER_CATALOG)
    delegated_to = []

    with patch.dict("localforge.tools.BACKENDS", {"stub": backend}):
        result = dispatcher.dispatch(
            TASK_MODALITIES["coding"]["tool_name"],
            "x" * 100,  # long enough that a short/empty result is flagged
            on_delegate=lambda modality, entry: delegated_to.append(entry.name),
        )

    assert result == "def solved(): return 42"
    assert backend.calls == ["big-coder", "small-coder"]
    assert delegated_to == ["big-coder", "small-coder"]  # user sees both attempts
    assert not result.startswith("[WARNING:")
    # both attempts cost local compute -- both must count toward the metric
    assert dispatcher.local_tokens_generated == 20


def test_dispatch_flags_result_with_warning_when_no_alternative_model_exists():
    backend = PerModelBackend({"small-coder": ""})
    dispatcher = Dispatcher(_hw(), catalog=CATALOG)  # only one model available

    with patch.dict("localforge.tools.BACKENDS", {"stub": backend}):
        result = dispatcher.dispatch(TASK_MODALITIES["coding"]["tool_name"], "x" * 100)

    assert result.startswith("[WARNING: this coding result may be unreliable (empty output)")
    # no alternative model -- retries the *same* model once, with reinforced instructions
    assert backend.calls == ["small-coder", "small-coder"]


def test_dispatch_flags_result_when_retry_is_also_bad():
    backend = PerModelBackend({"big-coder": "", "small-coder": "I cannot help with that."})
    dispatcher = Dispatcher(_hw(), catalog=TWO_TIER_CATALOG)

    with patch.dict("localforge.tools.BACKENDS", {"stub": backend}):
        result = dispatcher.dispatch(TASK_MODALITIES["coding"]["tool_name"], "x" * 100)

    assert backend.calls == ["big-coder", "small-coder"]
    assert result.startswith("[WARNING: this coding result may be unreliable (looks like a refusal")
    assert "I cannot help with that." in result


def test_dispatch_does_not_retry_for_a_normal_result():
    backend = PerModelBackend({"big-coder": "def f(): return 1", "small-coder": "should never be called"})
    dispatcher = Dispatcher(_hw(), catalog=TWO_TIER_CATALOG)

    with patch.dict("localforge.tools.BACKENDS", {"stub": backend}):
        result = dispatcher.dispatch(TASK_MODALITIES["coding"]["tool_name"], "write a function")

    assert result == "def f(): return 1"
    assert backend.calls == ["big-coder"]  # never retried -- the first result was fine


def test_looks_suspect_flags_empty_output():
    assert _looks_suspect("   ", "do something") == "empty output"


def test_looks_suspect_flags_refusal_phrases():
    assert _looks_suspect("I'm sorry, but I can't do that.", "do something") is not None
    assert _looks_suspect("As an AI language model, I cannot comply.", "do something") is not None


def test_looks_suspect_flags_short_output_for_a_long_request():
    assert _looks_suspect("ok", "x" * 100) == "suspiciously short for the size of the request"


def test_looks_suspect_allows_short_output_for_a_short_request():
    assert _looks_suspect("ok", "short request") is None


def test_looks_suspect_passes_normal_output():
    assert _looks_suspect("def add(a, b):\n    return a + b\n", "write an add function") is None
