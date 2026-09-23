"""Regression tests for edge cases found in a review pass over the codebase."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch_module
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.orchestrator import OrchestrationError, RunStats
from localforge.tools import build_tool_schemas


def _hw(ram_gb: float = 32, free_disk_gb: float = 100) -> HardwareProfile:
    return HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=8, ram_gb=ram_gb, free_disk_gb=free_disk_gb, gpus=[]
    )


# --- Bug 1: non-convergence used to discard all accumulated usage stats ---


class _StubDispatcher:
    def __init__(self, hardware, catalog=None, **kwargs):
        self.catalog = catalog or []
        self.local_tokens_generated = 999

    def dispatch(self, tool_name, instructions, on_delegate=None):
        return "some result"


def _tool_call_response():
    msg = MagicMock()
    msg.tool_calls = [MagicMock(id="call_1", function=MagicMock())]
    msg.tool_calls[0].function.name = "delegate_coding_task"
    msg.tool_calls[0].function.arguments = '{"instructions": "do it"}'
    msg.model_dump.return_value = {"role": "assistant", "content": None}
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    usage = MagicMock()
    usage.prompt_tokens, usage.completion_tokens = 10, 5
    resp.usage = usage
    return resp


def test_non_convergence_raises_error_carrying_the_usage_spent_so_far():
    """A run that never converges still burned real tokens/cost -- those must
    not be silently thrown away with the exception.
    """
    with (
        patch.object(orch_module, "completion", side_effect=lambda **kw: _tool_call_response()),
        patch.object(orch_module, "Dispatcher", _StubDispatcher),
        patch.object(orch_module, "build_tool_schemas", return_value=[]),
        patch.object(orch_module.litellm, "completion_cost", return_value=0.001),
        patch.object(orch_module, "CHECKPOINT_EVERY", orch_module.MAX_ROUNDS),  # run to the ceiling
    ):
        with pytest.raises(OrchestrationError) as excinfo:
            orch_module.run("task", "claude-opus-5", hardware=_hw())

    stats = excinfo.value.stats
    assert stats.frontier_prompt_tokens == 10 * orch_module.MAX_ROUNDS
    assert stats.frontier_completion_tokens == 5 * orch_module.MAX_ROUNDS
    assert stats.frontier_cost_usd == pytest.approx(0.001 * orch_module.MAX_ROUNDS)
    assert stats.local_tokens_generated == 999  # carried over from the dispatcher


def test_run_usage_flag_still_prints_usage_when_orchestration_does_not_converge():
    stats = RunStats(
        frontier_prompt_tokens=100, frontier_completion_tokens=50, frontier_cost_usd=0.02, local_tokens_generated=700
    )
    with patch.object(
        cli_module, "run_orchestrator", side_effect=OrchestrationError("did not converge", stats)
    ):
        result = CliRunner().invoke(cli_module.app, ["run", "task", "--model", "claude-opus-5", "--usage"])

    assert result.exit_code == 1
    assert "did not converge" in result.output
    assert "Usage" in result.output  # asked for, so shown even on failure
    assert "700 tokens" in result.output
    assert "150 tokens" in result.output  # 100 in + 50 out


# --- Bug 2: tool schemas were exposed for modalities nothing fits ---

CODING_ONLY_CATALOG = [
    ModelEntry(name="tiny-coder", modality="coding", runtime="ollama", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1),
    ModelEntry(name="huge-docs", modality="docs", runtime="ollama", min_vram_gb=0, min_ram_gb=999, disk_gb=1, quality_tier=1),
]


def test_tool_schemas_omit_modalities_with_no_fitting_model():
    """Exposing a tool the frontier model can only fail to use wastes a round
    trip and confuses it -- a modality nothing fits should be left out.
    """
    schemas = build_tool_schemas(_hw(ram_gb=8), CODING_ONLY_CATALOG)
    names = {s["function"]["name"] for s in schemas}

    assert "delegate_coding_task" in names  # tiny-coder fits 8GB RAM
    assert "delegate_docs_task" not in names  # huge-docs needs 999GB RAM
    assert "delegate_general_task" not in names  # nothing in this catalog at all


def test_tool_schemas_include_modalities_that_do_fit():
    schemas = build_tool_schemas(_hw(ram_gb=2000), CODING_ONLY_CATALOG)
    names = {s["function"]["name"] for s in schemas}
    assert {"delegate_coding_task", "delegate_docs_task"} <= names


# --- Bug 3: themed panel.border style was defined but never applied ---


def test_panels_use_the_themed_border_style():
    import inspect

    source = inspect.getsource(cli_module)
    # every Panel(...) call should set an explicit border_style so the active
    # theme actually applies to borders too, not just inline markup
    panel_calls = [line for line in source.splitlines() if "Panel(" in line and "console.print" in line]
    assert panel_calls, "expected to find Panel() calls in cli.py"
    for call in panel_calls:
        assert "border_style=" in call, f"Panel call without a themed border_style: {call.strip()}"
