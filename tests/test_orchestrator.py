from unittest.mock import MagicMock, patch

import pytest

import localforge.orchestrator as orch_module
from localforge.hardware import HardwareProfile
from localforge.orchestrator import KEEP_RECENT_TOOL_RESULTS, _collapse_old_tool_results


def _hw() -> HardwareProfile:
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def test_collapse_keeps_recent_and_collapses_older():
    messages = [{"role": "tool", "content": f"result-{i}"} for i in range(6)]
    _collapse_old_tool_results(messages, list(range(6)))

    excess = 6 - KEEP_RECENT_TOOL_RESULTS
    for i in range(excess):
        assert messages[i]["content"].startswith("[superseded:")
    for i in range(excess, 6):
        assert messages[i]["content"] == f"result-{i}"


def test_collapse_is_idempotent():
    already = "[superseded: earlier result, 5 chars -- no longer kept in full in context]"
    messages = [{"content": already}] * (KEEP_RECENT_TOOL_RESULTS + 1)
    _collapse_old_tool_results(messages, list(range(len(messages))))
    assert messages[0]["content"] == already  # untouched, not re-wrapped


def test_collapse_noop_under_threshold():
    messages = [{"content": "r"} for _ in range(KEEP_RECENT_TOOL_RESULTS)]
    _collapse_old_tool_results(messages, list(range(len(messages))))
    assert all(m["content"] == "r" for m in messages)


class StubDispatcher:
    def __init__(self, hardware, catalog=None, **kwargs):
        self.catalog = catalog or []
        self.local_tokens_generated = 321  # arbitrary fixed value to assert on

    def dispatch(self, tool_name, instructions, on_delegate=None):
        return "x" * 1000  # simulate a large result each round


def _usage(prompt_tokens=10, completion_tokens=5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    return usage


def _tool_call_response(call_id: str):
    msg = MagicMock()
    msg.tool_calls = [MagicMock(id=call_id, function=MagicMock())]
    msg.tool_calls[0].function.name = "delegate_coding_task"
    msg.tool_calls[0].function.arguments = '{"instructions": "do it"}'
    msg.model_dump.return_value = {"role": "assistant", "content": None}
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = _usage()
    return resp


def _final_response():
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = "done"
    msg.model_dump.return_value = {"role": "assistant", "content": "done"}
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage = _usage()
    return resp


def test_run_collapses_old_tool_results_over_a_long_task():
    rounds_before_final = 6
    call_count = {"n": 0}

    def fake_completion(model, messages, tools):
        n = call_count["n"]
        call_count["n"] += 1
        if n < rounds_before_final:
            return _tool_call_response(f"call_{n}")
        return _final_response()

    with (
        patch.object(orch_module, "completion", side_effect=fake_completion),
        patch.object(orch_module, "Dispatcher", StubDispatcher),
        patch.object(orch_module, "build_tool_schemas", return_value=[]),
        patch.object(orch_module.litellm, "completion_cost", return_value=0.002),
    ):
        result = orch_module.run("task", "claude-opus-5", hardware=_hw())

    assert result.answer == "done"
    assert call_count["n"] == rounds_before_final + 1
    # 7 completion() calls total (6 tool-call rounds + 1 final), each with usage(10, 5)
    assert result.stats.frontier_prompt_tokens == 10 * (rounds_before_final + 1)
    assert result.stats.frontier_completion_tokens == 5 * (rounds_before_final + 1)
    assert result.stats.frontier_cost_usd == pytest.approx(0.002 * (rounds_before_final + 1))
    assert result.stats.local_tokens_generated == 321


def test_run_never_sends_unbounded_history_to_the_frontier_model():
    """Integration check: capture the exact `messages` list sent on the final
    round and confirm old tool results were actually collapsed in place,
    not just theoretically collapsible.
    """
    rounds_before_final = 6
    snapshots: list[list[dict]] = []

    def fake_completion(model, messages, tools):
        snapshots.append([dict(m) for m in messages])
        n = len(snapshots) - 1
        if n < rounds_before_final:
            return _tool_call_response(f"call_{n}")
        return _final_response()

    with (
        patch.object(orch_module, "completion", side_effect=fake_completion),
        patch.object(orch_module, "Dispatcher", StubDispatcher),
        patch.object(orch_module, "build_tool_schemas", return_value=[]),
    ):
        orch_module.run("task", "claude-opus-5", hardware=_hw())

    final_messages = snapshots[-1]
    tool_msgs = [m for m in final_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == rounds_before_final
    collapsed = [m for m in tool_msgs if m["content"].startswith("[superseded:")]
    full = [m for m in tool_msgs if not m["content"].startswith("[superseded:")]
    assert len(collapsed) == rounds_before_final - KEEP_RECENT_TOOL_RESULTS
    assert len(full) == KEEP_RECENT_TOOL_RESULTS
