"""A paid cloud delegate target (delegate_target.py) isn't free local
compute and isn't orchestrator spend -- tracked as its own bucket through
RunStats -> usage_store.Totals -> the /usage panel, mirroring how
frontier_cost_usd/frontier_via_subscription already keep billed and
subscription cost apart.
"""
import localforge.cli as cli_module
from localforge import usage_store
from localforge.orchestrator import RunStats


def _stats(delegate_tokens=0, delegate_cost=0.0, delegate_notional=0.0):
    return RunStats(
        frontier_prompt_tokens=10, frontier_completion_tokens=5, frontier_cost_usd=0.001,
        local_tokens_generated=100,
        delegate_tokens_generated=delegate_tokens, delegate_cost_usd=delegate_cost,
        delegate_notional_cost_usd=delegate_notional,
    )


def test_delegate_cost_line_is_absent_with_no_delegate_usage():
    assert cli_module._delegate_cost_line(0, 0.0, 0.0) is None


def test_delegate_cost_line_shows_billed_cost():
    line = cli_module._delegate_cost_line(500, 0.03, 0.0)
    assert "500 tokens" in line
    assert "$0.0300 billed" in line


def test_delegate_cost_line_shows_subscription_cost_as_notional():
    line = cli_module._delegate_cost_line(500, 0.0, 0.05)
    assert "500 tokens" in line
    assert "subscription usage, not billed separately" in line
    assert "$0.0500" in line


def test_delegate_cost_line_handles_tokens_with_unknown_cost():
    line = cli_module._delegate_cost_line(500, 0.0, 0.0)
    assert "cost unknown" in line


def test_usage_store_totals_accumulate_delegate_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from localforge import trust

    trust.trust(tmp_path)
    root = tmp_path.resolve()

    usage_store.record(root, "s1", "claude-opus-5", _stats(delegate_tokens=100, delegate_cost=0.01))
    usage_store.record(root, "s1", "claude-opus-5", _stats(delegate_tokens=50, delegate_cost=0.005))

    totals = usage_store.all_time(root)
    assert totals.delegate_tokens_generated == 150
    assert round(totals.delegate_cost_usd, 6) == 0.015


def test_usage_store_totals_round_trip_delegate_fields_through_to_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from localforge import trust

    trust.trust(tmp_path)
    root = tmp_path.resolve()

    usage_store.record(root, "s1", "claude-opus-5", _stats(delegate_tokens=42, delegate_notional=0.02))
    totals = usage_store.Totals.from_dict(usage_store._read(root)["all_time"])
    assert totals.delegate_tokens_generated == 42
    assert round(totals.delegate_notional_cost_usd, 6) == 0.02


def test_stub_dispatcher_without_delegate_fields_does_not_break_stats_copy():
    """A hand-rolled Dispatcher stand-in from an older test file only ever
    had local_tokens_generated -- _copy_dispatch_stats must not crash."""
    import localforge.orchestrator as orch

    class OldStyleDispatcher:
        local_tokens_generated = 7

    stats = RunStats()
    orch._copy_dispatch_stats(stats, OldStyleDispatcher())
    assert stats.local_tokens_generated == 7
    assert stats.delegate_tokens_generated == 0
    assert stats.delegate_cost_usd == 0.0
