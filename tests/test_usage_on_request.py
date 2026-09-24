"""Token usage is shown only when asked for (/usage in a session, or
`run --usage` for a one-off), not after every task.
"""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge.orchestrator import OrchestrationError, RunResult, RunStats


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "_session_usage", [])
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)


def _stats(local=100, prompt=40, completion=10, cost=0.01, sub=False):
    return RunStats(
        frontier_prompt_tokens=prompt,
        frontier_completion_tokens=completion,
        frontier_cost_usd=cost,
        local_tokens_generated=local,
        frontier_via_subscription=sub,
    )


def _usage_text() -> str:
    """`/usage` output with panel borders and line wraps flattened away."""
    out = CliRunner().invoke(cli_module.app, ["usage"]).output
    return " ".join(out.replace("│", " ").split())


def _run(args, result=None, error=None):
    kwargs = {"side_effect": error} if error else {"return_value": result}
    with patch.object(cli_module, "run_orchestrator", **kwargs):
        return CliRunner().invoke(cli_module.app, ["run", *args])


def test_run_does_not_print_usage_by_default():
    out = _run(["build it", "-m", "gpt-5"], RunResult("the answer", _stats())).output
    assert "the answer" in out
    assert "Usage" not in out
    assert "tokens" not in out


def test_run_usage_flag_prints_the_panel():
    out = _run(["build it", "-m", "gpt-5", "--usage"], RunResult("the answer", _stats())).output
    assert "Usage" in out
    assert "Orchestrator read: 40 tokens" in out and ": 10 tokens" in out and "50 frontier tokens in all" in out


def test_usage_command_reports_last_task_then_session_totals():
    _run(["one", "-m", "gpt-5"], RunResult("a", _stats(local=100, prompt=40, completion=10, cost=0.01)))
    _run(["two", "-m", "gpt-5"], RunResult("b", _stats(local=300, prompt=60, completion=20, cost=0.02)))

    out = _usage_text()
    assert "last task" in out
    assert "Orchestrator read: 60 tokens" in out and ": 20 tokens" in out and "80 frontier tokens in all" in out
    assert "session (2 tasks)" in out
    assert "local models: 400 tokens" in out
    assert "Orchestrator read: 100 tokens" in out and "130 frontier tokens in all ($0.0300)" in out


def test_usage_session_keeps_billed_and_subscription_cost_apart():
    _run(["one", "-m", "gpt-5"], RunResult("a", _stats(cost=0.01)))
    _run(["two", "-m", "claude-opus-5"], RunResult("b", _stats(cost=0.25, sub=True)))

    out = _usage_text()
    assert "$0.0100; ~$0.2500 of subscription usage, not billed separately" in out


def test_failed_run_still_counts_toward_usage_but_prints_nothing():
    out = _run(["build it", "-m", "gpt-5"], error=OrchestrationError("no convergence", _stats(prompt=7, completion=3))).output
    assert "no convergence" in out
    assert "tokens" not in out

    usage = _usage_text()
    assert "Orchestrator read: 7 tokens" in usage and ": 3 tokens" in usage and "10 frontier tokens in all" in usage


def test_usage_before_any_task_explains_itself():
    out = _usage_text()
    assert "No tasks run in this session yet" in out
    assert "localforge run --usage" in out


def test_usage_is_listed_in_session_help():
    from localforge.repl import SLASH_HELP

    assert "/usage" in SLASH_HELP
