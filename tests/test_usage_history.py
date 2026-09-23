"""/usage remembers: this session, the one before, and the project total.

Reported: "I need the full summary as well as the last session usage, not
just the recent" -- usage only knew the running process and forgot it all
on exit.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
from localforge import memory, usage_store
from localforge.orchestrator import RunResult, RunStats


def _stats(local=1000, prompt=100, completion=50, cost=0.02, sub=False):
    return RunStats(
        frontier_prompt_tokens=prompt,
        frontier_completion_tokens=completion,
        frontier_cost_usd=cost,
        local_tokens_generated=local,
        frontier_via_subscription=sub,
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from localforge import trust

    trust.trust(tmp_path)
    cli_module._session.root = tmp_path.resolve()
    return tmp_path.resolve()


def test_usage_is_written_per_task_and_survives_the_session(project):
    usage_store.record(project, "session-1", "claude-opus-5", _stats())
    usage_store.record(project, "session-1", "claude-opus-5", _stats(local=500, cost=0.01))
    total = usage_store.all_time(project)
    assert total.tasks == 2 and total.local_tokens_generated == 1500
    assert total.frontier_cost_usd == pytest.approx(0.03) and total.models == ["claude-opus-5"]
    assert usage_store.session(project, "session-1").tasks == 2
    assert usage_store.previous_session(project, "session-1") is None  # only one so far


def test_the_previous_session_is_the_one_before_this_one(project):
    usage_store.record(project, "old", "gpt-5", _stats(local=10))
    usage_store.record(project, "new", "gpt-5", _stats(local=20))
    previous = usage_store.previous_session(project, "new")
    assert previous.local_tokens_generated == 10
    assert usage_store.all_time(project).local_tokens_generated == 30


def test_subscription_cost_is_kept_apart_from_billed_cost(project):
    usage_store.record(project, "s", "claude-opus-5", _stats(cost=0.25, sub=True))
    usage_store.record(project, "s", "gpt-5", _stats(cost=0.10))
    total = usage_store.all_time(project)
    assert total.subscription_cost_usd == pytest.approx(0.25) and total.frontier_cost_usd == pytest.approx(0.10)


def test_only_the_last_sessions_are_kept(project):
    for i in range(usage_store.SESSIONS_KEPT + 5):
        usage_store.record(project, f"s{i}", "gpt-5", _stats(local=1))
    import json

    data = json.loads(usage_store.usage_file(project).read_text())
    assert len(data["sessions"]) == usage_store.SESSIONS_KEPT
    assert data["all_time"]["tasks"] == usage_store.SESSIONS_KEPT + 5  # the total still counts them all


def test_usage_lives_outside_the_project_beside_its_memory(project):
    usage_store.record(project, "s", "gpt-5", _stats())
    assert usage_store.usage_file(project).parent == memory.project_dir(project)
    assert not list(project.glob("**/usage.json"))


def test_a_task_records_its_usage_even_when_stopped(project, monkeypatch):
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)
    from localforge.orchestrator import TaskCancelled

    with patch.object(cli_module, "run_orchestrator", return_value=RunResult("ok", _stats(local=700))):
        CliRunner().invoke(cli_module.app, ["run", "one", "-m", "gpt-5"])
    with patch.object(cli_module, "run_orchestrator", side_effect=TaskCancelled(_stats(local=300))):
        CliRunner().invoke(cli_module.app, ["run", "two", "-m", "gpt-5"])
    total = usage_store.all_time(project)
    assert total.tasks == 2 and total.local_tokens_generated == 1000


def test_usage_shows_the_history_with_the_local_share(project):
    usage_store.record(project, "earlier-session", "claude-opus-5", _stats(local=4000, prompt=500, completion=500, cost=0.05))
    usage_store.record(project, cli_module._session.id, "claude-opus-5", _stats(local=2000, prompt=100, completion=100, cost=0.01))
    out = " ".join(CliRunner().invoke(cli_module.app, ["usage"]).output.replace("│", " ").split())
    assert "Usage history for" in out
    assert "Previous session" in out and "This project, all time" in out
    assert "4,000 (80%)" in out  # the previous session's local share
    assert "6,000 (83%)" in out and "$0.06" in out  # all time
    assert "saved" in out  # what that local work would have cost from the frontier model


def test_usage_history_is_shown_even_before_the_first_task_of_a_session(project):
    usage_store.record(project, "earlier", "gpt-5", _stats(local=123))
    out = " ".join(CliRunner().invoke(cli_module.app, ["usage"]).output.split())
    assert "No tasks run in this session yet" in out and "This project, all time" in out and "123" in out


def test_a_broken_usage_file_is_ignored_rather_than_crashing(project):
    usage_store.usage_file(project).parent.mkdir(parents=True, exist_ok=True)
    usage_store.usage_file(project).write_text("{not json")
    assert usage_store.all_time(project).tasks == 0
    usage_store.record(project, "s", "gpt-5", _stats())  # and writing still works
    assert usage_store.all_time(project).tasks == 1
