"""A usage limit pauses the task instead of killing it.

Reported: "when localforge picks the paid model and the subscription or
limit ends, it breaks" -- the task died and its work was lost. Now the
limit is recognized (with the reset time the provider gives), the task
waits and carries on from where it stopped, and /model switches the
orchestrator to continue right away.
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import cli_transport, config
from localforge.background import TaskRunner
from localforge.cli_transport import UsageLimitError, looks_like_limit, parse_reset_time
from localforge.hardware import HardwareProfile

SPEND_LIMIT = (
    "You've hit your monthly spend limit · raise it at claude.ai/settings/usage "
    "· your session limit resets 11:10pm (Europe/Lisbon)"
)
# The real message Anthropic's CLI sends for a plan session limit (reported
# live: localforge fell back to a generic error and handed control back
# instead of waiting and resuming). Deliberately has no "spend limit"/"usage
# limit" phrase anywhere in it, unlike SPEND_LIMIT above -- that fixture
# already contained "spend limit" elsewhere in the string, so it matched
# regardless of whether "session limit" itself was ever recognized, which is
# exactly how this gap went unnoticed.
SESSION_LIMIT = "claude exited 1: You've hit your session limit · resets 11:40am (America/Chicago)"


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch):
    monkeypatch.setattr(cli_module, "LIMIT_POLL_SECONDS", 0.01)


# --- recognizing a limit ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (SPEND_LIMIT, True),
        (SESSION_LIMIT, True),
        ("You've exceeded your usage limit", True),
        ("429 Too Many Requests", True),
        ("Your credit balance is too low", True),
        ("Invalid API key · Please run /login", False),
        ("claude exited 1: Stream closed unexpectedly", False),
    ],
)
def test_limit_messages_are_told_apart_from_other_failures(text, expected):
    assert looks_like_limit(text) is expected


def test_the_real_session_limit_message_is_recognized_as_a_usage_limit():
    """orchestrator._as_usage_limit() is what decides whether on_limit ever
    gets called at all. The actual reported bug: this returned None for the
    real message (since looks_like_limit() didn't recognize "session
    limit"), so a plain CLINotAvailableError reached the caller instead --
    no wait, no auto-resume, just control handed back to the user."""
    exc = cli_transport.CLINotAvailableError(SESSION_LIMIT)  # what the CLI transport actually raises on exit 1
    limit = orch._as_usage_limit(exc, {"provider": "anthropic", "model": "claude-opus-5"})
    assert isinstance(limit, UsageLimitError)
    assert limit.reset_at is not None and limit.reset_at.hour == 11 and limit.reset_at.minute == 40


def test_the_real_session_limit_message_is_actually_waited_out(activity):
    """Full on_limit path carrying the real message text (reset_at set to
    shortly from now, like the other on_limit tests below do, so the test
    doesn't depend on what time of day it happens to run)."""
    at = datetime.now(timezone.utc) + timedelta(seconds=0.05)
    exc = UsageLimitError(SESSION_LIMIT, provider="anthropic", reset_at=at)
    assert activity.on_limit(exc) == "retry"


def test_reset_time_is_read_from_the_message_and_converted_to_local_time():
    now = datetime(2026, 9, 21, 21, 0, tzinfo=timezone(timedelta(hours=1)))  # 9pm where the limit message came from
    reset = parse_reset_time(SPEND_LIMIT, now=now)
    assert reset.hour == 23 and reset.minute == 10 and reset.date() == now.date()


def test_a_reset_time_already_past_means_tomorrow():
    now = datetime(2026, 9, 21, 23, 30, tzinfo=timezone(timedelta(hours=1)))
    reset = parse_reset_time("resets 11:10pm (Europe/Lisbon)", now=now)
    assert reset.day == 22 and reset.hour == 23


@pytest.mark.parametrize("text", ["You have exceeded your monthly quota.", "usage limit reached", "", "resets soon"])
def test_no_reset_time_is_none(text):
    assert parse_reset_time(text) is None


def test_an_unknown_timezone_falls_back_to_local_time():
    assert parse_reset_time("resets at 6am (Mars/Olympus)") is not None


def test_the_cli_transport_raises_a_usage_limit_error():
    proc = MagicMock(returncode=1, stdout=f'{{"result": "{SPEND_LIMIT}"}}', stderr="")
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", return_value=proc):
        with pytest.raises(UsageLimitError) as exc:
            cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [])
    assert exc.value.provider == "anthropic" and exc.value.reset_at is not None


def test_other_cli_failures_are_not_usage_limits():
    proc = MagicMock(returncode=1, stdout='{"result": "Invalid API key"}', stderr="")
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", return_value=proc):
        with pytest.raises(cli_transport.CLINotAvailableError) as exc:
            cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [])
    assert not isinstance(exc.value, UsageLimitError)


def test_an_api_rate_limit_is_recognized_too():
    class RateLimitError(Exception):
        pass

    limit = orch._as_usage_limit(RateLimitError("429 from the API"), {"provider": "openai"})
    assert isinstance(limit, UsageLimitError) and limit.provider == "openai"
    assert orch._as_usage_limit(ValueError("something else"), {}) is None


# --- the task waits and carries on -------------------------------------------------------


def _final(text="done"):
    msg = MagicMock(tool_calls=None, content=text)
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


def _run_hitting_limit(hooks, error=None, replies=None):
    """A run whose first orchestrator call hits a limit."""
    calls = {"n": 0}
    replies = replies or [_final()]

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise error or UsageLimitError(SPEND_LIMIT, "anthropic", datetime.now(timezone.utc) + timedelta(seconds=0.05))
        return replies[min(calls["n"] - 2, len(replies) - 1)]

    conv = orch.Conversation()
    with (
        patch.object(orch.cli_transport, "complete", side_effect=flaky),
        patch.object(orch, "_installed_models", return_value=None),
    ):
        result = orch.run("build it", "claude-opus-5", hardware=_hw(), cli_provider="anthropic", hooks=hooks, conversation=conv)
    return result, calls["n"], conv


def test_the_same_task_continues_after_the_limit_resets():
    decisions = []

    def on_limit(exc):
        decisions.append(exc)
        return "retry"

    result, calls, conv = _run_hitting_limit(orch.ActivityHooks(on_limit=on_limit))
    assert result.answer == "done" and calls == 2  # retried the same turn
    assert len(decisions) == 1 and decisions[0].reset_at is not None
    assert [m["content"] for m in conv.messages if m["role"] == "user"] == ["build it"]  # not asked again


def test_switching_the_orchestrator_continues_the_task_on_it():
    seen = {}

    def fake_local(model, messages, tools, on_text=None):
        seen["model"] = model
        return cli_transport.CLIResponse(
            choices=[cli_transport._Choice(message=cli_transport.message_from_reply('{"final_answer": "finished locally"}'))], local=True
        )

    hooks = orch.ActivityHooks(on_limit=lambda exc: ("switch", "ollama/qwen2.5:7b", None))
    with patch.object(orch.local_transport, "complete", side_effect=fake_local):
        result, calls, _ = _run_hitting_limit(hooks)
    assert result.answer == "finished locally" and seen["model"] == "ollama/qwen2.5:7b"


def test_without_a_decision_the_limit_is_still_raised():
    with pytest.raises(UsageLimitError):
        _run_hitting_limit(orch.ActivityHooks(on_limit=lambda exc: None))
    with pytest.raises(UsageLimitError):
        _run_hitting_limit(orch.ActivityHooks())  # no hook at all (one-off run)


def test_giving_up_on_a_limit_still_saves_open_work_and_usage(tmp_path):
    """Reported: resuming a task after a usage limit doesn't work. Root
    cause -- a UsageLimitError giving up used to skip both open-work saving
    and usage accounting (unlike OrchestrationError/TaskCancelled), so the
    next session never offered to resume and /usage silently lost that
    task's tokens."""
    from localforge import memory
    from localforge.workspace import Workspace

    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        raise UsageLimitError(SPEND_LIMIT, "anthropic", None)  # no reset time -> on_limit gives up

    conv = orch.Conversation()
    with (
        patch.object(orch.cli_transport, "complete", side_effect=flaky),
        patch.object(orch, "_installed_models", return_value=None),
    ):
        with pytest.raises(UsageLimitError) as excinfo:
            orch.run(
                "build it",
                "claude-opus-5",
                hardware=_hw(),
                cli_provider="anthropic",
                hooks=orch.ActivityHooks(on_limit=lambda exc: None),
                conversation=conv,
                workspace=Workspace(tmp_path),
            )

    # The exception carries the usage spent so far, like OrchestrationError/TaskCancelled do.
    assert excinfo.value.stats is not None

    # The task is offered again next session instead of silently vanishing.
    entries = memory.open_work(tmp_path)
    assert len(entries) == 1 and entries[0]["task"] == "build it"
    assert "usage limit" in entries[0]["why"]


def test_a_limit_is_never_treated_as_a_transient_failure():
    # transient failures retry twice by themselves; a limit must reach on_limit
    seen = []
    hooks = orch.ActivityHooks(on_limit=lambda exc: seen.append(exc) or "retry", on_tool=lambda *a: seen.append("retry-hook"))
    _run_hitting_limit(hooks)
    assert len(seen) == 1 and isinstance(seen[0], UsageLimitError)


# --- what the user sees, and the ways out --------------------------------------------------


@pytest.fixture
def activity(monkeypatch):
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    yield cli_module._LiveActivity("claude-opus-5")
    cli_module.console.pop_theme()


def _limit(seconds_away=0.05, reset=True):
    at = datetime.now(timezone.utc) + timedelta(seconds=seconds_away) if reset else None
    return UsageLimitError(SPEND_LIMIT, "anthropic", at)


def test_waiting_prints_the_countdown_and_then_retries(activity, capsys):
    assert activity.on_limit(_limit()) == "retry"
    out = capsys.readouterr().out
    assert "hit its usage limit" in out and "Waiting until" in out
    assert "/model switches the orchestrator" in out and "continuing" in out


def test_no_reset_time_means_no_waiting(activity, capsys):
    assert activity.on_limit(_limit(reset=False)) is None
    out = " ".join(capsys.readouterr().out.split())
    assert "didn't say when it resets" in out and "work so far is kept" in out


def test_a_reset_too_far_away_is_not_waited_for(activity, capsys):
    assert activity.on_limit(_limit(seconds_away=7 * 3600)) is None
    assert "doesn't reset until" in capsys.readouterr().out


def test_it_gives_up_after_a_few_waits_in_a_row(activity):
    for _ in range(cli_module._LiveActivity.MAX_LIMIT_WAITS):
        assert activity.on_limit(_limit()) == "retry"
    assert activity.on_limit(_limit()) is None


def test_switching_models_while_waiting_continues_immediately(activity, monkeypatch):
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "claude-opus-5")

    def switch_soon():
        time.sleep(0.05)
        config.save({config.FRONTIER_MODEL_ENV_VAR: "ollama/qwen2.5:7b"})

    threading.Thread(target=switch_soon, daemon=True).start()
    decision = activity.on_limit(_limit(seconds_away=30))
    assert decision[0] == "switch" and decision[1] == "ollama/qwen2.5:7b"
    assert decision[2] is None  # a local model uses no provider CLI


def test_the_background_session_shows_the_wait_and_allows_model_switching(monkeypatch):
    runner = TaskRunner(lambda t: None)
    activity = cli_module._BackgroundActivity("claude-opus-5", runner)
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "claude-opus-5")
        seen = []
        original = activity._limit_tick

        def spy(remaining):
            original(remaining)
            seen.append((runner.toolbar(), runner.waiting, "\n".join(runner.summary())))

        activity._limit_tick = spy
        assert activity.on_limit(_limit(seconds_away=0.05)) == "retry"
    finally:
        cli_module.console.pop_theme()
    toolbar, waiting, summary = seen[0]
    assert "usage limit resets in" in toolbar and "/model to switch" in toolbar
    assert waiting is True
    assert "Paused: usage limit resets in" in summary
    assert runner.state.waiting_for == ""  # cleared once the wait is over


def test_stopping_while_paused_cancels_the_task():
    runner = TaskRunner(lambda t: None)
    activity = cli_module._BackgroundActivity("claude-opus-5", runner)
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        runner.cancel_event.set()  # as /stop or Ctrl+C does
        with pytest.raises(KeyboardInterrupt):
            activity.on_limit(_limit(seconds_away=30))
    finally:
        cli_module.console.pop_theme()
    assert runner.state.waiting_for == ""


def test_model_is_allowed_while_paused_on_a_limit():
    from localforge import repl

    runner = MagicMock()
    runner.approval = None
    runner.busy = True
    runner.waiting = True
    reader = MagicMock()
    reader.read.side_effect = ["/model ollama/qwen2.5:7b", "/exit"]
    app = MagicMock()
    repl._read_eval(app, MagicMock(), reader, runner)
    app.assert_called_once_with(["model", "ollama/qwen2.5:7b"], standalone_mode=False)


def test_model_is_also_allowed_while_a_task_is_just_ordinarily_running():
    """Reported: "/model is not able to switch immediately" -- it used to be
    fully blocked while a task was busy, with an exception carved out only
    for the paused-on-a-usage-limit case. Switching model never touches
    state a running task holds a live reference to (it's just an env var
    write), so there's no reason to make the user wait for the task to
    finish before it takes effect for the next one."""
    from localforge import repl

    runner = MagicMock()
    runner.approval = None
    runner.busy = True
    runner.waiting = False  # ordinarily running, not paused on a limit
    reader = MagicMock()
    reader.read.side_effect = ["/model ollama/qwen2.5:7b", "/exit"]
    app = MagicMock()
    repl._read_eval(app, MagicMock(), reader, runner)
    app.assert_called_once_with(["model", "ollama/qwen2.5:7b"], standalone_mode=False)


def test_human_duration_reads_well():
    assert cli_module._human_duration(8040) == "2h 14m"
    assert cli_module._human_duration(75) == "1m 15s"
    assert cli_module._human_duration(9) == "9s"
