"""Tasks run in the background while the prompt stays live.

Reported: streaming "takes over the terminal", so no slash commands or
new tasks while localforge works. Now: a status bar instead of a token
dump, /summary mid-task, typed tasks queue, /tell adds to the running
task, /stop and Ctrl+C cancel it, and approvals are answered on the prompt.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge.background import TaskRunner
from localforge.hardware import HardwareProfile


def _wait(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --- the runner ----------------------------------------------------------------------------


def test_tasks_run_in_order_and_queue_behind_the_running_one():
    done, gate = [], threading.Event()

    def run(task):
        if task == "first":
            gate.wait(2)
        done.append(task)

    runner = TaskRunner(run)
    assert runner.submit("first") == 0
    assert _wait(lambda: runner.busy)
    assert runner.submit("second") == 1 and runner.submit("third") == 2
    gate.set()
    assert _wait(lambda: not runner.busy)
    assert done == ["first", "second", "third"]


def test_a_task_submitted_as_the_worker_finishes_is_not_lost():
    done = []
    runner = TaskRunner(done.append)
    for i in range(200):
        runner.submit(f"t{i}")
    assert _wait(lambda: not runner.busy and len(done) == 200)
    assert done == [f"t{i}" for i in range(200)]


def test_a_failing_task_does_not_stop_the_queue():
    done = []

    def run(task):
        if task == "bad":
            raise RuntimeError("boom")
        done.append(task)

    runner = TaskRunner(run)
    runner.submit("bad")
    runner.submit("good")
    assert _wait(lambda: done == ["good"])


def test_cancel_reaches_the_task_at_its_next_step():
    seen = []

    def run(task):
        try:
            while True:
                runner.check_cancel()
                time.sleep(0.01)
        except KeyboardInterrupt:
            seen.append("cancelled")

    runner = TaskRunner(run)
    runner.submit("long")
    assert _wait(lambda: runner.busy)
    assert runner.cancel() is True
    assert _wait(lambda: seen == ["cancelled"] and not runner.busy)
    assert runner.cancel() is False  # nothing left to stop


def test_approvals_are_handed_to_the_main_thread():
    answers = []

    def run(task):
        answers.append(runner.ask("write", "Create a.py"))

    runner = TaskRunner(run)
    runner.submit("t")
    assert _wait(lambda: runner.approval is not None)
    assert "waiting for you: Create a.py" in runner.toolbar()
    runner.answer(True, always=True)
    assert _wait(lambda: answers)
    assert answers[0].allowed and answers[0].always


def test_stopping_while_waiting_for_approval_declines_and_cancels():
    outcome = []

    def run(task):
        try:
            runner.ask("delete", "Delete x")
            outcome.append("continued")
        except KeyboardInterrupt:
            outcome.append("cancelled")

    runner = TaskRunner(run)
    runner.submit("t")
    assert _wait(lambda: runner.approval is not None)
    runner.cancel()
    assert _wait(lambda: outcome == ["cancelled"])


def test_notes_are_delivered_once():
    gate = threading.Event()
    runner = TaskRunner(lambda task: gate.wait(2))
    assert runner.tell("x") is False  # nothing running
    runner.submit("t")
    assert _wait(lambda: runner.busy)
    assert runner.tell("use sqlite") is True
    assert runner.take_notes() == ["use sqlite"] and runner.take_notes() == []
    gate.set()


def test_summary_and_toolbar_describe_who_is_doing_what():
    gate = threading.Event()
    runner = TaskRunner(lambda task: gate.wait(2))
    runner.submit("build it")
    runner.submit("then docs")
    assert _wait(lambda: runner.busy)
    s = runner.state
    s.orchestrator, s.phase = "claude-opus-5", "reviewing results (step 3)"
    s.local_model, s.local_what, s.local_tokens, s.local_started = "qwen2.5-coder:7b", "working on coding", 120, time.monotonic() - 4
    s.todos = [{"content": "write api", "status": "completed"}, {"content": "write tests", "status": "in_progress"}]
    text = "\n".join(runner.summary())
    assert "Task: build it" in text and "Orchestrator: claude-opus-5 — reviewing results (step 3)" in text
    assert "Local model: qwen2.5-coder:7b working on coding — 120 tokens so far" in text
    assert "[x] write api" in text and "[~] write tests" in text
    assert "Queued (1):" in text and "1. then docs" in text
    bar = runner.toolbar()
    assert "Forging with qwen2.5-coder:7b…" in bar and "working on coding" in bar and "queue: 1" in bar
    gate.set()


# --- the orchestrator side -----------------------------------------------------------------


def test_a_note_reaches_the_orchestrator_at_its_next_step():
    sent = []
    notes = [["also add logging"], []]

    def fake_completion(**kw):
        sent.append([m["content"] for m in kw["messages"]])
        msg = MagicMock(tool_calls=None, content="ok")
        msg.model_dump.return_value = {"role": "assistant", "content": "ok"}
        return MagicMock(choices=[MagicMock(message=msg)], usage=None)

    hooks = orch.ActivityHooks(poll_notes=lambda: notes.pop(0))
    hw = HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])
    with (
        patch.object(orch, "completion", side_effect=fake_completion),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        orch.run("build it", "gpt-5", hardware=hw, hooks=hooks)
    assert sent[0][-1] == "[Note from the user, added while you were working]: also add logging"


def test_background_activity_compresses_local_output(monkeypatch, capsys):
    runner = TaskRunner(lambda t: None)
    act = cli_module._BackgroundActivity("claude-opus-5", runner)
    entry = MagicMock()
    entry.name = "qwen2.5-coder:7b"
    act._on_delegate("coding", entry)
    for _ in range(500):
        act._on_token("def f(): pass\n")
    act._on_done("coding", entry, 500, 10.0)
    out = capsys.readouterr().out
    assert "def f(): pass" not in out  # not dumped
    assert "coding → qwen2.5-coder:7b" in out and "finished coding: 500 tokens" in out
    assert runner.state.local_tokens == 500


def test_background_answer_is_collected_then_printed_once(monkeypatch):
    from typer.testing import CliRunner

    from localforge.orchestrator import RunResult, RunStats

    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)
    runner = TaskRunner(lambda t: None)
    act = cli_module._BackgroundActivity("gpt-5", runner)
    monkeypatch.setattr(cli_module._session, "make_activity", lambda model: act)

    def fake_run(task, frontier_model, hooks=None, **kw):
        hooks.on_frontier(1)
        for piece in ["All ", "done ", "here."]:
            hooks.on_answer_text(piece)
        return RunResult("All done here.", RunStats())

    with patch.object(cli_module, "run_orchestrator", side_effect=fake_run):
        out = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"]).output
    assert out.count("All done here.") == 1
    assert runner.state.answer_words == 3


def test_approval_in_the_background_respects_always(monkeypatch):
    runner = TaskRunner(lambda t: None)
    act = cli_module._BackgroundActivity("m", runner)
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    try:
        from localforge.background import Approval

        monkeypatch.setattr(runner, "ask", lambda kind, title: Approval(kind, title, allowed=True, always=True))
        assert act.approve("write", "Create a.py", "+x") is True
        monkeypatch.setattr(runner, "ask", lambda kind, title: pytest.fail("should not ask again"))
        assert act.approve("write", "Create b.py", "+y") is True  # "always" remembered
    finally:
        cli_module.console.pop_theme()


# --- commands ------------------------------------------------------------------------------


@pytest.fixture
def busy_runner():
    gate = threading.Event()
    runner = TaskRunner(lambda t: gate.wait(3))
    cli_module._session.runner = runner
    runner.submit("building")
    assert _wait(lambda: runner.busy)
    yield runner
    gate.set()
    runner.shutdown()
    cli_module._session.runner = None


def test_summary_queue_stop_tell_commands(busy_runner):
    from typer.testing import CliRunner

    busy_runner.submit("next one")
    out = CliRunner().invoke(cli_module.app, ["summary"]).output
    assert "Task: building" in out and "next one" in out
    assert "1. next one" in CliRunner().invoke(cli_module.app, ["queue"]).output
    assert "next step" in CliRunner().invoke(cli_module.app, ["tell", "use", "sqlite"]).output
    assert busy_runner.take_notes() == ["use sqlite"]
    CliRunner().invoke(cli_module.app, ["queue", "clear"])
    assert not busy_runner.queue
    assert "Stopping" in CliRunner().invoke(cli_module.app, ["stop"]).output


def test_commands_without_a_background_runner_say_so():
    from typer.testing import CliRunner

    assert "Nothing is running" in CliRunner().invoke(cli_module.app, ["summary"]).output
    assert "queue is empty" in CliRunner().invoke(cli_module.app, ["queue"]).output
    assert "Nothing is running" in CliRunner().invoke(cli_module.app, ["stop"]).output


def test_the_repl_routes_tasks_to_the_runner_and_blocks_risky_commands(monkeypatch):
    from localforge import repl

    runner = MagicMock()
    runner.approval = None
    runner.busy = True
    runner.submit.return_value = 2
    console = MagicMock()
    reader = MagicMock()
    reader.read.side_effect = ["add a README", "/clear", "/usage", "/exit"]
    app = MagicMock()
    repl._read_eval(app, console, reader, runner)
    runner.submit.assert_called_once_with("add a README")
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
    assert "Queued (#2)" in printed
    assert "/clear has to wait until the current task is done" in printed
    app.assert_called_once_with(["usage"], standalone_mode=False)  # read-only commands still run
    runner.shutdown.assert_called_once()  # /exit while busy stops the task


def test_the_repl_answers_approvals_from_the_prompt():
    from localforge import repl

    runner = MagicMock()
    runner.busy = False
    runner.approval = MagicMock(kind="write")
    reader = MagicMock()
    reader.read.side_effect = ["a", "/exit"]
    repl._read_eval(MagicMock(), MagicMock(), reader, runner)
    runner.answer.assert_called_once_with(True, always=True)


def test_ctrl_c_with_a_task_running_stops_the_task_not_the_session():
    from localforge import repl

    runner = MagicMock()
    runner.busy = True
    runner.approval = None
    reader = MagicMock()
    reader.read.side_effect = [KeyboardInterrupt, "/exit"]
    console = MagicMock()
    repl._read_eval(MagicMock(), console, reader, runner)
    runner.cancel.assert_called_once()
    assert reader.read.call_count == 2  # the session kept going


# --- the status line has to look alive ------------------------------------------------
# Reported: "when nothing is happening, the user is confused" -- a slow step
# used to look like a hang.


def _running_runner(**state):
    runner = TaskRunner(lambda t: None)
    runner._running = True
    runner._begin("build a todo API")
    for key, value in state.items():
        setattr(runner.state, key, value)
    return runner


def test_the_status_line_reads_like_claude_codes():
    runner = _running_runner(orchestrator="claude-opus-5", phase="thinking with medium effort about the plan")
    bar = runner.toolbar(unicode=True)
    assert "Forging with claude-opus-5…" in bar and "thinking with medium effort about the plan" in bar
    assert "↓ 0 tokens" in bar and "0s" in bar and "/stop" in bar


def test_it_keeps_moving_while_a_step_is_slow():
    runner = _running_runner(orchestrator="m", phase="planning")
    frames = {runner.toolbar(unicode=True)[:6] for _ in _ticks()}
    assert len(frames) > 1  # the spinner and hammer animate on their own


def _ticks(n=12):
    for _ in range(n):
        time.sleep(0.06)
        yield


def test_tokens_add_up_across_delegations_and_the_answer():
    runner = _running_runner(local_total=2500, answer_chars=400)
    assert "↓ 2.6k tokens" in runner.toolbar(unicode=True)
    assert runner.state.tokens() == 2600


def test_a_quiet_step_says_how_long_it_has_been_quiet():
    runner = _running_runner(orchestrator="m", phase="planning")
    runner.state.last_event = time.monotonic() - 45
    assert "quiet for 45s" in runner.toolbar(unicode=True)
    runner.note_event()
    assert "quiet for" not in runner.toolbar(unicode=True)


def test_local_work_shows_the_model_and_speed():
    runner = _running_runner(local_model="qwen2.5-coder:7b", local_what="writing app.py", local_tokens=600, local_started=time.monotonic() - 30)
    bar = runner.toolbar(unicode=True)
    assert "Forging with qwen2.5-coder:7b…" in bar and "writing app.py, 20 tok/s" in bar


def test_plain_terminals_get_ascii_instead_of_emoji():
    runner = _running_runner(orchestrator="m", phase="planning")
    assert "🔨" not in runner.toolbar(unicode=False) and "🔨" in runner.toolbar(unicode=True)
    runner._running = False
    assert runner.toolbar(unicode=False).startswith(" [*] ready")


def test_counters_and_the_quiet_timer_are_fed_by_the_hooks():
    runner = TaskRunner(lambda t: None)
    runner._running = True
    runner._begin("t")
    act = cli_module._BackgroundActivity("claude-opus-5", runner)
    runner.state.last_event = time.monotonic() - 60
    entry = MagicMock()
    entry.name = "coder"
    act._on_delegate("coding", entry)
    assert "quiet for" not in runner.toolbar(unicode=True)
    for _ in range(10):
        act._on_token("x")
    act._on_answer_text("hello there")
    assert runner.state.local_total == 10 and runner.state.answer_chars == 11
