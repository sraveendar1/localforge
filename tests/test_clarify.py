"""Asking "why is this needed?" at a prompt, instead of only y/n/a.

Reported: "when there is a request for permissions or trust to folder, I
can't ask a question like why this is needed".
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge.background import Approval, TaskRunner
from localforge.catalog import ModelEntry


@pytest.fixture(autouse=True)
def _theme():
    cli_module.console.push_theme(cli_module.theme.get_theme("matrix"))
    yield
    cli_module.console.pop_theme()


def _pending(kind="write", title="Update src/app.py", detail="--- a/src/app.py\n+++ b/src/app.py\n+print(1)"):
    runner = TaskRunner(lambda t: None)
    runner._running = True
    runner._begin("add a health endpoint")
    runner.state.steps.append("● Read src/app.py")
    runner.approval = Approval(kind, title, detail)
    cli_module._session.runner = runner
    return runner


@pytest.fixture(autouse=True)
def _no_runner_afterwards():
    yield
    cli_module._session.runner = None
    cli_module._session.conversation = None


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("write", "the file stays exactly as it is"),
        ("delete", "no undo"),
        ("command", "with your permissions"),
        ("download", "real download"),
    ],
)
def test_each_kind_of_request_is_explained_plainly(kind, expected):
    _pending(kind=kind, title=f"{kind} something")
    out = " ".join(CliRunner().invoke(cli_module.app, ["why"]).output.split())
    assert expected in out
    assert "add a health endpoint" in out  # what it came up during
    assert "Read src/app.py" in out  # what happened just before
    assert "(y)es, (n)o, or (a)lways" in out


def test_why_with_nothing_pending_says_so(capsys):
    out = CliRunner().invoke(cli_module.app, ["why"]).output
    assert "Nothing is waiting for an answer" in out


def test_a_typed_question_is_answered_without_deciding_the_request(capsys):
    runner = _pending()
    runner.question_handler = cli_module.explain_to_user
    with patch.object(cli_module, "_answer_with_local_model", return_value=None):
        assert runner.ask_question("why do you need to change that file?") is True
    out = " ".join(capsys.readouterr().out.split())
    assert "the file stays exactly as it is" in out
    assert runner.approval is not None and not runner.approval.answered.is_set()  # still waiting


def test_a_local_model_answers_the_specific_question(capsys):
    runner = _pending()
    runner.question_handler = cli_module.explain_to_user
    seen = {}

    class Local:
        def ensure_available(self, name, on_progress=None):
            pass

        def generate(self, name, prompt, on_token=None, **kw):
            seen["prompt"] = prompt
            return {"type": "text", "content": "It adds one print line to app.py; nothing else changes.", "tokens": 12}

    from localforge.catalog import ModelEntry

    entry = ModelEntry(name="keeper", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)
    with (
        patch.object(cli_module.memory, "keeper", return_value=entry),
        patch.dict("localforge.backends.BACKENDS", {"stub": Local()}),
    ):
        runner.ask_question("what exactly does this change?")
    out = " ".join(capsys.readouterr().out.replace("│", " ").split())
    assert "adds one print line" in out and "from keeper (open-weighted model)" in out
    assert "Update src/app.py" in seen["prompt"] and "+print(1)" in seen["prompt"]
    assert "add a health endpoint" in seen["prompt"]  # the task is context for the answer


def test_without_a_local_model_the_plain_explanation_is_still_given(capsys):
    runner = _pending()
    runner.question_handler = cli_module.explain_to_user
    with patch.object(cli_module.memory, "keeper", return_value=None):
        runner.ask_question("is this safe?")
    assert "the file stays exactly as it is" in " ".join(capsys.readouterr().out.split())


def test_the_repl_routes_questions_and_still_takes_answers():
    from localforge import repl

    runner = MagicMock()
    runner.busy = False
    runner.approval = MagicMock(kind="write")
    runner.ask_question.return_value = True
    reader = MagicMock()
    reader.read.side_effect = ["why is this needed?", "y", "/exit"]
    repl._read_eval(MagicMock(), MagicMock(), reader, runner)
    runner.ask_question.assert_called_once_with("why is this needed?")
    runner.answer.assert_called_once_with(True, always=False)


def test_a_side_question_is_answered_from_cache_not_a_fresh_read(capsys):
    """Reported: asking a question while a task runs just queues it. The
    answer must come from what's already cached (brief/facts/memory), not
    a fresh file read -- so it's instant and never competes with the
    running task's own compute beyond the one generation call."""
    runner = TaskRunner(lambda t: None)
    runner._running = True
    runner._begin("add a health endpoint")
    cli_module._session.runner = runner
    conversation = orch.Conversation(brief="Built with FastAPI.", facts="- never use print()", memory="Last time we added auth.")
    cli_module._session.conversation = conversation

    seen = {}

    class Local:
        def ensure_available(self, name, on_progress=None):
            pass

        def generate(self, name, prompt, on_token=None, **kw):
            seen["prompt"] = prompt
            return {"type": "text", "content": "This project is built with FastAPI.", "tokens": 8}

    entry = ModelEntry(name="keeper", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)
    with (
        patch.object(cli_module.memory, "keeper", return_value=entry),
        patch.dict("localforge.backends.BACKENDS", {"stub": Local()}),
    ):
        cli_module._answer_side_question("what is this project built with?")
    out = " ".join(capsys.readouterr().out.replace("│", " ").split())
    assert "built with FastAPI" in out and "keeper (open-weighted model" in out
    assert "Built with FastAPI." in seen["prompt"]  # the cached brief, not a fresh read
    assert "never use print()" in seen["prompt"]
    assert "Last time we added auth." in seen["prompt"]
    assert "add a health endpoint" in seen["prompt"]  # the running task, for context


def test_a_side_question_prints_nothing_without_a_local_model(capsys):
    runner = TaskRunner(lambda t: None)
    runner._running = True
    cli_module._session.runner = runner
    with patch.object(cli_module.memory, "keeper", return_value=None):
        cli_module._answer_side_question("what does this do?")
    assert capsys.readouterr().out == ""


def test_the_trust_prompt_answers_questions_too(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_module, "_drain_buffered_input", lambda: None)
    answers = iter(["why do you need this?", "1"])
    monkeypatch.setattr(cli_module.console, "input", lambda prompt="": next(answers))
    assert cli_module._ask_trust(tmp_path) is True
    out = " ".join(capsys.readouterr().out.split())
    assert "asked once per folder" in out and "still asks you separately" in out
    assert "Type 1 or 2" not in out  # a question isn't treated as a typo
