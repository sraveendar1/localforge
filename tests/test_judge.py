"""A dedicated judge model gives the orchestrator a real semantic verdict on
delegated output, instead of it having to scrutinize every result itself.

Layered on top of (never instead of) _looks_suspect()'s free heuristic. For a
file write, the judge call overlaps with verify.check() (and its retry)
rather than adding its own wait on top -- see Dispatcher._judge_check_async.
"""

import threading
from unittest.mock import patch

import pytest

from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import Dispatcher
from localforge.workspace import Workspace

CODER = ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)
JUDGE = ModelEntry(name="judge-model", modality="judge", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)
CATALOG = [CODER, JUDGE]


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


class Local:
    """A stub backend covering both the coder and the judge -- BACKENDS is
    keyed by runtime, and both entries above share runtime="stub", so which
    model answers is decided by `name` like a real Ollama install would."""

    def __init__(self, coder_reply="```python\nprint(1)\n```\nNOTES: none", judge_reply='{"pass": true, "reason": "fine"}'):
        self.coder_reply = coder_reply
        self.judge_reply = judge_reply
        self.calls: list[str] = []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.calls.append(name)
        if name == JUDGE.name:
            return {"type": "text", "content": self.judge_reply, "tokens": 5}
        return {"type": "text", "content": self.coder_reply, "tokens": 300}


@pytest.fixture
def project(tmp_path):
    return tmp_path


def _dispatcher(project, local, installed=None):
    dispatcher = Dispatcher(
        _hw(), catalog=CATALOG, installed=installed, workspace=Workspace(project, lambda *a: True)
    )
    return dispatcher, patch.dict("localforge.tools.BACKENDS", {"stub": local})


# --- resolving the judge model ----------------------------------------------


def test_judge_resolves_the_dedicated_model_when_installed(project):
    dispatcher, ctx = _dispatcher(project, Local(), installed={CODER.name, JUDGE.name})
    with ctx:
        entry = dispatcher.judge()
    assert entry is not None and entry.name == JUDGE.name


def test_judge_falls_back_to_an_installed_stand_in_when_not_installed(project):
    """Never a download for this -- but reusing an already-installed model
    beats not checking at all (same reasoning as resolve()'s TEXT_MODALITIES
    stand-in)."""
    dispatcher, ctx = _dispatcher(project, Local(), installed={CODER.name})  # dedicated judge not installed
    with ctx:
        entry = dispatcher.judge()
    assert entry is not None and entry.name == CODER.name


def test_judge_is_none_when_nothing_at_all_is_installed(project):
    dispatcher, ctx = _dispatcher(project, Local(), installed=set())
    with ctx:
        assert dispatcher.judge() is None


def test_judge_skips_the_check_rather_than_grade_its_own_output(project):
    """On the common single-local-model machine there's nothing else to
    fall back to -- the check is skipped rather than run at a known
    disadvantage (self-grading bias) on every task."""
    dispatcher, ctx = _dispatcher(project, Local(), installed={CODER.name})
    with ctx:
        assert dispatcher.judge(exclude=CODER.name) is None


def test_judge_is_used_when_installed_state_is_unknown(project):
    dispatcher, ctx = _dispatcher(project, Local(), installed=None)
    with ctx:
        entry = dispatcher.judge()
    assert entry is not None and entry.name == JUDGE.name


# --- the verdict itself ------------------------------------------------------


def test_judge_check_is_silent_when_it_approves(project):
    dispatcher, ctx = _dispatcher(project, Local(judge_reply='{"pass": true, "reason": "does the job"}'), installed={JUDGE.name})
    with ctx:
        assert dispatcher._judge_check("do the thing", "the output") is None


def test_judge_check_flags_a_failing_verdict(project):
    local = Local(judge_reply='{"pass": false, "reason": "ignores the empty-input case"}')
    dispatcher, ctx = _dispatcher(project, local, installed={JUDGE.name})
    with ctx:
        reason = dispatcher._judge_check("handle empty input", "the output")
    assert reason == "ignores the empty-input case"
    assert dispatcher.local_tokens_generated == 5  # the judge's own tokens count too


def test_judge_check_tolerates_an_unparseable_reply(project):
    dispatcher, ctx = _dispatcher(project, Local(judge_reply="not json at all"), installed={JUDGE.name})
    with ctx:
        assert dispatcher._judge_check("task", "output") is None


def test_judge_check_tolerates_a_backend_error(project):
    class Failing(Local):
        def generate(self, name, prompt, on_token=None, **kw):
            if name == JUDGE.name:
                raise RuntimeError("model crashed")
            return super().generate(name, prompt, on_token, **kw)

    dispatcher, ctx = _dispatcher(project, Failing(), installed={JUDGE.name})
    with ctx:
        assert dispatcher._judge_check("task", "output") is None


def test_nothing_installed_means_no_check_and_no_call(project):
    local = Local()
    dispatcher, ctx = _dispatcher(project, local, installed=set())
    with ctx:
        assert dispatcher._judge_check("task", "output") is None
    assert local.calls == []  # never called when nothing can judge


def test_a_stand_in_model_is_actually_used_when_the_dedicated_judge_is_missing(project):
    """Confirms judge()'s fallback isn't just resolved but wired all the way
    through _judge_check -- a different installed model gets called and its
    verdict is honored, with no dedicated judge installed at all."""
    other = ModelEntry(name="general-model", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)

    class TwoModelLocal(Local):
        def generate(self, name, prompt, on_token=None, **kw):
            self.calls.append(name)
            if name == other.name:
                return {"type": "text", "content": '{"pass": false, "reason": "missing edge case"}', "tokens": 5}
            return super().generate(name, prompt, on_token, **kw)

    local = TwoModelLocal()
    dispatcher = Dispatcher(_hw(), catalog=[CODER, other], installed={CODER.name, other.name}, workspace=Workspace(project, lambda *a: True))
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        reason = dispatcher._judge_check("task", "output", exclude=CODER.name)
    assert reason == "missing edge case"
    assert local.calls == [other.name]  # never the excluded writer


def test_the_stand_in_prefers_a_different_model_over_the_writer_when_one_exists(project):
    """A second text model gets excluded()'d in favor of a third, distinct
    one -- not just "some installed model", specifically not the writer."""
    other = ModelEntry(name="general-model", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)
    dispatcher = Dispatcher(_hw(), catalog=[CODER, other], installed={CODER.name, other.name}, workspace=Workspace(project, lambda *a: True))
    with patch.dict("localforge.tools.BACKENDS", {"stub": Local()}):
        entry = dispatcher.judge(exclude=CODER.name)
    assert entry.name == other.name


# --- wired into delegation ---------------------------------------------------


def test_plain_delegation_gets_a_warning_when_the_judge_flags_it(project):
    local = Local(judge_reply='{"pass": false, "reason": "does not match the request"}')
    dispatcher, ctx = _dispatcher(project, local, installed={CODER.name, JUDGE.name})
    with ctx:
        result = dispatcher.dispatch("delegate_coding_task", {"instructions": "write a function"})
    assert "[WARNING: the judge model flagged this result (does not match the request)" in result


def test_plain_delegation_is_unflagged_when_the_judge_approves(project):
    dispatcher, ctx = _dispatcher(project, Local(), installed={CODER.name, JUDGE.name})
    with ctx:
        result = dispatcher.dispatch("delegate_coding_task", {"instructions": "write a function"})
    assert "WARNING" not in result


def test_a_file_write_is_flagged_but_still_written_when_the_judge_disagrees(project):
    local = Local(judge_reply='{"pass": false, "reason": "does not add the requested validation"}')
    dispatcher, ctx = _dispatcher(project, local, installed={CODER.name, JUDGE.name})
    with ctx:
        result = dispatcher.dispatch("delegate_coding_task", {"instructions": "add input validation", "path": "out.py"})
    assert "[WARNING: the judge model flagged this file (does not add the requested validation)" in result
    assert (project / "out.py").read_text() == "print(1)\n"  # the WARNING never blocks the write


def test_a_file_write_is_unflagged_when_the_judge_approves(project):
    dispatcher, ctx = _dispatcher(project, Local(), installed={CODER.name, JUDGE.name})
    with ctx:
        result = dispatcher.dispatch("delegate_coding_task", {"instructions": "write it", "path": "out.py"})
    assert "WARNING" not in result
    assert (project / "out.py").is_file()


def test_the_judge_call_overlaps_with_the_syntax_check(project):
    """The judge is started right after the file content is extracted and
    only joined once verify.check() (and any retry) is done -- so a slow
    judge call doesn't add its own wait on top of the syntax check. Proven
    deterministically: the fake verify.check() unblocks the judge, so the
    judge call can only finish *after* verify.check ran, which is only
    possible if both were in flight at once."""
    order: list[str] = []
    verify_ran = threading.Event()

    class SlowJudge(Local):
        def generate(self, name, prompt, on_token=None, **kw):
            if name == JUDGE.name:
                order.append("judge-start")
                if not verify_ran.wait(timeout=2):
                    raise AssertionError("verify.check() never ran while the judge call was in flight")
                order.append("judge-end")
                return {"type": "text", "content": '{"pass": true, "reason": "fine"}', "tokens": 5}
            return super().generate(name, prompt, on_token, **kw)

    def fake_check(path, content):
        order.append("verify")
        verify_ran.set()
        return None, None  # no checker for this extension: never blocks the write

    dispatcher, ctx = _dispatcher(project, SlowJudge(), installed={CODER.name, JUDGE.name})
    with ctx, patch("localforge.tools.verify.check", side_effect=fake_check):
        result = dispatcher.dispatch("delegate_coding_task", {"instructions": "write it", "path": "out.txt"})
    assert "WARNING" not in result
    assert order.index("verify") < order.index("judge-end")


def test_exclude_tracks_whichever_model_actually_wrote_the_output_after_a_crash_retry(project):
    """resolve(modality) is cached to its *first*-choice pick and doesn't
    change after a crash-retry switches models -- excluding that cached
    pick instead of the model that actually produced the final content
    would let the real writer ("small") grade its own output instead of
    the model that never wrote anything ("big"). No dedicated judge is
    installed here, so this exercises the text-model stand-in path where
    `exclude` actually decides which model gets picked as judge."""
    big = ModelEntry(name="big", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=2)
    small = ModelEntry(name="small", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)

    class CrashingLocal(Local):
        def generate(self, name, prompt, on_token=None, **kw):
            self.calls.append(name)
            if name == big.name:
                raise RuntimeError("big: model requires more system memory than is available")
            return {"type": "text", "content": "code from small " + "x" * 40, "tokens": 5}

    local = CrashingLocal()
    dispatcher = Dispatcher(
        _hw(), catalog=[big, small], installed={big.name, small.name}, workspace=Workspace(project, lambda *a: True)
    )
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        result = dispatcher.dispatch("delegate_coding_task", {"instructions": "write x"})
    assert result.startswith("code from small")
    # "big" (never the actual writer) is called twice: the crashing first
    # attempt, then again as the stand-in judge -- proving `small`, the
    # real writer, was correctly excluded rather than grading itself.
    assert local.calls == ["big", "small", "big"]
