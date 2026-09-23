"""Keeping local models grounded, checked, and heard (asked for: "keep the
open-weight models grounded and ensure they're doing their work properly and
updating the orchestrator properly").

1. Every delegated task carries the project brief's stack, commands and
   conventions, and rules against inventing APIs -- before, a local model
   saw a one-line preface and the orchestrator's instructions, nothing else.
2. A file a local model writes must parse before the user is asked about it;
   one free local retry with the exact error, and nothing is written if that
   fails too.
3. The local model reports back: a few NOTES lines (assumptions, anything
   not done) go to the orchestrator with the syntax result -- never into the file.
"""

import shutil
from unittest.mock import patch

import pytest

from localforge import brief, verify
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import LOCAL_RULES, Dispatcher, _split_notes
from localforge.workspace import Workspace

CODER = [ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)]
AGENTS = """# Project brief

## What this project is
A todo API for the team.

## How it's built
Python 3.12, FastAPI, SQLAlchemy 2. Entry point: app/main.py.

## Layout
app/ holds the code.

## Running and testing
uv run pytest -q

## Conventions and decisions
Type hints everywhere; no print(), use logging.
"""


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[], memory_bandwidth_gbps=400)


class Local:
    """Replies in turn from `replies` (the last one repeats)."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"type": "text", "content": reply, "tokens": 10}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "AGENTS.md").write_text(AGENTS)
    return tmp_path


def _run(project, local, args, approver=lambda *a: True):
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project, approver))
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        return d.dispatch("delegate_coding_task", args)


# --- 1. grounded --------------------------------------------------------------------


def test_grounding_takes_the_stack_commands_and_conventions(project):
    text = brief.grounding_for_local(project)
    assert "FastAPI" in text and "uv run pytest -q" in text and "no print()" in text
    assert "A todo API for the team." not in text  # the story isn't needed to write code
    assert "app/ holds the code." not in text


def test_another_tools_brief_is_used_from_its_opening(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Always run make lint.\n" + "x\n" * 2000)
    text = brief.grounding_for_local(tmp_path)
    assert text.startswith("Always run make lint.") and len(text) <= brief.GROUNDING_CHARS + 10


def test_every_delegation_carries_the_rules_and_the_project_notes(project):
    local = Local("some helpful output here")
    _run(project, local, {"instructions": "add a health endpoint"})
    prompt = local.prompts[0]
    assert LOCAL_RULES in prompt and "never invent" in prompt
    assert "FastAPI" in prompt and "Type hints everywhere" in prompt
    assert prompt.rstrip().endswith("add a health endpoint")  # the task comes last


def test_without_a_brief_the_rules_still_apply(tmp_path):
    local = Local("some helpful output here")
    _run(tmp_path, local, {"instructions": "add a health endpoint"})
    assert LOCAL_RULES in local.prompts[0] and "Project notes" not in local.prompts[0]


# --- 2. checked before approval -------------------------------------------------------


def test_a_syntax_error_is_fixed_locally_before_the_user_is_asked(project):
    approvals = []
    local = Local("```python\ndef health(:\n    return 1\n```", "```python\ndef health():\n    return 1\n```\nNOTES: none")
    out = _run(project, local, {"instructions": "add health()", "path": "app/health.py"}, lambda *a: approvals.append(a) or True)
    assert len(local.prompts) == 2 and "failed a syntax check" in local.prompts[1] and "line 1" in local.prompts[1]
    assert len(approvals) == 1  # asked once, about the fixed file only
    assert (project / "app/health.py").read_text() == "def health():\n    return 1\n"
    assert "Syntax check: Python parses." in out


def test_if_the_retry_still_does_not_parse_nothing_is_written_or_asked(project):
    approvals = []
    local = Local("```python\ndef health(:\n```")
    out = _run(project, local, {"instructions": "add health()", "path": "app/health.py"}, lambda *a: approvals.append(a) or True)
    assert out.startswith("delegate_coding_task failed:") and "doesn't parse" in out
    assert approvals == [] and not (project / "app/health.py").exists()


@pytest.mark.parametrize(
    "path, good, bad",
    [
        ("config.json", '{"a": 1}', '{"a": 1,}'),
        ("config.yaml", "a: 1\nb: [2]", "a: [1\nb: 2"),
        ("pyproject.toml", '[tool]\nname = "x"', "[tool\nname = x"),
    ],
)
def test_data_files_are_checked(path, good, bad):
    assert verify.check(path, good) == (verify.CHECKERS[path[path.rindex("."):]][0], None)
    language, error = verify.check(path, bad)
    assert language and error


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_is_checked_with_node():
    assert verify.check("a.js", "const x = 1;\n")[1] is None
    assert "JavaScript syntax error" in verify.check("a.js", "const x = ;\n")[1]


def test_a_kind_of_file_with_no_checker_is_written_without_a_claim(project):
    out = _run(project, Local("```\nplain notes about the release, for the team\n```"), {"instructions": "write notes", "path": "NOTES.txt"})
    assert out.startswith("Created NOTES.txt") and "Syntax check" not in out
    assert verify.check("x.rs", "fn main( {") == (None, None)


# --- 3. heard ------------------------------------------------------------------------


def test_the_local_models_notes_reach_the_orchestrator_not_the_file(project):
    reply = "```python\nimport logging\n\ndef health():\n    return {'ok': True}\n```\nNOTES: assumed no auth is needed; didn't add a route."
    out = _run(project, Local(reply), {"instructions": "add health()", "path": "app/health.py"})
    assert "The local model's notes: assumed no auth is needed; didn't add a route." in out
    assert "NOTES" not in (project / "app/health.py").read_text()


def test_notes_none_is_left_out(project):
    out = _run(project, Local("```python\nx = 1\n```\nNOTES: none"), {"instructions": "x", "path": "x.py"})
    assert "notes" not in out.lower()


def test_notes_after_a_readme_with_its_own_code_blocks():
    readme = "# Tool\n\n```bash\npip install tool\n```\n\nDone.\n"
    body, notes = _split_notes(f"```markdown\n{readme}```\nNOTES: left out the license section")
    assert notes == "left out the license section"
    assert body.rstrip().endswith("```") and "NOTES" not in body


def test_notes_without_a_code_block_are_split_off_too():
    body, notes = _split_notes("x = 1\nNOTES: assumed Python 3.12")
    assert body == "x = 1" and notes == "assumed Python 3.12"
