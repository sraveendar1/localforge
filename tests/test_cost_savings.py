"""Delegation that actually saves money.

Reported: "most of the work is done by the paid models". Causes found:
the orchestrator had to read files itself to give local models context,
nothing stopped it writing the code into its instructions, every local
write came back as a full diff into its (re-sent) context, and it ran at
full thinking effort for routine steps.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import cli_transport, config
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.orchestrator import RunStats
from localforge.tools import RESULT_PREVIEW_LINES, Dispatcher, build_tool_schemas
from localforge.workspace import Workspace


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


CODER = [ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)]


class Local:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        return {"type": "text", "content": self.reply, "tokens": 300}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "api.py").write_text("def get_user(user_id: int) -> dict:\n    ...\n")
    (tmp_path / "secret_elsewhere.txt").write_text("x")
    return tmp_path


def test_context_files_go_to_the_local_model_not_the_orchestrator(project):
    local = Local("```python\nfrom api import get_user\n```")
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project, lambda *a: True))
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        out = d.dispatch("delegate_coding_task", {"instructions": "use get_user", "path": "client.py", "context_files": ["api.py"]})
    assert "def get_user(user_id: int) -> dict:" in local.prompts[0]  # the local model saw the file
    assert "def get_user" not in out  # ...and the orchestrator was never handed it


def test_context_files_stay_inside_the_project(project):
    local = Local("x" * 50)
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project))
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "summarize", "context_files": ["../../etc/passwd", "api.py"]})
    assert "unavailable" in local.prompts[0] and "root:" not in local.prompts[0]
    assert "def get_user" in local.prompts[0]


def test_context_is_capped(project, monkeypatch):
    import localforge.tools as tools

    monkeypatch.setattr(tools, "MAX_CONTEXT_CHARS", 30)
    (project / "big.py").write_text("y" * 500)
    local = Local("x" * 50)
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project))
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "s", "context_files": ["big.py", "api.py"]})
    assert "cut to fit" in local.prompts[0] and "context limit reached" in local.prompts[0]


def test_a_written_file_comes_back_as_a_summary_not_the_whole_diff(project):
    code = "\n".join(f"line_{i} = {i}" for i in range(60))
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project, lambda *a: True))
    with patch.dict("localforge.tools.BACKENDS", {"stub": Local(f"```python\n{code}\n```")}):
        out = d.dispatch("delegate_coding_task", {"instructions": "write 60 lines", "path": "gen.py"})
    assert out.startswith("Created gen.py (+60 -0, 60 lines).")
    assert "line_0 = 0" in out and f"line_{RESULT_PREVIEW_LINES - 1} =" in out
    assert "line_59 = 59" not in out and "read_file gen.py" in out
    assert (project / "gen.py").read_text().count("\n") == 60  # the file itself is complete


def test_the_user_still_sees_the_full_diff_when_approving(project):
    seen = {}
    code = "\n".join(f"v{i} = {i}" for i in range(40))

    def approver(kind, title, detail):
        seen["diff"] = detail
        return True

    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project, approver))
    with patch.dict("localforge.tools.BACKENDS", {"stub": Local(f"```\n{code}\n```")}):
        d.dispatch("delegate_coding_task", {"instructions": "x", "path": "v.py"})
    assert "+v39 = 39" in seen["diff"]


def test_delegate_tools_offer_context_files():
    schema = {s["function"]["name"]: s["function"]["parameters"] for s in build_tool_schemas(_hw(), CODER)}
    props = schema["delegate_coding_task"]["properties"]
    assert props["context_files"]["type"] == "array"
    assert "never write the code yourself" in props["instructions"]["description"]


def test_system_prompt_makes_the_orchestrator_direct_not_do():
    prompt = orch.SYSTEM_PROMPT
    assert "You are the expensive model" in prompt
    assert "Never write code" in prompt and "context_files" in prompt


# --- thinking effort -----------------------------------------------------------------------


def _proc():
    return MagicMock(returncode=0, stdout=json.dumps({"result": '{"final_answer": "ok"}', "usage": {}}), stderr="")


def test_claude_orchestrates_at_medium_effort_by_default(monkeypatch):
    monkeypatch.delenv(config.ORCHESTRATOR_EFFORT_ENV_VAR, raising=False)
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", return_value=_proc()) as run:
        cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [])
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--effort") + 1] == "medium"


def test_effort_is_configurable(monkeypatch):
    monkeypatch.setenv(config.ORCHESTRATOR_EFFORT_ENV_VAR, "high")
    with patch.object(cli_transport, "available", return_value=True), patch.object(cli_transport.subprocess, "run", return_value=_proc()) as run:
        cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [])
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--effort") + 1] == "high"


# --- showing what was saved ---------------------------------------------------------------


def test_usage_shows_an_estimate_of_what_local_work_saved():
    line = cli_module._savings_line(4000, "claude-opus-5")
    assert "saved" in line and "4000 tokens of open-weighted output" in line
    price = __import__("litellm").model_cost["claude-opus-5"]["output_cost_per_token"]
    assert f"${4000 * price:.4f}" in line


def test_no_estimate_for_a_local_orchestrator_or_unknown_price():
    assert cli_module._savings_line(4000, "ollama/qwen2.5:7b") == ""
    assert cli_module._savings_line(4000, "some-unknown-model") == ""
    assert cli_module._savings_line(0, "claude-opus-5") == ""


def test_usage_command_includes_the_savings(monkeypatch):
    from typer.testing import CliRunner

    cli_module._session_usage.append(("claude-opus-5", RunStats(frontier_prompt_tokens=100, frontier_completion_tokens=50, local_tokens_generated=2000)))
    out = " ".join(CliRunner().invoke(cli_module.app, ["usage"]).output.replace("│", " ").split())
    assert "saved" in out and "2000 tokens of open-weighted output would have cost from claude-opus-5" in out
