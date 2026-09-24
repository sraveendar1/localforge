"""Keeping the paid orchestrator to planning, delegating, validating and
reconciling -- with the writing done locally (reported: "the work is mostly
done by the frontier model and not the open-weight models").

Measured live through `claude -p` on one ordinary step:
* Claude Code's own default system prompt: ~8,100 tokens every step, with
  its tools off and irrelevant here. Replaced by localforge's (~1,960).
* The built-in `advisor` tool, left on by `--tools ""`, sent the whole
  conversation to Opus: 75% of that step's cost, invisible in `usage`.
* Cached input is reported apart from input_tokens and wasn't counted.
"""

import json
from unittest.mock import patch

from localforge import cli_transport, config
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import MAX_EDIT_LINES, Dispatcher
from localforge.workspace import Workspace

MESSAGES = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "add hello.py"}]
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read.", "parameters": {"properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


# --- claude gets localforge's system prompt, not Claude Code's ---------------------------


def test_claude_gets_our_instructions_as_its_system_prompt():
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["stdin"] = cmd, kw.get("input")
        out = {"result": '{"final_answer": "ok"}', "usage": {"input_tokens": 1, "output_tokens": 1}}
        return type("P", (), {"returncode": 0, "stdout": json.dumps(out), "stderr": ""})()

    with patch.object(cli_transport.subprocess, "run", fake_run), patch.object(cli_transport, "available", return_value=True):
        cli_transport.complete("anthropic", MESSAGES, TOOLS)
    cmd = seen["cmd"]
    system = cmd[cmd.index("--system-prompt") + 1]
    assert system.startswith("RULES") and "read_file(path)" in system and "final_answer" in system
    assert "RULES" not in seen["stdin"] and "add hello.py" in seen["stdin"]  # the conversation goes over stdin


def test_every_other_transport_still_gets_one_prompt():
    prompt = cli_transport._render_prompt(MESSAGES, TOOLS)
    assert prompt.startswith("RULES") and "add hello.py" in prompt and prompt.rstrip().endswith('"<your reply to the user>"}')


def test_claudes_advisor_tool_is_switched_off():
    args = config.FRONTIER_CLI_AUTH["anthropic"]["isolation_args"]
    assert args[args.index("--disallowed-tools") + 1] == "advisor"
    # the user's own ~/.claude settings (advisorModel: opus, hooks...) never apply to localforge's calls
    assert args[args.index("--setting-sources") + 1] == "project"
    for flag in ("--tools", "--disallowed-tools"):  # variadic: must be followed by another option
        assert args[args.index(flag) + 2].startswith("--")


def test_the_protocol_asks_for_independent_steps_in_one_reply():
    assert "in ONE reply as several tool_calls" in cli_transport._PROTOCOL


# --- usage counts everything the orchestrator read ------------------------------------------


def test_cached_input_counts_as_input():
    usage = cli_transport._read_usage({"input_tokens": 9, "cache_creation_input_tokens": 8099, "cache_read_input_tokens": 100, "output_tokens": 42})
    assert (usage.prompt_tokens, usage.completion_tokens) == (8208, 42)


def test_every_model_a_step_used_is_counted():
    envelope = {
        "result": "{}",
        "usage": {"input_tokens": 27, "output_tokens": 1171},  # the main model only
        "total_cost_usd": 0.077,
        "modelUsage": {
            "claude-haiku-4-5": {"inputTokens": 27, "cacheReadInputTokens": 15416, "cacheCreationInputTokens": 6351, "outputTokens": 1171},
            "claude-opus-5-5": {"inputTokens": 7813, "outputTokens": 1309},  # the advisor
        },
    }
    _, usage, cost = cli_transport._unwrap_envelope(json.dumps(envelope), config.FRONTIER_CLI_AUTH["anthropic"])
    assert usage.prompt_tokens == 27 + 15416 + 6351 + 7813 and usage.completion_tokens == 1171 + 1309
    assert cost == 0.077


# --- the orchestrator doesn't write the code itself ---------------------------------------

CODER = [ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)]


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[], memory_bandwidth_gbps=400)


class Local:
    def __init__(self):
        self.prompts = []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        return {"type": "text", "content": "a reasonable answer from the local model", "tokens": 5}


def test_a_big_edit_file_is_turned_back_to_the_local_model(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(tmp_path, lambda *a: True))
    big = "\n".join(f"line_{i} = {i}" for i in range(MAX_EDIT_LINES + 5))
    out = d.dispatch("edit_file", {"path": "app.py", "old_string": "x = 1", "new_string": big})
    assert out.startswith("edit_file failed:") and "delegate_coding_task" in out
    assert (tmp_path / "app.py").read_text() == "x = 1\n"  # nothing changed


def test_a_small_fix_up_is_still_allowed(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(tmp_path, lambda *a: True))
    out = d.dispatch("edit_file", {"path": "app.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert out.startswith("Updated app.py")


def test_instructions_full_of_code_are_refused(tmp_path):
    local = Local()
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(tmp_path, lambda *a: True))
    code = "```python\n" + "\n".join(f"v{i} = {i}" for i in range(20)) + "\n```"
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        out = d.dispatch("delegate_coding_task", {"instructions": f"write exactly this:\n{code}", "path": "v.py"})
    assert out.startswith("delegate_coding_task failed:") and "20 lines of code" in out
    assert local.prompts == []  # nothing was sent


def test_a_short_interface_in_the_instructions_is_fine(tmp_path):
    local = Local()
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(tmp_path, lambda *a: True))
    with patch.dict("localforge.tools.BACKENDS", {"stub": local}):
        d.dispatch("delegate_coding_task", {"instructions": "implement:\n```python\ndef slugify(text: str) -> str: ...\n```"})
    assert len(local.prompts) == 1


def test_the_orchestrator_is_told_its_four_jobs():
    from localforge.orchestrator import SYSTEM_PROMPT

    for job in ("plan the work", "delegate the writing", "validate what they produce", "reconcile"):
        assert job in SYSTEM_PROMPT
    assert "as good as if you had written all of it yourself" in SYSTEM_PROMPT
