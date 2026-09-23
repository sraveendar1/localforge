"""What passes between the orchestrator, the local models and memory.

Found on a review of delegation/memory/compaction: local calls ran in
Ollama's default few-thousand-token window (a big attached file silently
pushed the task's own instructions out), a Markdown file with code blocks
of its own was cut down to its first paragraph, output that hit the length
cap was written as half a file, the memory keeper never saw which tools the
orchestrator called on the API-key path, a nudge could pass for a user
turn, and /compact could download a model on a machine with none.
"""

import json
from unittest.mock import patch

import httpx
import pytest

import localforge.cli as cli_module
from localforge import memory
from localforge.backends import ollama
from localforge.backends.ollama import OllamaBackend
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.orchestrator import (
    NOTE_PREFIX,
    NUDGE,
    Conversation,
    _collapse_old_tool_results,
)
from localforge.tools import Dispatcher, _extract_file_content
from localforge.workspace import Workspace


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


CODER = [ModelEntry(name="coder", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1)]


class Local:
    def __init__(self, reply, truncated=False):
        self.reply, self.truncated, self.prompts = reply, truncated, []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.prompts.append(prompt)
        return {"type": "text", "content": self.reply, "tokens": 300, "truncated": self.truncated}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "api.py").write_text("def get_user(user_id: int) -> dict:\n    ...\n")
    return tmp_path


def _dispatcher(project, local):
    return Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project, lambda *a: True)), patch.dict(
        "localforge.tools.BACKENDS", {"stub": local}
    )


# --- the context window ----------------------------------------------------


def _capture(response_lines):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in response_lines).encode())

    def client(*_):
        return httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))

    return requests, client


@pytest.mark.parametrize("streaming", [True, False])
def test_every_local_call_asks_for_a_window_that_fits_its_prompt(streaming):
    lines = [{"response": "ok", "done": True, "eval_count": 1, "done_reason": "stop"}]
    backend = OllamaBackend()
    for size in (100, 30_000, 60_000):
        requests, client = _capture(lines)
        with patch.object(backend, "_client", client):
            if streaming:
                backend.generate("m", "x" * size, on_token=lambda _: None)
            else:
                backend.generate("m", "x" * size)
        options = requests[0]["options"]
        needed = ollama.estimate_tokens("x" * size) + options["num_predict"]
        assert options["num_ctx"] >= min(needed, ollama.max_context())
        assert options["num_ctx"] in (8192, 16384, 32768)  # a few fixed sizes, so the model isn't reloaded each call


def test_similar_prompts_share_one_window_size():
    assert ollama.context_size(100, 4096) == ollama.context_size(1000, 4096) == 8192
    assert ollama.context_size(9000, 8192) == 32768
    assert ollama.context_size(10**6, 8192) == ollama.max_context()  # capped


def test_a_quiet_call_has_an_output_cap_too():
    requests, client = _capture([{"response": "ok", "done": True}])
    backend = OllamaBackend()
    with patch.object(backend, "_client", client):
        backend.generate("m", "summarize this")
    assert requests[0]["options"]["num_predict"] == ollama.MAX_QUIET_OUTPUT_TOKENS


def test_output_that_hit_the_length_cap_is_reported_as_cut_off():
    _, client = _capture([{"response": "def f(", "done": False}, {"response": "", "done": True, "done_reason": "length"}])
    backend = OllamaBackend()
    with patch.object(backend, "_client", client):
        assert backend.generate("m", "p", on_token=lambda _: None)["truncated"] is True


def test_big_reference_files_are_cut_to_fit_and_the_orchestrator_is_told(project, monkeypatch):
    monkeypatch.setenv(ollama.MAX_NUM_CTX_ENV_VAR, "12000")  # room for ~11k chars of prompt beside the reply
    (project / "huge.py").write_text("y = 1\n" * 5000)  # 30k chars
    local = Local("summary " * 20)
    d, backends = _dispatcher(project, local)
    with backends:
        out = d.dispatch("delegate_coding_task", {"instructions": "summarize huge.py", "context_files": ["huge.py", "api.py"]})
    prompt = local.prompts[0]
    assert prompt.startswith("You are a focused coding assistant") and "summarize huge.py" in prompt  # the task survives
    assert len(prompt) <= ollama.prompt_budget()
    assert "huge.py was cut to its first" in out and "api.py was left out" in out


def test_a_file_too_big_to_rewrite_whole_is_never_sent_or_cut(project, monkeypatch):
    monkeypatch.setenv(ollama.MAX_NUM_CTX_ENV_VAR, "10000")
    (project / "big.py").write_text("z = 2\n" * 4000)
    local = Local("```python\nz = 3\n```")
    d, backends = _dispatcher(project, local)
    with backends:
        out = d.dispatch("delegate_coding_task", {"instructions": "change z", "path": "big.py"})
    assert out.startswith("delegate_coding_task failed:") and "too large" in out and "edit_file" in out
    assert local.prompts == []  # nothing sent...
    assert (project / "big.py").read_text() == "z = 2\n" * 4000  # ...and nothing lost


def test_a_long_current_file_is_followed_by_the_task_again(project):
    local = Local("```python\ndef get_user(user_id): return {}\n```")
    d, backends = _dispatcher(project, local)
    with backends:
        d.dispatch("delegate_coding_task", {"instructions": "implement it", "path": "api.py"})
    assert local.prompts[0].rstrip().endswith("Now write the complete new contents of `api.py`.")


# --- output that got cut off ------------------------------------------------


def test_cut_off_output_is_never_written(project):
    local = Local("```python\ndef get_user(user_id):\n    return {", truncated=True)
    d, backends = _dispatcher(project, local)
    with backends:
        out = d.dispatch("delegate_coding_task", {"instructions": "implement it", "path": "api.py"})
    assert out.startswith("[WARNING:") and "cut off" in out and "Nothing was written" in out
    assert len(local.prompts) == 1  # the same prompt would only be cut off again
    assert (project / "api.py").read_text() == "def get_user(user_id: int) -> dict:\n    ...\n"


def test_an_unclosed_code_block_counts_as_cut_off(project):
    local = Local("```python\ndef get_user(user_id):\n    return {")  # Ollama didn't say so, but it was
    d, backends = _dispatcher(project, local)
    with backends:
        out = d.dispatch("delegate_coding_task", {"instructions": "implement it", "path": "api.py"})
    assert out.startswith("[WARNING:") and "never closed" in out
    assert "return {" not in (project / "api.py").read_text()


# --- Markdown files with code blocks of their own ---------------------------


README = "# Tool\n\nInstall:\n\n```bash\npip install tool\n```\n\nUse:\n\n```python\nimport tool\n```\n\nThat's it.\n"


def test_a_readme_with_its_own_code_blocks_is_kept_whole():
    assert _extract_file_content(f"```markdown\n{README}```") == README
    assert _extract_file_content(f"```markdown\n{README}```\n") == README


def test_a_readme_delegation_writes_the_whole_readme(project):
    local = Local(f"```markdown\n{README}```")
    d, backends = _dispatcher(project, local)
    with backends:
        d.dispatch("delegate_coding_task", {"instructions": "write the README", "path": "README.md"})
    assert (project / "README.md").read_text() == README


def test_older_reply_shapes_still_extract():
    assert _extract_file_content("Here you go:\n```py\nx = 1\n```\nEnjoy") == "x = 1\n"
    assert _extract_file_content("plain text") == "plain text\n"
    assert _extract_file_content("```\nopened and never closed") is None


# --- what the memory keeper is shown ----------------------------------------


def test_the_keeper_sees_which_tools_were_called_on_the_api_path():
    messages = [
        {"role": "user", "content": "add a health check"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "delegate_coding_task", "arguments": '{"path": "health.py"}'}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "Created health.py (+10 -0, 10 lines)."},
    ]
    text = memory._render(messages)
    assert 'delegate_coding_task({"path": "health.py"})' in text
    assert "Created health.py" in text


def test_a_collapsed_result_keeps_how_the_step_went():
    messages = [{"role": "system", "content": "s"}] + [
        {"role": "tool", "tool_call_id": str(i), "content": f"Updated f{i}.py (+1 -1, 3 lines).\n--- diff ---\n" + "x" * 500}
        for i in range(6)
    ]
    _collapse_old_tool_results(messages, list(range(1, 7)))
    assert messages[1]["content"].startswith("[superseded:")
    assert "It began: Updated f0.py (+1 -1, 3 lines)." in messages[1]["content"]
    assert "x" * 100 not in messages[1]["content"]
    rendered = memory._render(messages[1:2])
    assert rendered == "[tool] Updated f0.py (+1 -1, 3 lines)."  # the keeper gets the outcome, not the placeholder


def test_nudges_and_notes_are_not_turns():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "build the API"},
        {"role": "assistant", "content": "I will create it."},
        {"role": "user", "content": NUDGE},
        {"role": "user", "content": NOTE_PREFIX + "use FastAPI"},
    ]
    assert memory._turn_starts(messages) == [1, 3]


def test_compaction_keeps_the_real_request_not_the_nudge(project, monkeypatch):
    conv = Conversation(
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "first task"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "build the API"},
            {"role": "assistant", "content": "I will create it."},
            {"role": "user", "content": NUDGE},
        ]
    )
    monkeypatch.setattr(memory, "_summarize", lambda *a: "summary")
    d = Dispatcher(_hw(), catalog=CODER, workspace=Workspace(project))
    assert memory.compact(conv, d, root=project)
    assert conv.messages[1]["content"] == "build the API"


# --- /compact never downloads -----------------------------------------------


def test_compact_with_nothing_installed_never_downloads(monkeypatch):
    pulls = []

    class NoModels(Local):
        def ensure_available(self, name, on_progress=None):
            pulls.append(name)

    monkeypatch.setattr(cli_module, "_installed_model_names", lambda _backend: set())  # Ollama up, nothing on disk
    monkeypatch.setattr(cli_module, "detect_hardware", _hw)
    cli_module._session.conversation = Conversation(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    with patch.dict("localforge.memory.BACKENDS", {"ollama": NoModels("x")}):
        cli_module.compact()
    assert pulls == []


# --- long tasks carry on by themselves ---------------------------------------


def _orchestrator_stub(calls_per_round):
    """A frontier model that calls tools: `calls_per_round(n)` gives round
    n's arguments, or None to give the final answer instead."""
    from unittest.mock import MagicMock

    import localforge.orchestrator as orch

    seen = {"rounds": 0, "prompts": []}

    def completion(**kw):
        seen["rounds"] += 1
        seen["prompts"].append(kw["messages"][-1].get("content"))
        args = calls_per_round(seen["rounds"])
        msg = MagicMock()
        if args is None:
            msg.tool_calls, msg.content = None, "all done"
            msg.model_dump.return_value = {"role": "assistant", "content": "all done"}
        else:
            call = MagicMock(id=f"c{seen['rounds']}")
            call.function.name, call.function.arguments = "read_file", json.dumps(args)
            msg.tool_calls, msg.content = [call], None
            msg.model_dump.return_value = {"role": "assistant", "content": None}
        resp = MagicMock()
        resp.choices, resp.usage = [MagicMock(message=msg)], None
        return resp

    class Stub:
        def __init__(self, *a, **kw):
            self.catalog, self.local_tokens_generated = [], 0

        def dispatch(self, name, args, on_delegate=None):
            return f"contents of {args['path']}"

    patches = [
        patch.object(orch, "completion", side_effect=completion),
        patch.object(orch, "Dispatcher", Stub),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=set()),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ]
    return seen, patches


def _run_with(patches, conversation=None):
    import contextlib

    import localforge.orchestrator as orch

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return orch.run("big task", "claude-opus-5", hardware=_hw(), conversation=conversation)


def test_a_long_task_that_is_making_progress_passes_the_checkpoint_by_itself():
    import localforge.orchestrator as orch

    finish_at = orch.CHECKPOINT_EVERY + 5
    seen, patches = _orchestrator_stub(lambda n: None if n == finish_at else {"path": f"f{n}.py"})
    result = _run_with(patches)
    assert result.answer == "all done"  # no "did not converge", no typing "continue"
    assert seen["rounds"] == finish_at
    assert seen["prompts"][orch.CHECKPOINT_EVERY].startswith(orch.CHECKPOINT_PREFIX)  # the frontier model was asked


def test_a_task_going_in_circles_stops_at_the_checkpoint_and_can_be_continued():
    import localforge.orchestrator as orch
    from localforge.orchestrator import OrchestrationError

    seen, patches = _orchestrator_stub(lambda n: {"path": "same.py"})  # re-reads one file forever
    conv = Conversation()
    with pytest.raises(OrchestrationError, match="looked stuck") as excinfo:
        _run_with(patches, conv)
    assert seen["rounds"] == 2 * orch.CHECKPOINT_EVERY  # the first stretch had one real step, the second none
    assert 'type "continue"' in str(excinfo.value)
    assert conv.messages[-1]["role"] == "assistant"  # the history ends cleanly, ready for "continue"


def test_the_ceiling_is_announced_before_it_is_reached():
    import localforge.orchestrator as orch

    seen, patches = _orchestrator_stub(lambda n: {"path": f"f{n}.py"})
    with pytest.raises(orch.OrchestrationError):
        _run_with(patches)
    assert seen["rounds"] == orch.MAX_ROUNDS
    assert seen["prompts"][orch.MAX_ROUNDS - orch.WRAP_UP_ROUNDS].startswith(orch.STEPS_LEFT_PREFIX)


def test_a_collapsed_result_names_the_call_it_came_from():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "function": {"name": "read_file", "arguments": '{"path": "src/StatusBar.tsx"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "1: import React\n" + "x" * 300},
    ] + [{"role": "tool", "tool_call_id": f"t{i}", "content": "r"} for i in range(4)]
    _collapse_old_tool_results(messages, [2, 3, 4, 5, 6])
    assert "of read_file src/StatusBar.tsx" in messages[2]["content"]
