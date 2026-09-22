"""When a delegated step fails, the flow must keep going: the orchestrator
gets to see the error and retry or work around it, and the dispatcher makes
its own recovery attempt first.

Reported: "when the orchestrator delegates and some issue happens, the
flow stops". Root cause: the repeat-call guard cached *failed* results too,
so a sensible retry of the same call was answered with "You already made
this exact call ... give your final answer".
"""

from unittest.mock import MagicMock, patch

import pytest

import localforge.orchestrator as orch
from localforge.catalog import ModelEntry
from localforge.hardware import HardwareProfile
from localforge.tools import Dispatcher


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def _tool_reply(name, args):
    msg = MagicMock()
    msg.tool_calls = [MagicMock(id="c", function=MagicMock(arguments=args))]
    msg.tool_calls[0].function.name = name
    msg.model_dump.return_value = {"role": "assistant", "content": None}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


def _final(text="done"):
    msg = MagicMock(tool_calls=None, content=text)
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


class FlakyDispatcher:
    """Raises on the first N calls, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0
        self.catalog, self.local_tokens_generated = [], 0

    def dispatch(self, name, args, on_delegate=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("ollama: read timed out")
        return "wrote the file"


def _run(replies, dispatcher):
    conv = orch.Conversation()
    with (
        patch.object(orch, "completion", side_effect=lambda **kw: next(replies)),
        patch.object(orch, "Dispatcher", lambda *a, **kw: dispatcher),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        result = orch.run("build it", "gpt-5", hardware=_hw(), conversation=conv)
    return result, conv


def test_retrying_a_failed_call_actually_runs_it_again():
    call = ("delegate_coding_task", '{"instructions": "write x", "path": "x.py"}')
    replies = iter([_tool_reply(*call), _tool_reply(*call), _final()])
    flaky = FlakyDispatcher(fail_times=1)
    result, conv = _run(replies, flaky)
    assert flaky.calls == 2  # the retry reached the dispatcher
    assert result.answer == "done"
    assert not any("already made this exact" in str(m.get("content")) for m in conv.messages)


def test_a_call_that_keeps_failing_is_stopped_with_a_useful_message():
    call = ("delegate_coding_task", '{"instructions": "write x"}')
    replies = iter([_tool_reply(*call)] * 4 + [_final("blocked: explained to the user")])
    flaky = FlakyDispatcher(fail_times=99)
    result, conv = _run(replies, flaky)
    assert flaky.calls == orch.MAX_ATTEMPTS_PER_CALL
    last_tool = [m["content"] for m in conv.messages if m.get("role") == "tool"][-1]
    assert f"failed {orch.MAX_ATTEMPTS_PER_CALL} times" in last_tool and "ollama: read timed out" in last_tool
    assert "different approach" in last_tool


def test_errors_are_explained_to_the_orchestrator_with_next_steps():
    call = ("delegate_coding_task", '{"instructions": "write x"}')
    replies = iter([_tool_reply(*call), _final()])
    _, conv = _run(replies, FlakyDispatcher(fail_times=1))
    tool = [m["content"] for m in conv.messages if m.get("role") == "tool"][0]
    assert "ollama: read timed out" in tool and "retry" in tool.lower()


def test_system_prompt_says_to_unblock_rather_than_stop():
    assert "When a step fails" in orch.SYSTEM_PROMPT
    assert "don't stop at the first error" in orch.SYSTEM_PROMPT.lower()


# --- the dispatcher recovers on its own first ------------------------------------------


CATALOG = [
    ModelEntry(name="big", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=2),
    ModelEntry(name="small", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=1, quality_tier=1),
]


class Backend:
    def __init__(self, failing):
        self.failing, self.calls = set(failing), []

    def ensure_available(self, name, on_progress=None):
        pass

    def generate(self, name, prompt, on_token=None, **kw):
        self.calls.append((name, prompt))
        if name in self.failing:
            raise RuntimeError(f"{name}: model requires more system memory than is available")
        return {"type": "text", "content": f"code from {name} " + "x" * 40, "tokens": 5}


def test_a_crashing_local_model_is_swapped_for_another_installed_one():
    backend = Backend(failing={"big"})
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"big", "small"})
    with patch.dict("localforge.tools.BACKENDS", {"stub": backend}):
        out = d.dispatch("delegate_coding_task", {"instructions": "write x"})
    assert out.startswith("code from small")
    assert [n for n, _ in backend.calls] == ["big", "small"]


def test_with_one_model_the_retry_mentions_the_error():
    backend = Backend(failing=set())
    calls = {"n": 0}
    real_generate = backend.generate

    def flaky(name, prompt, on_token=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection reset")
        return real_generate(name, prompt)

    backend.generate = flaky
    d = Dispatcher(_hw(), catalog=CATALOG[:1], installed={"big"})
    with patch.dict("localforge.tools.BACKENDS", {"stub": backend}):
        out = d.dispatch("delegate_coding_task", {"instructions": "write x"})
    assert out.startswith("code from big")
    assert "connection reset" in backend.calls[-1][1]  # told what went wrong last time


def test_when_every_attempt_fails_the_error_says_what_was_tried():
    backend = Backend(failing={"big", "small"})
    d = Dispatcher(_hw(), catalog=CATALOG, installed={"big", "small"})
    with patch.dict("localforge.tools.BACKENDS", {"stub": backend}):
        with pytest.raises(RuntimeError) as exc:
            d.dispatch("delegate_coding_task", {"instructions": "write x"})
    msg = str(exc.value)
    assert "big" in msg and "small" in msg and "more system memory" in msg


# --- a stuck local model can't hang the task --------------------------------------------


def _ollama_stream(chunks, **final):
    import json

    import httpx

    lines = [json.dumps({"response": c, "done": False}) for c in chunks] + [json.dumps({"response": "", "done": True, **final})]

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content="\n".join(lines).encode())

    seen = {}
    return handler, seen


def test_a_model_repeating_itself_is_stopped():
    import httpx

    from localforge.backends.ollama import LocalModelStuck, OllamaBackend

    loop = "and then the function calls itself again, "  # 43 chars, varied
    handler, _ = _ollama_stream(["def f():\n"] + [loop] * 200)
    backend = OllamaBackend()
    with patch.object(backend, "_client", lambda *a: httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))):
        with pytest.raises(LocalModelStuck, match="repeating"):
            backend.generate("m", "p", on_token=lambda c: None)


def test_normal_code_with_divider_lines_is_not_mistaken_for_a_loop():
    import httpx

    from localforge.backends.ollama import OllamaBackend

    chunks = [f"# {'=' * 130}\nprint({i})\n" for i in range(120)]
    handler, seen = _ollama_stream(chunks, eval_count=500)
    backend = OllamaBackend()
    with patch.object(backend, "_client", lambda *a: httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))):
        result = backend.generate("m", "p", on_token=lambda c: None)
    assert result["tokens"] == 500
    assert seen["body"]["options"]["num_predict"] == 8192  # output is capped


def test_is_repeating_edges():
    from localforge.backends.ollama import _is_repeating

    assert _is_repeating("start " + "I will now write the code for you. " * 40)
    assert not _is_repeating("x" * 5000)  # too few distinct characters to be a loop
    assert not _is_repeating("\n".join(f"line {i}: value = {i * 7}" for i in range(300)))


# --- a hiccup in the orchestrator call itself -------------------------------------------


def test_a_transient_orchestrator_failure_is_retried_once(monkeypatch):
    from localforge import cli_transport

    monkeypatch.setattr(orch, "RETRY_DELAY_SECONDS", 0)
    calls = {"n": 0}

    def flaky_cli(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cli_transport.CLINotAvailableError("claude exited 1: Stream closed unexpectedly")
        return cli_transport.CLIResponse(choices=[cli_transport._Choice(message=cli_transport.message_from_reply('{"final_answer": "ok"}'))])

    retries = []
    with patch.object(orch.cli_transport, "complete", side_effect=flaky_cli), patch.object(orch, "_installed_models", return_value=None):
        result = orch.run("hi", "claude-opus-5", hardware=_hw(), cli_provider="anthropic", hooks=orch.ActivityHooks(on_tool=lambda t, s: retries.append(t)))
    assert result.answer == "ok" and calls["n"] == 2 and retries == ["retry"]


@pytest.mark.parametrize(
    "message",
    [
        "claude exited 1: You've hit your monthly spend limit",
        "claude exited 1: Invalid API key · Please run /login",
        "The 'claude' CLI is required for anthropic CLI login.",
    ],
)
def test_limits_and_sign_in_problems_are_not_retried(monkeypatch, message):
    from localforge import cli_transport

    monkeypatch.setattr(orch, "RETRY_DELAY_SECONDS", 0)
    with patch.object(orch.cli_transport, "complete", side_effect=cli_transport.CLINotAvailableError(message)) as call, patch.object(
        orch, "_installed_models", return_value=None
    ):
        with pytest.raises(cli_transport.CLINotAvailableError):
            orch.run("hi", "claude-opus-5", hardware=_hw(), cli_provider="anthropic")
    assert call.call_count == 1


# --- Ctrl+C stops the task, not the session ------------------------------------------------


def test_ctrl_c_mid_task_leaves_a_usable_conversation():
    replies = iter([_tool_reply("read_file", '{"path": "a.py"}')])

    class Interrupting(FlakyDispatcher):
        def dispatch(self, name, args, on_delegate=None):
            raise KeyboardInterrupt

    conv = orch.Conversation()
    with (
        patch.object(orch, "completion", side_effect=lambda **kw: next(replies)),
        patch.object(orch, "Dispatcher", lambda *a, **kw: Interrupting(0)),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        with pytest.raises(orch.TaskCancelled):
            orch.run("read a.py", "gpt-5", hardware=_hw(), conversation=conv)
    roles = [m["role"] for m in conv.messages]
    assert roles == ["system", "user", "assistant"]  # no dangling tool call
    assert "stopped this task" in conv.messages[-1]["content"]


def test_ctrl_c_in_a_session_keeps_the_session(monkeypatch):
    from typer.testing import CliRunner

    import localforge.cli as cli_module

    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)
    with patch.object(cli_module, "run_orchestrator", side_effect=orch.TaskCancelled(orch.RunStats(frontier_prompt_tokens=5))):
        result = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"])
    assert result.exit_code == 130 and "Stopped." in result.output
    assert cli_module._session_usage[-1][1].frontier_prompt_tokens == 5  # still counted in /usage
