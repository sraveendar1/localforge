"""The orchestrator's answer streams to the screen as it's written, like the
local models' output already did. Covers the JSON-answer extractor, the
three transports (CLI login, local Ollama, API key) and the terminal side.
"""

import json
import sys
import textwrap
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import cli_transport, config, local_transport
from localforge.answer_stream import AnswerStreamer
from localforge.hardware import HardwareProfile
from localforge.orchestrator import RunResult, RunStats


def _hw():
    return HardwareProfile(os="Linux", arch="x86_64", cpu_cores=8, ram_gb=32, free_disk_gb=100, gpus=[])


def _collect(chunks):
    out = []
    streamer = AnswerStreamer(out.append)
    for c in chunks:
        streamer.feed(c)
    return "".join(out), out


# --- pulling the answer out of streamed JSON -------------------------------------------


def test_live_sample_from_claude_decodes_including_a_split_escape():
    # the exact deltas `claude -p --output-format stream-json` produced live
    deltas = ['```json\n{"final_', 'answer": "Hello there,\\', 'nthis is a \\"streamed', '\\" test answer with several words in it', '."}\n```']
    text, pieces = _collect(deltas)
    assert text == 'Hello there,\nthis is a "streamed" test answer with several words in it.'
    assert len(pieces) >= 3  # arrived in pieces, not all at the end


def test_unicode_escapes_even_when_split():
    text, _ = _collect(['{"final_answer": "caf\\u00', 'e9 \\u2713 done"}'])
    assert text == "café ✓ done"


def test_tool_call_replies_stream_nothing():
    text, pieces = _collect(['{"tool_calls": [{"name": "read_file", ', '"arguments": {"path": "final_answer.md"}}]}'])
    assert text == "" and pieces == []


def test_one_char_at_a_time():
    raw = json.dumps({"final_answer": 'Line 1\n\tLine "2" \\ end'})
    text, _ = _collect(list(raw))
    assert text == 'Line 1\n\tLine "2" \\ end'


# --- CLI login: real subprocess, fake CLI -----------------------------------------------


@pytest.fixture
def fake_claude(tmp_path):
    """A stand-in `claude` that reads the prompt on stdin and prints the same
    stream-json event lines the real one produced in a live sample."""
    script = tmp_path / "fake_claude.py"
    script.write_text(textwrap.dedent('''
        import json, sys, time
        prompt = sys.stdin.read()
        assert "Decide the next step" in prompt
        print(json.dumps({"type": "system", "subtype": "init"}), flush=True)
        for d in ['{"final_', 'answer": "Streamed ', 'hello, \\\\"', 'world\\\\""}']:
            print(json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": d}}}), flush=True)
            time.sleep(0.01)
        print(json.dumps({"type": "result", "subtype": "success", "result": '{"final_answer": "Streamed hello, \\\\"world\\\\""}', "usage": {"input_tokens": 12, "output_tokens": 7}, "total_cost_usd": 0.001}), flush=True)
    '''))
    return script


def test_cli_transport_streams_through_a_real_subprocess(fake_claude, monkeypatch):
    spec = dict(config.FRONTIER_CLI_AUTH["anthropic"], command=sys.executable, headless_args=[str(fake_claude)], isolation_args=[], model_flag=None)
    monkeypatch.setitem(config.FRONTIER_CLI_AUTH, "anthropic", spec)
    monkeypatch.setattr(cli_transport, "available", lambda p: True)
    pieces = []
    resp = cli_transport.complete("anthropic", [{"role": "user", "content": "hi"}], [], on_text=pieces.append)

    assert "".join(pieces) == 'Streamed hello, "world"'
    assert len(pieces) >= 2
    assert resp.choices[0].message.content == 'Streamed hello, "world"'
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (12, 7)
    assert resp.notional_cost_usd == 0.001


def test_cli_streaming_uses_stream_json_and_json_when_not_streaming(monkeypatch):
    monkeypatch.setattr(cli_transport, "available", lambda p: True)
    seen = {}

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        proc = MagicMock()
        proc.stdout = iter([json.dumps({"type": "result", "result": '{"final_answer": "x"}', "usage": {}}) + "\n"])
        proc.stderr.read.return_value = ""
        proc.returncode = 0
        return proc

    with patch.object(cli_transport.subprocess, "Popen", fake_popen):
        cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [], on_text=lambda t: None)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--include-partial-messages" in cmd and "--verbose" in cmd
    assert cmd[cmd.index("--tools") + 2].startswith("--")  # variadic --tools still followed by an option


def test_cli_streaming_times_out(tmp_path, monkeypatch):
    slow = tmp_path / "slow.py"
    slow.write_text("import sys, time\nsys.stdin.read()\ntime.sleep(30)\n")
    spec = dict(config.FRONTIER_CLI_AUTH["anthropic"], command=sys.executable, headless_args=[str(slow)], isolation_args=[], model_flag=None)
    monkeypatch.setitem(config.FRONTIER_CLI_AUTH, "anthropic", spec)
    monkeypatch.setattr(cli_transport, "available", lambda p: True)
    with pytest.raises(cli_transport.CLINotAvailableError, match="timed out"):
        cli_transport.complete("anthropic", [{"role": "user", "content": "x"}], [], timeout=1, on_text=lambda t: None)


# --- local orchestrator ------------------------------------------------------------------


def test_local_transport_streams_the_answer():
    events = [
        {"message": {"content": '{"final_'}},
        {"message": {"content": 'answer": "local '}},
        {"message": {"content": 'answer"}'}},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 50, "eval_count": 9},
    ]

    def handler(request):
        if request.url.path == "/api/chat":
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(200, content="\n".join(json.dumps(e) for e in events).encode())
        return httpx.Response(200, json={})

    real = httpx.Client
    pieces = []
    with (
        patch.object(local_transport.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)),
        patch.object(local_transport.OllamaBackend, "is_running", return_value=True),
        patch.object(local_transport.OllamaBackend, "ensure_available"),
    ):
        resp = local_transport.complete("ollama/m", [{"role": "user", "content": "hi"}], [], on_text=pieces.append)
    assert "".join(pieces) == "local answer" and len(pieces) == 2
    assert resp.choices[0].message.content == "local answer"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (50, 9)


# --- API key path --------------------------------------------------------------------------


def test_api_path_streams_native_text_and_rebuilds_the_response():
    def chunk(text):
        return MagicMock(choices=[MagicMock(delta=MagicMock(content=text))])

    built = MagicMock()
    built.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Hi there"))]
    built.choices[0].message.model_dump.return_value = {"role": "assistant", "content": "Hi there"}
    pieces = []
    with (
        patch.object(orch, "completion", return_value=iter([chunk("Hi "), chunk("there"), chunk(None)])) as completion,
        patch.object(orch.litellm, "stream_chunk_builder", return_value=built) as builder,
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
        patch.object(orch, "_installed_models", return_value=None),
    ):
        result = orch.run("hi", "gpt-5", hardware=_hw(), hooks=orch.ActivityHooks(on_answer_text=pieces.append))

    assert pieces == ["Hi ", "there"] and result.answer == "Hi there"
    assert completion.call_args.kwargs["stream"] is True
    assert completion.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert len(builder.call_args.args[0]) == 3


# --- what the user sees ---------------------------------------------------------------------


def test_a_streamed_answer_is_not_printed_twice(monkeypatch):
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)

    def fake_run(task, frontier_model, hooks=None, **kw):
        hooks.on_frontier(1)
        for piece in ["## Done\n", "- wrote **hello.py**"]:
            hooks.on_answer_text(piece)
        return RunResult("## Done\n- wrote **hello.py**", RunStats())

    with patch.object(cli_module, "run_orchestrator", side_effect=fake_run):
        out = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"]).output
    assert out.count("wrote hello.py") == 1
    assert "## Done" not in out and "**" not in out  # rendered as markdown


def test_an_unstreamed_answer_is_still_printed(monkeypatch):
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)

    def fake_run(task, frontier_model, hooks=None, **kw):
        hooks.on_frontier(1)
        return RunResult("plain answer", RunStats())

    with patch.object(cli_module, "run_orchestrator", side_effect=fake_run):
        out = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"]).output
    assert out.count("plain answer") == 1


def test_streamed_text_before_a_tool_call_stays_and_the_final_answer_follows(monkeypatch):
    monkeypatch.delenv("LOCALFORGE_AUTH_METHOD", raising=False)

    def fake_run(task, frontier_model, hooks=None, **kw):
        hooks.on_frontier(1)
        hooks.on_answer_text("Let me look at the files first.")
        hooks.on_tool("read_file", "app.py")
        hooks.on_tool_result("read_file", "app.py (3 lines)")
        hooks.on_frontier(2)
        hooks.on_answer_text("All good.")
        return RunResult("All good.", RunStats())

    with patch.object(cli_module, "run_orchestrator", side_effect=fake_run):
        out = CliRunner().invoke(cli_module.app, ["run", "t", "-m", "gpt-5"]).output
    assert out.index("Let me look") < out.index("Read app.py") < out.index("All good.")
    assert out.count("All good.") == 1
