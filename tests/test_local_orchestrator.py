"""A local (Ollama) model as the orchestrator must never touch a provider CLI.

Reported on a second laptop: with a local model chosen as the orchestrator,
localforge still ran `claude` ("Orchestrating via your `claude` login") and
failed with the Claude account's spend limit, then advised `claude login`.
Cause: choosing the local provider saved only the model id, and
config.save() merges, so the earlier setup's LOCALFORGE_AUTH_METHOD=cli_login
survived and kept routing every turn through `claude`.
"""

import json
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

import localforge.cli as cli_module
import localforge.orchestrator as orch
from localforge import config, local_transport
from localforge.hardware import HardwareProfile
from localforge.orchestrator import RunResult, RunStats
from localforge.workspace import Workspace

SPEND_LIMIT = (
    "claude exited 1: You've hit your monthly spend limit · raise it at "
    "claude.ai/settings/usage?from=cc_cli_limit_message · your session limit resets 11:10pm (UTC)"
)


def _hw() -> HardwareProfile:
    return HardwareProfile(os="Darwin", arch="arm64", cpu_cores=10, ram_gb=16, free_disk_gb=50, gpus=[])


@pytest.fixture
def stale_claude_login(monkeypatch):
    monkeypatch.setenv(config.AUTH_METHOD_ENV_VAR, config.AUTH_CLI_LOGIN)
    monkeypatch.setenv(config.FRONTIER_PROVIDER_ENV_VAR, "anthropic")


# --- routing ---------------------------------------------------------------------


def test_a_local_model_never_routes_through_a_provider_cli(stale_claude_login):
    assert cli_module._cli_provider_for("ollama/gemma3:4b", explicit=False) is None
    assert cli_module._cli_provider_for("claude-opus-5", explicit=False) == "anthropic"
    assert cli_module._cli_provider_for("claude-opus-5", explicit=True) is None  # --model means the API path


def test_run_with_a_saved_local_model_skips_claude_entirely(stale_claude_login, monkeypatch):
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "ollama/gemma3:4b")
    config.save({config.FRONTIER_MODEL_ENV_VAR: "ollama/gemma3:4b"})
    with patch.object(cli_module, "run_orchestrator", return_value=RunResult("hi!", RunStats())) as run, patch.object(
        cli_module.cli_transport, "available"
    ) as available:
        out = CliRunner().invoke(cli_module.app, ["run", "hi"]).output

    assert run.call_args.kwargs["cli_provider"] is None
    available.assert_not_called()
    assert "claude" not in out
    assert "gemma3:4b (local, via Ollama" in out


def test_orchestrator_sends_local_models_to_ollama_even_if_a_cli_is_passed():
    sent = {}

    def fake_local(model, messages, tools):
        sent["model"] = model
        return local_transport.cli_transport.CLIResponse(
            choices=[local_transport.cli_transport._Choice(message=local_transport.cli_transport.message_from_reply('{"final_answer": "done"}'))],
            local=True,
        )

    with (
        patch.object(orch.local_transport, "complete", side_effect=fake_local),
        patch.object(orch.cli_transport, "complete") as cli_complete,
        patch.object(orch, "_installed_models", return_value=None),
    ):
        result = orch.run("hi", "ollama/gemma3:4b", hardware=_hw(), cli_provider="anthropic")

    cli_complete.assert_not_called()
    assert sent["model"] == "ollama/gemma3:4b" and result.answer == "done"
    assert result.stats.frontier_via_subscription is False and result.stats.frontier_cost_usd == 0


# --- saving the choice actually replaces the old one ------------------------------------


def test_setup_choosing_local_overwrites_a_previous_claude_login(monkeypatch):
    config.save({config.FRONTIER_MODEL_ENV_VAR: "claude-opus-5", config.AUTH_METHOD_ENV_VAR: config.AUTH_CLI_LOGIN, config.FRONTIER_PROVIDER_ENV_VAR: "anthropic"})
    answers = "4\n1\n"  # provider 4 = local, model 1
    with (
        patch.object(cli_module, "shutil") as mock_shutil,
        patch.object(cli_module.OllamaBackend, "is_running", return_value=True),
        patch.object(cli_module.OllamaBackend, "ensure_available"),
        patch.object(cli_module, "_installed_model_names", return_value=set()),
        patch.object(cli_module, "detect_hardware", return_value=_hw()),
        patch.object(cli_module, "recommend_models", return_value={}),
    ):
        mock_shutil.which.return_value = "/usr/bin/ollama"
        CliRunner().invoke(cli_module.app, ["setup"], input=answers)

    saved = config.CONFIG_FILE.read_text()
    assert "LOCALFORGE_AUTH_METHOD=local" in saved and "cli_login" not in saved
    assert "LOCALFORGE_FRONTIER_PROVIDER=local" in saved


def test_config_save_applies_to_the_running_session():
    config.save({config.AUTH_METHOD_ENV_VAR: config.AUTH_LOCAL})
    import os

    assert os.environ[config.AUTH_METHOD_ENV_VAR] == config.AUTH_LOCAL


# --- the local transport ---------------------------------------------------------------


def test_local_transport_asks_ollama_for_json_with_a_big_context_and_parses_tools():
    seen = {}

    def handler(request: httpx.Request):
        if request.url.path == "/api/chat":
            seen.update(json.loads(request.content))
            reply = {"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}
            return httpx.Response(200, json={"message": {"content": json.dumps(reply)}, "prompt_eval_count": 900, "eval_count": 30})
        return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}], "version": "x"})

    real = httpx.Client
    with (
        patch.object(local_transport.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)),
        patch.object(local_transport.OllamaBackend, "is_running", return_value=True),
        patch.object(local_transport.OllamaBackend, "ensure_available"),
    ):
        resp = local_transport.complete("ollama/gemma3:4b", [{"role": "user", "content": "read a.py"}], [])

    assert seen["model"] == "gemma3:4b" and seen["format"] == "json"
    assert seen["options"]["num_ctx"] >= 16384
    call = resp.choices[0].message.tool_calls[0]
    assert call.function.name == "read_file" and json.loads(call.function.arguments) == {"path": "a.py"}
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (900, 30) and resp.local


def test_local_transport_explains_when_ollama_is_down():
    with patch.object(local_transport.OllamaBackend, "is_running", return_value=False):
        with pytest.raises(local_transport.LocalOrchestratorError, match="Ollama is not running"):
            local_transport.complete("ollama/x", [], [])


# --- honest error messages -------------------------------------------------------------


def test_a_spend_limit_is_not_reported_as_an_expired_login():
    advice = cli_module._cli_failure_advice("anthropic", SPEND_LIMIT)
    assert "usage limit" in advice and "/model" in advice
    assert "claude login" not in advice


def test_run_reports_the_spend_limit_with_the_right_fix(monkeypatch):
    monkeypatch.setenv(config.AUTH_METHOD_ENV_VAR, config.AUTH_CLI_LOGIN)
    monkeypatch.setenv(config.FRONTIER_PROVIDER_ENV_VAR, "anthropic")
    monkeypatch.setenv(config.FRONTIER_MODEL_ENV_VAR, "claude-opus-5")
    with patch.object(cli_module.cli_transport, "available", return_value=True), patch.object(
        cli_module, "run_orchestrator", side_effect=cli_module.cli_transport.CLINotAvailableError(SPEND_LIMIT)
    ):
        out = " ".join(CliRunner().invoke(cli_module.app, ["run", "hi"]).output.split())
    assert "hit a usage limit" in out and "/model" in out
    assert "session may have expired" not in out


# --- /model ----------------------------------------------------------------------------


def test_model_lists_installed_local_models_and_switches(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch.object(cli_module, "_installed_model_names", return_value={"gemma3:4b"}), patch.object(
        cli_module.cli_transport, "available", return_value=False
    ):
        listing = CliRunner().invoke(cli_module.app, ["model"]).output
        switched = CliRunner().invoke(cli_module.app, ["model", "ollama/gemma3:4b"]).output

    assert "ollama/gemma3:4b  (local — no account, no billing)" in listing
    assert "Orchestrator is now gemma3:4b (local" in switched
    saved = config.CONFIG_FILE.read_text()
    assert "LOCALFORGE_FRONTIER_MODEL=ollama/gemma3:4b" in saved and "LOCALFORGE_AUTH_METHOD=local" in saved


def test_model_refuses_a_local_model_that_isnt_downloaded():
    with patch.object(cli_module, "_installed_model_names", return_value=set()):
        result = CliRunner().invoke(cli_module.app, ["model", "ollama/qwen2.5:7b"])
    assert result.exit_code == 1 and "ollama pull qwen2.5:7b" in result.output


# --- setup's local menu fits the machine -----------------------------------------------


@pytest.mark.real_local_menu
def test_local_orchestrator_choices_put_installed_first_and_fit_the_hardware():
    choices = local_transport.orchestrator_choices(_hw(), {"gemma3:4b"})
    assert choices[0] == "ollama/gemma3:4b"
    assert "ollama/llama3.1:70b" not in choices and "ollama/qwen2.5:72b" not in choices  # won't run on 16 GB
    assert len(choices) > 1  # plus models that fit and could be downloaded


# --- home folder --------------------------------------------------------------------------


def test_working_in_the_home_folder_is_flagged_to_the_orchestrator(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "project").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    assert "home folder" in Workspace(home).snapshot()
    assert "home folder" not in Workspace(home / "project").snapshot()


# --- small-model safety rails ------------------------------------------------------------


def test_an_identical_repeated_tool_call_is_not_run_again():
    """Seen live with gemma3:4b: the same file delegated three times."""
    from unittest.mock import MagicMock

    def tool_response(args):
        msg = MagicMock()
        msg.tool_calls = [MagicMock(id="c", function=MagicMock(arguments=args))]
        msg.tool_calls[0].function.name = "read_file"
        msg.model_dump.return_value = {"role": "assistant", "content": None}
        return MagicMock(choices=[MagicMock(message=msg)], usage=None)

    final = MagicMock()
    final.tool_calls, final.content = None, "done"
    final.model_dump.return_value = {"role": "assistant", "content": "done"}
    responses = iter([
        tool_response('{"path": "a.py"}'),
        tool_response('{ "path":"a.py" }'),  # same call, different spacing
        MagicMock(choices=[MagicMock(message=final)], usage=None),
    ])

    class Recorder:
        def __init__(self, *a, **kw):
            self.catalog, self.local_tokens_generated, self.calls = [], 0, []

        def dispatch(self, name, args, on_delegate=None):
            self.calls.append((name, args))
            return "contents of a.py"

    recorder = Recorder()
    with (
        patch.object(orch, "completion", side_effect=lambda **kw: next(responses)),
        patch.object(orch, "Dispatcher", lambda *a, **kw: recorder),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        conv = orch.Conversation()
        orch.run("read a.py", "gpt-5", hardware=_hw(), conversation=conv)

    assert recorder.calls == [("read_file", {"path": "a.py"})]
    assert any("already made this exact read_file call" in str(m.get("content")) for m in conv.messages)


def test_system_prompt_says_to_only_do_what_was_asked():
    assert "Do only what the user asked" in orch.SYSTEM_PROMPT
    assert "greeting" in orch.SYSTEM_PROMPT


# --- no surprise downloads mid-task ----------------------------------------------------

from localforge.catalog import ModelEntry  # noqa: E402
from localforge.tools import Dispatcher  # noqa: E402

OLLAMA_TALKER = [ModelEntry(name="talker:3b", modality="general", runtime="ollama", min_vram_gb=0, min_ram_gb=4, disk_gb=2, quality_tier=1)]
TEXT_CATALOG = [
    ModelEntry(name="coder:4b", modality="coding", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=3, quality_tier=1),
    ModelEntry(name="talker:3b", modality="general", runtime="stub", min_vram_gb=0, min_ram_gb=4, disk_gb=2, quality_tier=1),
]


class _Stub:
    def __init__(self):
        self.pulled = []

    def ensure_available(self, name, on_progress=None):
        self.pulled.append(name)

    def generate(self, name, prompt, on_token=None, **kw):
        return {"type": "text", "content": f"written by {name}: " + "x" * 50, "tokens": 3}


def test_an_installed_text_model_stands_in_instead_of_downloading():
    """Seen live: a "general" subtask downloaded qwen2.5:3b although the
    installed coder could have written it."""
    stub = _Stub()
    d = Dispatcher(_hw(), catalog=TEXT_CATALOG, installed={"coder:4b"})
    with patch.dict("localforge.tools.BACKENDS", {"stub": stub}):
        out = d.dispatch("delegate_general_task", {"instructions": "write a greeting"})
    assert out.startswith("written by coder:4b")


def test_a_needed_download_asks_first(tmp_path):
    asked = []

    def approver(kind, title, detail):
        asked.append((kind, title))
        return False

    d = Dispatcher(_hw(), catalog=OLLAMA_TALKER, installed=set(), workspace=Workspace(tmp_path, approver))
    stub = _Stub()
    with patch.dict("localforge.tools.BACKENDS", {"ollama": stub}):
        with pytest.raises(RuntimeError, match="download wasn't approved"):
            d.dispatch("delegate_general_task", {"instructions": "hi"})
    assert asked == [("download", "Download talker:3b")]
    assert stub.pulled == []  # nothing fetched


def test_an_approved_download_proceeds_once(tmp_path):
    d = Dispatcher(_hw(), catalog=OLLAMA_TALKER, installed=set(), workspace=Workspace(tmp_path, lambda *a: True))
    stub = _Stub()
    with patch.dict("localforge.tools.BACKENDS", {"ollama": stub}):
        d.dispatch("delegate_general_task", {"instructions": "hi"})
        d.dispatch("delegate_general_task", {"instructions": "again"})
    assert d.installed == {"talker:3b"}


# --- small orchestrators get an honest heads-up -------------------------------------------


@pytest.mark.parametrize("model, size", [("ollama/gemma3:4b", 4), ("ollama/qwen2.5:0.5b", 0.5), ("ollama/qwen2.5-coder:14b", 14), ("ollama/llama3:latest", None)])
def test_parameter_size_is_read_from_the_tag(model, size):
    assert local_transport.parameter_billions(model) == size


def test_small_local_orchestrator_label_warns():
    assert "unreliable at planning" in cli_module._orchestrator_label("ollama/gemma3:4b", None)
    assert "unreliable" not in cli_module._orchestrator_label("ollama/qwen2.5:14b", None)


def test_prompt_tells_the_orchestrator_to_just_answer_chat():
    prompt = local_transport.cli_transport._render_prompt([{"role": "user", "content": "hi"}], [])
    assert "answer it directly with final_answer and use no tools" in prompt


def test_an_answer_that_only_promises_work_gets_one_nudge():
    """Seen live: gemma3:4b replied "Okay, I will create a file named hello.py"
    with no tool call, and nothing happened."""
    from unittest.mock import MagicMock

    def final(text):
        msg = MagicMock(tool_calls=None, content=text)
        msg.model_dump.return_value = {"role": "assistant", "content": text}
        return MagicMock(choices=[MagicMock(message=msg)], usage=None)

    replies = iter([final("Okay, I will create a file named hello.py."), final("Still just talking. I will create it.")])
    with (
        patch.object(orch, "completion", side_effect=lambda **kw: next(replies)),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        conv = orch.Conversation()
        result = orch.run("create hello.py", "gpt-5", hardware=_hw(), conversation=conv)

    nudges = [m for m in conv.messages if "didn't call any tool" in str(m.get("content"))]
    assert len(nudges) == 1  # only once, then the answer stands
    assert result.answer == "Still just talking. I will create it."


def test_a_normal_answer_is_not_nudged():
    from unittest.mock import MagicMock

    msg = MagicMock(tool_calls=None, content="Hello! What would you like to build?")
    msg.model_dump.return_value = {"role": "assistant", "content": msg.content}
    with (
        patch.object(orch, "completion", return_value=MagicMock(choices=[MagicMock(message=msg)], usage=None)),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        conv = orch.Conversation()
        orch.run("hi", "gpt-5", hardware=_hw(), conversation=conv)
    assert not any("didn't call any tool" in str(m.get("content")) for m in conv.messages)
