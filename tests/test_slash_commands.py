"""Every slash command typed in the desktop app answers in the chat, and nothing typed there
is left silent or misleading. (Reported: "some of the slash commands in the app are not working":
/memory, /scratch, /queue, /usage, /scan, /model, /advanced-model, /auto and /stream only updated
a side panel or a setting, so typing them looked like nothing happened; a bare /run started a task
with no text; the terminal-only commands said just "Unknown command".)"""
import io
import json

import pytest

import localforge.serve as serve_module
from localforge import memory, trust
from localforge.orchestrator import Conversation
from localforge.serve import StdioServer


def _server(tmp_path):
    out = io.StringIO()
    (tmp_path / "scratch").mkdir(exist_ok=True)
    trust.trust(tmp_path)
    server = StdioServer(tmp_path, "claude-opus-5", None, out=out, run_fn=lambda *a, **k: None,
                         scratch_root=tmp_path / "scratch", conversation=Conversation())
    return server, out


def _events(out):
    return [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


def _say(server, out, text):
    """Type `text` as the user would; return (texts shown in the chat, errors)."""
    out.seek(0), out.truncate()
    server.handle({"type": "user_message", "text": text})
    ev = _events(out)
    return [e["text"] for e in ev if e["type"] == "system_text"], [e["message"] for e in ev if e["type"] == "error"], ev


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    monkeypatch.setattr(serve_module.OllamaBackend, "is_running", lambda self: False)
    monkeypatch.setattr(serve_module.OllamaBackend, "list_installed", lambda self: [])


@pytest.mark.parametrize("command, needle", [
    ("/memory", "Nothing is remembered"),
    ("/memory clear", "Cleared"),
    ("/memory forget nothing", "no remembered fact"),
    ("/memory forget", "Say which one"),
    ("/scratch", "scratchpad is empty"),
    ("/scratch clear", "Scratchpad cleared"),
    ("/queue", "Nothing is queued"),
    ("/queue clear", "Queue cleared"),
    ("/stop", "Nothing is running"),
    ("/usage", "Usage for this project"),
    ("/scan", "This computer:"),
    ("/model", "The orchestrator"),
    ("/advanced-model", "Models for each kind of work"),
    ("/advanced-model coding", "coding:"),
    ("/auto on", "Auto-approve is on"),
    ("/auto off", "Auto-approve is off"),
    ("/stream on", "Streaming is on"),
    ("/stream off", "Streaming is off"),
    ("/run", "Usage: /run"),
    ("/budget", "this month"),
])
def test_each_command_answers_in_the_chat(tmp_path, command, needle):
    server, out = _server(tmp_path)
    texts, errors, _ = _say(server, out, command)
    assert not errors, errors
    assert any(needle.lower() in t.lower() for t in texts), (command, texts)


def test_a_bare_run_does_not_start_a_task(tmp_path):
    server, out = _server(tmp_path)
    _, _, ev = _say(server, out, "/run")
    assert not any(e["type"] == "run_started" for e in ev) and not server.busy


def test_memory_lists_what_is_remembered(tmp_path):
    server, out = _server(tmp_path)
    memory.remember(tmp_path, "prefers-tabs", "Use tabs.", "User prefers tabs over spaces", "feedback")
    texts, _, _ = _say(server, out, "/memory")
    assert "prefers-tabs" in texts[0] and "tabs over spaces" in texts[0]
    texts, _, _ = _say(server, out, "/memory forget prefers-tabs")
    assert "Forgot" in texts[0]


def test_queue_lists_waiting_requests(tmp_path):
    server, out = _server(tmp_path)
    server._queue.append(("add dark mode", None))
    texts, _, _ = _say(server, out, "/queue")
    assert "1. add dark mode" in texts[0]


def test_switching_the_model_says_so_and_a_refusal_is_an_error(tmp_path, monkeypatch):
    server, out = _server(tmp_path)
    texts, errors, _ = _say(server, out, "/model claude-sonnet-5-5")
    assert not errors and "Switched the orchestrator to claude-sonnet-5-5" in texts[0]
    monkeypatch.setattr(server, "_switch_orchestrator", lambda m: "Can't use it: too big. Nothing was changed.")
    texts, errors, _ = _say(server, out, "/model huge:70b")
    assert errors and "too big" in errors[0] and not any("Switched" in t for t in texts)


def test_a_bad_advanced_model_choice_is_an_error_not_silence(tmp_path):
    server, out = _server(tmp_path)
    _, errors, _ = _say(server, out, "/advanced-model docs not-a-model")
    assert errors and "isn't installed" in errors[0]


@pytest.mark.parametrize("command", ["/goals", "/tasks", "/upgrade", "/setup", "/theme", "/delete", "/exit"])
def test_terminal_only_commands_say_where_to_go_instead_of_unknown(tmp_path, command):
    server, out = _server(tmp_path)
    _, errors, _ = _say(server, out, command)
    assert errors and "isn't available in the app" in errors[0] and "Unknown command" not in errors[0]


def test_a_mistyped_command_points_at_help(tmp_path):
    server, out = _server(tmp_path)
    _, errors, _ = _say(server, out, "/bogus")
    assert errors == ["Unknown command: /bogus. Type /help to see the list."]


def test_help_lists_every_command_in_the_menu():
    import re
    from pathlib import Path

    menu = set(re.findall(r'name: "(/[a-z-]+)"', (Path(__file__).parents[1] / "desktop/src/SlashMenu.tsx").read_text()))
    server = StdioServer.__new__(StdioServer)
    captured = []
    server.emit = lambda kind, **f: captured.append((kind, f))
    server.handle_help_command()
    helped = " ".join(c["name"] for c in captured[0][1]["commands"])
    missing = [m for m in menu if m not in helped]
    assert not missing, f"/help leaves out: {missing}"
