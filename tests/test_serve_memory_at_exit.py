"""The desktop app has no `/exit` -- a folder switch or quit just kills the
backend process outright (lib.rs's kill_session() writes {"type":
"shutdown"} then kills the child). Before this, that meant even the CLI's
one piece of cross-session memory for a folder -- the condensed session
note and remembered facts, see cli.py's `_save_memory_at_exit()` -- was
silently lost every time a desktop session ended, on top of the raw chat
transcript that was never going to survive a process restart anyway.
Reported as "when I switch folders the chats are completely lost".

StdioServer.close() now mirrors `_save_memory_at_exit()`: extract_facts()
then compact() run against whatever local model localforge.memory.keeper()
resolves, exactly like ending a terminal session with /exit already did.
"""

import io

from localforge import trust
from localforge.orchestrator import Conversation
from localforge.serve import StdioServer

import localforge.serve as serve_module


def make_server(tmp_path, conversation=None, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    trust.trust(tmp_path)
    server = StdioServer(
        tmp_path, "test-model", out=out, run_fn=lambda *a, **k: None,
        scratch_root=scratch, conversation=conversation or Conversation(), **kwargs,
    )
    return server, out


def test_a_session_with_no_user_turns_saves_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(serve_module.memory, "keeper", lambda d: calls.append("keeper") or object())
    monkeypatch.setattr(serve_module.memory, "extract_facts", lambda *a, **k: calls.append("extract_facts"))
    monkeypatch.setattr(serve_module.memory, "compact", lambda *a, **k: calls.append("compact"))

    server, _ = make_server(tmp_path)  # fresh Conversation(): messages == []
    server.close()

    assert calls == []


def test_a_finished_task_extracts_facts_and_compacts_on_close(tmp_path, monkeypatch):
    calls = []
    fake_keeper = object()
    monkeypatch.setattr(serve_module.memory, "keeper", lambda d: fake_keeper)
    monkeypatch.setattr(serve_module.memory, "extract_facts", lambda *a, **k: calls.append(("extract_facts", a[2])))
    monkeypatch.setattr(serve_module.memory, "compact", lambda *a, **k: calls.append(("compact", k.get("root"))))

    conv = Conversation(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "build a hello.py"},
        {"role": "assistant", "content": "done"},
    ])
    server, _ = make_server(tmp_path, conversation=conv)
    server.close()

    assert [c[0] for c in calls] == ["extract_facts", "compact"]
    assert calls[0][1] == server.root
    assert calls[1][1] == server.root


def test_no_installed_local_model_skips_saving_without_raising(tmp_path, monkeypatch):
    # No Ollama reachable (the sandbox this was written in has none running)
    # -> memory.keeper() returns None for real -- close() must not raise.
    conv = Conversation(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ])
    server, _ = make_server(tmp_path, conversation=conv)
    server.close()  # no exception


def test_shutdown_over_the_protocol_triggers_the_same_save(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(serve_module.memory, "keeper", lambda d: object())
    monkeypatch.setattr(serve_module.memory, "extract_facts", lambda *a, **k: calls.append("extract_facts"))
    monkeypatch.setattr(serve_module.memory, "compact", lambda *a, **k: calls.append("compact"))

    conv = Conversation(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
    ])
    server, out = make_server(tmp_path, conversation=conv, inp=io.StringIO('{"type": "shutdown"}\n'))
    server.serve_forever()  # reads the shutdown line, calls handle() -> False -> close()

    assert calls == ["extract_facts", "compact"]
