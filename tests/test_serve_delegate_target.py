"""serve.py's protocol surface for delegate targets: /local-model typed in
the chat box (parity with the terminal REPL's slash command, which serve.py
doesn't get for free -- unlike repl.py, it dispatches an explicit elif
chain, not "any Typer command"), plus delegate_options_request/
set_delegate_target for the desktop app's "Change" picker on the Active
LLMs panel.
"""
import io

from localforge import delegate_target as dt
from localforge import trust
from localforge.serve import StdioServer


def make_server(tmp_path, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", object())
    trust.trust(tmp_path)
    server = StdioServer(tmp_path, "test-model", out=out, run_fn=lambda *a, **k: None, scratch_root=scratch, **kwargs)
    return server, out


def events(out):
    import json

    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def last(out, event_type):
    matches = [e for e in events(out) if e["type"] == event_type]
    assert matches, f"no {event_type!r} event; got {[e['type'] for e in events(out)]}"
    return matches[-1]


def test_bare_local_model_chat_command_lists_all_three(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "user_message", "text": "/local-model"})
    ev = last(out, "local_model")
    assert set(ev["targets"].keys()) == set(dt.MODALITIES)
    assert ev["targets"]["coding"]["target"] == "auto"


def test_local_model_chat_command_pins_a_real_model(tmp_path):
    from localforge.catalog import load_catalog

    a_coding_model = next(m.name for m in load_catalog() if m.modality == "coding")
    server, out = make_server(tmp_path)
    server.handle({"type": "user_message", "text": f"/local-model coding {a_coding_model}"})
    ev = last(out, "local_model")
    assert ev["targets"]["coding"]["target"] == f"ollama:{a_coding_model}"
    assert dt.get("coding") == dt.DelegateTarget(kind="ollama", model=a_coding_model)


def test_local_model_chat_command_rejects_a_bad_model(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "user_message", "text": "/local-model coding not-a-real-model"})
    err = last(out, "error")
    assert "isn't a coding model in the catalog" in err["message"]


def test_local_model_chat_command_rejects_unknown_modality(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "user_message", "text": "/local-model nonsense x"})
    err = last(out, "error")
    assert "Unknown task type" in err["message"]


def test_local_model_request_message_type_also_works(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "local_model_request"})
    ev = last(out, "local_model")
    assert "coding" in ev["targets"]


def test_delegate_options_lists_local_catalog_entries_for_the_modality(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "coding"})
    ev = last(out, "delegate_options")
    assert ev["modality"] == "coding"
    assert ev["current"] == "auto"
    assert isinstance(ev["local"], list) and len(ev["local"]) > 0
    assert all("name" in m and "installed" in m for m in ev["local"])
    assert isinstance(ev["cloud"], list)


def test_delegate_options_rejects_unknown_modality(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "nonsense"})
    err = last(out, "error")
    assert "Unknown task type" in err["message"]


def test_delegate_options_lists_an_available_api_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    server, out = make_server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "coding"})
    ev = last(out, "delegate_options")
    assert any(c["kind"] == "api" and c["provider"] == "anthropic" for c in ev["cloud"])


def test_delegate_options_excludes_a_provider_with_no_key_or_cli(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from localforge import serve as serve_module

    monkeypatch.setattr(serve_module.cli_transport, "available", lambda provider: False)
    server, out = make_server(tmp_path)
    server.handle({"type": "delegate_options_request", "modality": "coding"})
    ev = last(out, "delegate_options")
    assert ev["cloud"] == []


def test_set_delegate_target_pins_a_local_model(tmp_path):
    from localforge.catalog import load_catalog

    a_docs_model = next(m.name for m in load_catalog() if m.modality == "docs")
    server, out = make_server(tmp_path)
    server.handle({"type": "set_delegate_target", "modality": "docs", "target": a_docs_model})
    ev = last(out, "local_model")
    assert ev["targets"]["docs"]["target"] == f"ollama:{a_docs_model}"


def test_set_delegate_target_pins_a_cloud_target(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    server, out = make_server(tmp_path)
    server.handle({"type": "set_delegate_target", "modality": "coding", "target": "api:anthropic:claude-haiku-4-5"})
    ev = last(out, "local_model")
    assert ev["targets"]["coding"]["target"] == "api:anthropic:claude-haiku-4-5"


def test_set_delegate_target_clears_back_to_auto(tmp_path):
    from localforge.catalog import load_catalog

    a_coding_model = next(m.name for m in load_catalog() if m.modality == "coding")
    server, out = make_server(tmp_path)
    server.handle({"type": "set_delegate_target", "modality": "coding", "target": a_coding_model})
    server.handle({"type": "set_delegate_target", "modality": "coding", "target": "auto"})
    ev = last(out, "local_model")
    assert ev["targets"]["coding"]["target"] == "auto"


def test_set_delegate_target_rejects_an_invalid_pick_without_changing_state(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "set_delegate_target", "modality": "coding", "target": "not-a-real-model"})
    err = last(out, "error")
    assert "isn't a coding model in the catalog" in err["message"]
    assert dt.get("coding") == dt.AUTO
