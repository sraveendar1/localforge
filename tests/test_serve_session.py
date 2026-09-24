import io
import json
import types

from localforge.serve import StdioServer, Scratchpad


def make_server(tmp_path, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", object())
    server = StdioServer(tmp_path, "test-model", out=out, run_fn=lambda *args: None, scratch_root=scratch, **kwargs)
    return server, out


def events(out):
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def of_type(out, event_type):
    return [e for e in events(out) if e["type"] == event_type]


def test_system_stats_request(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "system_stats"})
    stats = of_type(out, "system_stats")[0]
    assert "hardware" in stats
    assert "cpu_percent" in stats
    assert "ram_used_gb" in stats
    assert "ram_total_gb" in stats
    assert isinstance(stats["hardware"], dict)
    assert isinstance(stats["cpu_percent"], (int, float))
    assert isinstance(stats["ram_used_gb"], (int, float))
    assert isinstance(stats["ram_total_gb"], (int, float))


def test_memory_list(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "memory_list"})
    assert "facts" in of_type(out, "memory")[0]
    assert isinstance(of_type(out, "memory")[0]["narrative"], str)


def test_memory_forget_no_name(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "memory_forget"})
    assert "message" in of_type(out, "error")[0]
    assert "Missing name" in of_type(out, "error")[0]["message"]


def test_memory_clear(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "memory_clear"})
    assert "facts" in of_type(out, "memory")[0]


def test_scratch_list(tmp_path):
    sp = Scratchpad(tmp_path)
    sp.ensure()
    sp_path = sp.root
    (sp_path / "test.txt").write_text("test content")
    server, out = make_server(tmp_path)
    server._scratchpad = sp
    server.handle({"type": "scratch_list"})
    assert "path" in of_type(out, "scratch")[0]["files"][0]
    assert "test.txt" in of_type(out, "scratch")[0]["files"][0]["path"]
    assert "size" in of_type(out, "scratch")[0]["files"][0]
    assert len("test content") == of_type(out, "scratch")[0]["files"][0]["size"]


def test_scratch_clear(tmp_path):
    sp = Scratchpad(tmp_path)
    sp.ensure()
    sp_path = sp.root
    (sp_path / "test.txt").write_text("test content")
    server, out = make_server(tmp_path)
    server._scratchpad = sp
    server.handle({"type": "scratch_clear"})
    assert not (sp_path / "test.txt").exists()
    assert not of_type(out, "scratch")[0]["files"]


def test_scratch_clear_while_busy(tmp_path):
    sp = Scratchpad(tmp_path)
    sp.ensure()
    (sp.root / "test.txt").write_text("test content")
    server, out = make_server(tmp_path)
    server._scratchpad = sp
    server._worker = types.SimpleNamespace(is_alive=lambda: True)
    server.handle({"type": "scratch_clear"})
    assert (sp.root / "test.txt").exists()
    assert "message" in of_type(out, "error")[0]
    assert "already in progress" in of_type(out, "error")[0]["message"]


def test_new_session_while_busy(tmp_path):
    server, out = make_server(tmp_path)
    server._worker = types.SimpleNamespace(is_alive=lambda: True)
    server.handle({"type": "new_session"})
    assert not of_type(out, "session_reset")
    assert "message" in of_type(out, "error")[0]
    assert "already in progress" in of_type(out, "error")[0]["message"]


def test_get_state(tmp_path):
    server, out = make_server(tmp_path)
    server.handle({"type": "get_state"})
    assert "model" in of_type(out, "settings")[0]
    assert "test-model" == of_type(out, "settings")[0]["model"]
    assert "auto_approve" in of_type(out, "settings")[0]
    assert False == of_type(out, "settings")[0]["auto_approve"]
    assert "busy" in of_type(out, "settings")[0]
    assert False == of_type(out, "settings")[0]["busy"]
