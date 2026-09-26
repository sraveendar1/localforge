"""The desktop app's backend used to fail outright in an untrusted folder --
`localforge serve --stdio` pre-checked `trust.is_trusted()` before starting
at all and wrote one fatal error telling the user to "run localforge there
once interactively to trust it", forcing a trip to a terminal every time the
GUI opened a folder it hadn't seen before. trust.py's own docstring already
anticipated a "native desktop front end" sharing its decide()/apply_choice()
helpers, but nothing actually wired them into serve.py.

Now StdioServer resolves trust over the protocol itself: it announces
`trust_required` once at startup (and again if anything else arrives first),
and a `trust_response` message answers it, exactly mirroring the terminal
REPL's "No" exits / "Yes" continues behavior but rendered by the GUI instead
of blocking on stdin.
"""

import io

from localforge import trust
from localforge.serve import StdioServer

from tests.test_serve import events, wait_for, _done


def make_untrusted_server(tmp_path, **kwargs):
    out = io.StringIO()
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    kwargs.setdefault("conversation", object())
    # Deliberately not trusted -- that's the point of this test file.
    server = StdioServer(tmp_path, "test-model", out=out, run_fn=_done, scratch_root=scratch, **kwargs)
    return server, out


def test_an_untrusted_folder_is_announced_right_after_ready(tmp_path):
    server, out = make_untrusted_server(tmp_path, inp=io.StringIO(""))
    server.serve_forever()
    got = events(out)
    assert got[0]["type"] == "ready"
    assert got[1] == {"type": "trust_required", "folder": str(tmp_path.resolve())}


def test_anything_but_trust_response_is_a_no_op_until_answered(tmp_path):
    server, out = make_untrusted_server(tmp_path)
    try:
        assert server.trusted is False
        # A real user_message must not slip through and actually run a task.
        assert server.handle({"type": "user_message", "text": "do something"}) is True
        got = events(out)
        assert got[-1]["type"] == "trust_required"
        assert not server.busy
    finally:
        server.close()


def test_trusting_it_lets_the_session_continue(tmp_path):
    server, out = make_untrusted_server(tmp_path)
    try:
        assert server.handle({"type": "trust_response", "trust": True}) is True
        assert server.trusted is True
        assert trust.is_trusted(tmp_path)
        result = wait_for(out, "trust_result")
        assert result["trusted"] is True

        # Now behaves like any other trusted session.
        server.handle({"type": "user_message", "text": "x"})
        assert wait_for(out, "run_finished")["answer"] == "done"
    finally:
        server.close()


def test_declining_ends_the_session_like_the_terminal_prompts_no(tmp_path):
    server, out = make_untrusted_server(tmp_path)
    try:
        assert server.handle({"type": "trust_response", "trust": False}) is False
        assert server.trusted is False
        assert not trust.is_trusted(tmp_path)
        assert wait_for(out, "trust_result")["trusted"] is False
    finally:
        server.close()


def test_shutdown_works_even_before_trust_is_resolved(tmp_path):
    server, out = make_untrusted_server(tmp_path)
    try:
        assert server.handle({"type": "shutdown"}) is False
    finally:
        server.close()


def test_an_already_trusted_folder_skips_the_prompt_entirely(tmp_path):
    trust.trust(tmp_path)
    server, out = make_untrusted_server(tmp_path)
    try:
        assert server.trusted is True
        server.handle({"type": "user_message", "text": "x"})
        assert wait_for(out, "run_finished")["answer"] == "done"
        assert not any(e["type"] == "trust_required" for e in events(out))
    finally:
        server.close()
