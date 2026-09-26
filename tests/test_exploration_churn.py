"""Reported (with a screenshot): dozens of read_file/search calls cycling
over the same handful of files, no edit in between, until the task ran out
of rounds having made no real progress. The existing repeat-call guard
(`done_calls` in orchestrator._loop) only catches an *exact* repeat -- same
tool name and same JSON args -- so re-reading the same file at a different
offset, or re-searching with a slightly different pattern, is invisible to
it. The checkpoint's own "no progress" self-stop doesn't catch this either,
since each of those calls is distinct and succeeds, so it counts toward
`progress` exactly like a real edit would.

EXPLORE_REPEAT_THRESHOLD tracks repeats by *target* (the path or pattern),
not the exact call, and appends a nudge to the tool result once a target has
been explored EXPLORE_REPEAT_THRESHOLD times with no real change in between.
"""

from unittest.mock import MagicMock, patch

import localforge.orchestrator as orch
from localforge.hardware import HardwareProfile


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


class FakeDispatcher:
    """Every call succeeds with a plain, non-error-looking result."""

    def __init__(self):
        self.catalog, self.local_tokens_generated = [], 0

    def dispatch(self, name, args, on_delegate=None):
        if name == "read_file":
            return "1: def x():\n2:     pass\n"
        if name == "edit_file":
            return "Updated x.py (+1 -1)"
        return "ok"


def _run(replies, dispatcher=None):
    conv = orch.Conversation()
    dispatcher = dispatcher or FakeDispatcher()
    with (
        patch.object(orch, "completion", side_effect=lambda **kw: next(replies)),
        patch.object(orch, "Dispatcher", lambda *a, **kw: dispatcher),
        patch.object(orch, "build_tool_schemas", return_value=[]),
        patch.object(orch, "_installed_models", return_value=None),
        patch.object(orch.litellm, "completion_cost", return_value=0.0),
    ):
        result = orch.run("investigate x.py", "gpt-5", hardware=_hw(), conversation=conv)
    return result, conv


def _tool_contents(conv):
    return [m["content"] for m in conv.messages if m.get("role") == "tool"]


def test_re_reading_the_same_file_at_different_offsets_gets_a_nudge():
    reads = [
        _tool_reply("read_file", '{"path": "x.py", "offset": 1}'),
        _tool_reply("read_file", '{"path": "x.py", "offset": 100}'),
        _tool_reply("read_file", '{"path": "x.py", "offset": 200}'),
        _final(),
    ]
    _, conv = _run(iter(reads))
    contents = _tool_contents(conv)
    assert len(contents) == 3
    assert "already been called" not in contents[0] and "Note:" not in contents[0]
    assert "Note:" not in contents[1]
    # The third look at the same target crosses EXPLORE_REPEAT_THRESHOLD (3).
    assert "'x.py'" in contents[2] and "3 times this task" in contents[2]
    assert "act on it now" in contents[2]


def test_different_targets_dont_share_a_counter():
    reads = [
        _tool_reply("read_file", '{"path": "a.py"}'),
        _tool_reply("read_file", '{"path": "b.py"}'),
        _tool_reply("read_file", '{"path": "a.py", "offset": 50}'),
        _final(),
    ]
    _, conv = _run(iter(reads))
    contents = _tool_contents(conv)
    assert all("Note:" not in c for c in contents)  # a.py only seen twice, b.py once


def test_a_real_edit_resets_the_exploration_counter():
    calls = [
        _tool_reply("read_file", '{"path": "x.py", "offset": 1}'),
        _tool_reply("read_file", '{"path": "x.py", "offset": 50}'),
        _tool_reply("edit_file", '{"path": "x.py", "old_string": "a", "new_string": "b"}'),
        _tool_reply("read_file", '{"path": "x.py", "offset": 1}'),
        _tool_reply("read_file", '{"path": "x.py", "offset": 50}'),
        _final(),
    ]
    _, conv = _run(iter(calls))
    contents = _tool_contents(conv)
    assert len(contents) == 5
    # Two more reads after the edit -- still short of the threshold again.
    assert all("Note:" not in c for c in contents)


def test_search_is_tracked_by_pattern_not_by_exact_args():
    calls = [
        _tool_reply("search", '{"pattern": "def foo", "glob": "*.py"}'),
        _tool_reply("search", '{"pattern": "def foo", "path": "src"}'),
        _tool_reply("search", '{"pattern": "def foo"}'),
        _final(),
    ]
    _, conv = _run(iter(calls))
    contents = _tool_contents(conv)
    assert "'def foo'" in contents[2] and "3 times this task" in contents[2]
