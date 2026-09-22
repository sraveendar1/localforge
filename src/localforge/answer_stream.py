"""Stream the orchestrator's answer as it's written.

Through a CLI login or a local orchestrator, each reply is a JSON decision:
{"tool_calls": [...]} or {"final_answer": "..."}. Showing that raw would put
JSON on screen, so `AnswerStreamer` watches the partial text as it arrives
and, once it sees `"final_answer": "`, decodes just that string --
including escapes split across chunks (a live sample split `\\"` between two
deltas) -- and hands the decoded text on piece by piece. A tool-call reply
produces nothing here; those show up as activity lines instead.
"""

from __future__ import annotations

import re
from typing import Callable

_START = re.compile(r'"final_answer"\s*:\s*"')
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class AnswerStreamer:
    def __init__(self, on_text: Callable[[str], None]):
        self.on_text = on_text
        self._buffer = ""
        self._pos: int | None = None  # index in _buffer of the next undecoded char of the answer
        self._done = False
        self.emitted = ""

    def feed(self, chunk: str) -> None:
        if self._done or not chunk:
            return
        self._buffer += chunk
        if self._pos is None:
            match = _START.search(self._buffer)
            if match is None:
                return
            self._pos = match.end()
        self._decode()

    def _decode(self) -> None:
        out = []
        buf, i = self._buffer, self._pos
        while i < len(buf):
            ch = buf[i]
            if ch == '"':
                self._done = True
                i += 1
                break
            if ch != "\\":
                out.append(ch)
                i += 1
                continue
            if i + 1 >= len(buf):
                break  # escape split across chunks: wait for the rest
            code = buf[i + 1]
            if code == "u":
                if i + 6 > len(buf):
                    break
                try:
                    out.append(chr(int(buf[i + 2 : i + 6], 16)))
                except ValueError:
                    out.append(buf[i : i + 6])
                i += 6
            else:
                out.append(_ESCAPES.get(code, code))
                i += 2
        self._pos = i
        if out:
            text = "".join(out)
            self.emitted += text
            self.on_text(text)
