from __future__ import annotations

from typing import Literal, Protocol, TypedDict


class BackendResult(TypedDict, total=False):
    type: Literal["text", "file"]
    content: str  # text content, or a filesystem path when type == "file"
    tokens: int  # tokens the local model generated, for usage metrics; 0 if unknown
    truncated: bool  # the output hit the length cap and stops mid-way


class Backend(Protocol):
    """A runtime capable of serving one or more models for a given modality."""

    def ensure_available(self, model_name: str, on_progress: object = None) -> None:
        """Make sure `model_name` is pulled/loaded and ready to serve.
        `on_progress`, if given, is called with each raw progress event the
        backend emits while downloading (shape is backend-specific).
        """
        ...

    def generate(self, model_name: str, prompt: str, on_token: object = None, **kwargs) -> BackendResult:
        """Run `model_name` on `prompt` and return its output. `on_token`, if
        given, is called with each chunk of text as it's generated (a backend
        that can't stream may ignore it and just return the full result).
        """
        ...
