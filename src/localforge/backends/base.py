from __future__ import annotations

from typing import Literal, Protocol, TypedDict


class BackendResult(TypedDict):
    type: Literal["text", "file"]
    content: str  # text content, or a filesystem path when type == "file"
    tokens: int  # tokens the local model generated, for usage metrics; 0 if unknown


class Backend(Protocol):
    """A runtime capable of serving one or more models for a given modality."""

    def ensure_available(self, model_name: str, on_progress: object = None) -> None:
        """Make sure `model_name` is pulled/loaded and ready to serve.
        `on_progress`, if given, is called with each raw progress event the
        backend emits while downloading (shape is backend-specific).
        """
        ...

    def generate(self, model_name: str, prompt: str, **kwargs) -> BackendResult:
        """Run `model_name` on `prompt` and return its output."""
        ...
