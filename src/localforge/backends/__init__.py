"""Runtime backends. Each backend knows how to run one class of local model
and return a result in the shape the orchestrator expects: a dict with a
"type" of "text" or "file".
"""

from localforge.backends.base import Backend, BackendResult
from localforge.backends.ollama import OllamaBackend

BACKENDS: dict[str, Backend] = {
    "ollama": OllamaBackend(),
}

__all__ = ["Backend", "BackendResult", "BACKENDS"]
