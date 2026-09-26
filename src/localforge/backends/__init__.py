"""Runtime backends. Each backend knows how to run one class of local model
and return a result in the shape the orchestrator expects: a dict with a
"type" of "text" or "file".
"""

from localforge.backends.base import Backend, BackendResult
from localforge.backends.ollama import OllamaBackend
from localforge.backends.cloud import CloudApiBackend, CloudCliBackend

BACKENDS: dict[str, Backend] = {
    "ollama": OllamaBackend(),
    # A per-modality "advanced" override can point at a paid cloud model
    # instead of a local one -- see delegate_target.py and
    # tools.Dispatcher.resolve(). "api" bills per token; "cli" draws on a
    # provider's CLI subscription instead, same distinction the
    # orchestrator's own two transports already make.
    "api": CloudApiBackend(),
    "cli": CloudCliBackend(),
}

__all__ = ["Backend", "BackendResult", "BACKENDS"]
