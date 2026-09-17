"""Placeholder for the image/video backend (ComfyUI).

Not implemented yet. Image and video generation are job-based (submit a
prompt + params, poll for completion, fetch a file) rather than a single
request/response like text generation, so this backend's `generate()` will
return a BackendResult with type="file" pointing at the rendered output on
disk. The catalog and orchestrator already treat "image"/"video" as first-
class modalities (see catalog_data.yaml) so wiring this in later does not
require changes outside this file and backends/__init__.py.
"""

from __future__ import annotations

from localforge.backends.base import BackendResult


class ComfyUIBackend:
    def ensure_available(self, model_name: str) -> None:
        raise NotImplementedError(
            f"Image/video generation ({model_name}) is not implemented yet. "
            "Set up ComfyUI and wire it in here."
        )

    def generate(self, model_name: str, prompt: str, **kwargs) -> BackendResult:
        raise NotImplementedError(
            f"Image/video generation ({model_name}) is not implemented yet."
        )
