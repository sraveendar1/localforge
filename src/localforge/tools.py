"""Defines the tools exposed to the frontier orchestrator model, and dispatches
each tool call to the best-fitting local model via the matching backend.
"""

from __future__ import annotations

from typing import Callable

from localforge.backends import BACKENDS
from localforge.catalog import ModelEntry, best_match, load_catalog
from localforge.hardware import HardwareProfile

# Called right before a subtask is handed to a local model, so callers (the
# CLI, the wizard) can show the user what's actually doing the work and why.
DelegateCallback = Callable[[str, ModelEntry], None]

# Modality -> (tool name, description, prompt-building instructions)
TASK_MODALITIES = {
    "coding": {
        "tool_name": "delegate_coding_task",
        "description": "Delegate a coding subtask to a local coding-specialist model.",
    },
    "docs": {
        "tool_name": "delegate_docs_task",
        "description": "Delegate a documentation-writing subtask to a local model.",
    },
    "general": {
        "tool_name": "delegate_general_task",
        "description": "Delegate a general-purpose text subtask to a local model.",
    },
}


def build_tool_schemas() -> list[dict]:
    """OpenAI/LiteLLM-style tool schemas for every modality with an available
    local model on this machine.
    """
    schemas = []
    for modality, meta in TASK_MODALITIES.items():
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": meta["tool_name"],
                    "description": meta["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instructions": {
                                "type": "string",
                                "description": "What the local model should do, with all context it needs.",
                            }
                        },
                        "required": ["instructions"],
                    },
                },
            }
        )
    return schemas


def _prompt_for(modality: str, instructions: str) -> str:
    prefaces = {
        "coding": "You are a focused coding assistant. Produce working code for this task:\n\n",
        "docs": "You are a technical writer. Write clear documentation for this task:\n\n",
        "general": "",
    }
    return prefaces.get(modality, "") + instructions


class Dispatcher:
    """Resolves a tool call to modality -> catalog entry -> backend, and runs it."""

    def __init__(self, hardware: HardwareProfile, catalog: list[ModelEntry] | None = None):
        self.hardware = hardware
        self.catalog = catalog if catalog is not None else load_catalog()
        self._resolved_models: dict[str, ModelEntry] = {}

    def _tool_name_to_modality(self, tool_name: str) -> str:
        for modality, meta in TASK_MODALITIES.items():
            if meta["tool_name"] == tool_name:
                return modality
        raise ValueError(f"Unknown tool: {tool_name}")

    def resolve(self, modality: str) -> ModelEntry:
        if modality not in self._resolved_models:
            self._resolved_models[modality] = best_match(modality, self.hardware, self.catalog)
        return self._resolved_models[modality]

    def dispatch(self, tool_name: str, instructions: str, on_delegate: DelegateCallback | None = None) -> str:
        modality = self._tool_name_to_modality(tool_name)
        entry = self.resolve(modality)
        if on_delegate is not None:
            on_delegate(modality, entry)
        backend = BACKENDS[entry.runtime]
        backend.ensure_available(entry.name)
        result = backend.generate(entry.name, _prompt_for(modality, instructions))
        if result["type"] == "file":
            return f"[generated file: {result['content']}]"
        return result["content"]
