"""Defines the tools exposed to the frontier orchestrator model, and dispatches
each tool call to the best-fitting local model via the matching backend.
"""

from __future__ import annotations

from typing import Callable

from localforge.backends import BACKENDS
from localforge.catalog import ModelEntry, best_match, candidates, load_catalog
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


def build_tool_schemas(hardware: HardwareProfile, catalog: list[ModelEntry] | None = None) -> list[dict]:
    """OpenAI/LiteLLM-style tool schemas for every modality with an available
    local model on this machine. A modality with no fitting catalog entry
    (hardware too limited) is left out entirely, rather than exposing a tool
    the frontier model could call only to get a NoFittingModelError back.
    """
    schemas = []
    for modality, meta in TASK_MODALITIES.items():
        if not candidates(modality, hardware, catalog):
            continue
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


# Cheap, deterministic tripwires for the clearest local-model failure modes
# (empty output, an outright refusal, or something absurdly short given the
# request). This is not a correctness check -- it can't tell if generated
# code actually works -- it's a floor that catches obviously broken results
# before they reach the frontier model unflagged, so they get escalated to a
# better model or clearly marked instead of silently accepted.
_REFUSAL_MARKERS = (
    "as an ai language model",
    "i cannot ",
    "i can't ",
    "i'm sorry, but",
    "i am unable to",
    "i do not have the ability",
)


def _looks_suspect(content: str, instructions: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return "empty output"
    lowered = stripped.lower()
    for marker in _REFUSAL_MARKERS:
        if marker in lowered:
            return f"looks like a refusal (contains {marker!r})"
    if len(stripped) < 20 and len(instructions) > 80:
        return "suspiciously short for the size of the request"
    return None


class Dispatcher:
    """Resolves a tool call to modality -> catalog entry -> backend, and runs it."""

    def __init__(self, hardware: HardwareProfile, catalog: list[ModelEntry] | None = None):
        self.hardware = hardware
        self.catalog = catalog if catalog is not None else load_catalog()
        self._resolved_models: dict[str, ModelEntry] = {}
        self.local_tokens_generated = 0  # running total, for usage metrics

    def _tool_name_to_modality(self, tool_name: str) -> str:
        for modality, meta in TASK_MODALITIES.items():
            if meta["tool_name"] == tool_name:
                return modality
        raise ValueError(f"Unknown tool: {tool_name}")

    def resolve(self, modality: str) -> ModelEntry:
        if modality not in self._resolved_models:
            self._resolved_models[modality] = best_match(modality, self.hardware, self.catalog)
        return self._resolved_models[modality]

    def _retry_candidate(self, modality: str, current: ModelEntry) -> ModelEntry | None:
        """A different model for `modality` to retry with, if this
        hardware fits more than one. `resolve()` already picks the single
        best-fitting model up front, so on most machines there is no
        "better" model to escalate to -- this only helps when several
        models tie or otherwise fit; the fallback for everyone else is
        `_reinforced_instructions()` retrying the *same* model. Prefers the
        highest quality_tier among the alternatives, on a one-off basis
        that never changes `resolve()`'s cached pick for the rest of the run.
        """
        alternatives = [c for c in candidates(modality, self.hardware, self.catalog) if c.name != current.name]
        return max(alternatives, key=lambda m: m.quality_tier) if alternatives else None

    def _reinforced_instructions(self, instructions: str, reason: str) -> str:
        return (
            f"{instructions}\n\n(Your previous attempt at this was rejected: {reason}. "
            "Provide a complete, direct response this time -- do not refuse and do not "
            "leave it incomplete.)"
        )

    def _run(self, modality: str, entry: ModelEntry, instructions: str, on_delegate: DelegateCallback | None) -> dict:
        if on_delegate is not None:
            on_delegate(modality, entry)
        backend = BACKENDS[entry.runtime]
        backend.ensure_available(entry.name)
        result = backend.generate(entry.name, _prompt_for(modality, instructions))
        self.local_tokens_generated += result.get("tokens", 0)  # every attempt costs local compute, retries included
        return result

    def dispatch(self, tool_name: str, instructions: str, on_delegate: DelegateCallback | None = None) -> str:
        modality = self._tool_name_to_modality(tool_name)
        entry = self.resolve(modality)
        result = self._run(modality, entry, instructions, on_delegate)
        if result["type"] == "file":
            return f"[generated file: {result['content']}]"

        content = result["content"]
        reason = _looks_suspect(content, instructions)
        if reason is None:
            return content

        retry_entry = self._retry_candidate(modality, entry) or entry
        retry_instructions = instructions if retry_entry is not entry else self._reinforced_instructions(instructions, reason)
        retry_result = self._run(modality, retry_entry, retry_instructions, on_delegate)
        if retry_result["type"] == "file":
            return f"[generated file: {retry_result['content']}]"
        retry_content = retry_result["content"]
        retry_reason = _looks_suspect(retry_content, instructions)
        if retry_reason is None:
            return retry_content

        return (
            f"[WARNING: this {modality} result may be unreliable ({retry_reason}) -- verify before use, "
            "or delegate again with clearer/simpler instructions]\n\n" + retry_content
        )
