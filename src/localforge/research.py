"""Live research for better open-weight models (the desktop's `/upgrade research`).

`upgrades.plan()` only knows the models in our hand-curated catalog. This looks further: the user's
own orchestrator searches the web (localforge's `web_search`/`fetch_url`, so it works with an API
key or a CLI login alike) for newer models in the Ollama library that fit this machine, and replies
with a list of names. **Nothing in that reply is trusted.** Every name is checked against Ollama's
registry (a made-up tag has no manifest, so it is dropped) and against this machine's memory and
disk (`model_fit.download_problem`, the same refusal as any download), reasoning ("thinking")
models are left out the way the catalog leaves them out, and the result is a *suggestion*: the
user chooses, the model is downloaded, and only then is the model it replaces removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from localforge import cli_transport
from localforge.hardware import HardwareProfile

TEXT_ROLES = ("coding", "docs", "general")
# "decision" is the small model that makes the pass/fail call on delegated work (a Jev-style decision
# model, or any judge model).
MODALITIES = TEXT_ROLES + ("decision",)
MAX_SUGGESTIONS = 4
# Reasoning models put their thinking in the text, and it would end up inside generated files (the
# catalog leaves them out for the same reason).
REASONING_MARKERS = ("thinking", "reason", "-r1", "r1:", "qwq", "gpt-oss", "deepseek-r1", "magistral")
NAME_RE = re.compile(r"[a-z0-9][a-z0-9._\-/]*:[a-z0-9._\-]+")

PROMPT = """\
Research task (do NOT write or change any files, do NOT delegate anything): find open-weight models in the \
Ollama library that would be a better local model for this machine than the ones it uses now.

This machine: {hardware}.
Currently used (installed): {current}.
Already installed: {installed}.

Use web_search and fetch_url (start from https://ollama.com/library?sort=newest and the model pages) to find \
models released more recently than the ones above, in these roles: coding, docs (writing), general, and decision (below). Only \
suggest a model whose exact Ollama tag you actually saw on ollama.com (for example `qwen3-coder:30b`), whose \
download plausibly fits this machine's memory and free disk, and that is NOT a reasoning/"thinking" model. \
Also look for a small *decision / judge* model (role "decision"): one that picks an answer from options or \
gives a pass/fail verdict quickly, like TypeSafe AI's Jev (closed, hosted) and its open-weight look-alikes, or a \
judge model such as Atla's Selene. Only suggest one whose exact Ollama tag you saw. \
At most {limit} suggestions, best first. Give a short reason for each (what is newer or better, in one sentence).

Reply with ONLY this JSON, nothing else:
{{"candidates": [{{"name": "<ollama tag>", "modality": "coding|docs|general|decision", "why": "<one sentence>"}}]}}
"""


@dataclass
class Suggestion:
    name: str
    modality: str
    why: str
    size_gb: float
    replaces: str | None  # the installed model it would take over from
    replaces_managed: bool  # LocalForge installed that one (the user's own needs an explicit OK to delete)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Outcome:
    suggestions: list[Suggestion] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # {"name", "reason"}
    note: str = ""  # something to tell the user when there's nothing to suggest


def task_text(hardware: HardwareProfile, installed: set[str], current: dict[str, str]) -> str:
    gpu = f", {hardware.gpus[0].name} ({hardware.gpus[0].vram_gb:g} GB)" if hardware.gpus else ""
    machine = f"{hardware.os} {hardware.arch}, {hardware.ram_gb:g} GB RAM{gpu}, {hardware.free_disk_gb:g} GB free disk"
    now = ", ".join(f"{m}: {n}" for m, n in current.items()) or "nothing yet"
    return PROMPT.format(hardware=machine, current=now, installed=", ".join(sorted(installed)) or "nothing", limit=MAX_SUGGESTIONS)


def parse(answer: str) -> list[dict]:
    """The candidate entries in the orchestrator's reply; an unusable reply is an empty list."""
    data = cli_transport._extract_json(answer or "")
    items = data.get("candidates") if isinstance(data, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def verify(
    candidates: list[dict],
    hardware: HardwareProfile,
    installed: set[str],
    current: dict[str, str],
    managed: set[str],
    registry_size,
    fit_problem,
) -> Outcome:
    """Keep only what is real and fits. `registry_size(name)` -> GB or None (not in the registry);
    `fit_problem(name)` -> why it can't be used here, or None."""
    out = Outcome()
    seen: set[str] = set()
    for item in candidates:
        name = str(item.get("name", "")).strip().lower()
        modality = str(item.get("modality", "")).strip().lower()
        why = re.sub(r"\s+", " ", str(item.get("why", ""))).strip()[:200]

        def reject(reason: str) -> None:
            out.rejected.append({"name": name or "(no name)", "reason": reason})

        if not NAME_RE.fullmatch(name):
            reject("not an exact Ollama tag (it needs a name and a tag like qwen3-coder:30b)")
        elif name in seen:
            continue
        elif modality not in MODALITIES:
            reject(f"unknown role {modality!r}")
        elif any(marker in name for marker in REASONING_MARKERS):
            reject("a reasoning model; its thinking text could end up inside generated files")
        elif name in installed:
            reject("already installed")
        elif (size := registry_size(name)) is None:
            reject("not found in the Ollama registry (or the registry couldn't be reached)")
        elif problem := fit_problem(name):
            reject(problem)
        else:
            seen.add(name)
            old = current.get(modality)
            old = old if old in installed else None
            out.suggestions.append(Suggestion(name, modality, why, round(size, 1), old, bool(old and old in managed)))
    out.suggestions = out.suggestions[:MAX_SUGGESTIONS]
    if not out.suggestions:
        out.note = (
            "The research found nothing new that is real and fits this machine."
            if candidates
            else "The research didn't come back with a usable list of models (the orchestrator may not have reached the web)."
        )
    return out
