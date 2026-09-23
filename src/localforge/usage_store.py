"""Token usage kept across sessions, per project.

`/usage` used to know only the running process, so closing the session threw
the numbers away and there was no way to see what a project has cost over
time. Totals are stored next to that project's memory, in localforge's own
config folder -- never in the project.

Written after every task, so an interrupted session still counts. Each
session is one entry; only the last SESSIONS_KEPT are kept, plus an
all-time total that never resets.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

SESSIONS_KEPT = 20
FIELDS = ("frontier_prompt_tokens", "frontier_completion_tokens", "frontier_cost_usd", "local_tokens_generated", "tasks")


@dataclass
class Totals:
    """Usage over some span (a session, or all time)."""

    frontier_prompt_tokens: int = 0
    frontier_completion_tokens: int = 0
    frontier_cost_usd: float = 0.0
    local_tokens_generated: int = 0
    tasks: int = 0
    subscription_cost_usd: float = 0.0  # notional: drawn from a subscription, not billed
    models: list[str] = field(default_factory=list)
    started: float = 0.0
    updated: float = 0.0

    @property
    def frontier_total_tokens(self) -> int:
        return self.frontier_prompt_tokens + self.frontier_completion_tokens

    def add(self, model: str, stats) -> None:
        self.frontier_prompt_tokens += stats.frontier_prompt_tokens
        self.frontier_completion_tokens += stats.frontier_completion_tokens
        self.local_tokens_generated += stats.local_tokens_generated
        if stats.frontier_via_subscription:
            self.subscription_cost_usd += stats.frontier_cost_usd or 0.0
        else:
            self.frontier_cost_usd += stats.frontier_cost_usd or 0.0
        self.tasks += 1
        self.updated = time.time()
        if model and model not in self.models:
            self.models.append(model)

    def to_dict(self) -> dict:
        return {
            "frontier_prompt_tokens": self.frontier_prompt_tokens,
            "frontier_completion_tokens": self.frontier_completion_tokens,
            "frontier_cost_usd": round(self.frontier_cost_usd, 6),
            "subscription_cost_usd": round(self.subscription_cost_usd, 6),
            "local_tokens_generated": self.local_tokens_generated,
            "tasks": self.tasks,
            "models": self.models,
            "started": self.started,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Totals":
        totals = cls()
        for key, value in (data or {}).items():
            if hasattr(totals, key) and value is not None:
                setattr(totals, key, value)
        return totals


def usage_file(root: Path) -> Path:
    from localforge import memory  # local import: memory owns the per-project folder layout

    return memory.project_dir(root) / "usage.json"


def _read(root: Path) -> dict:
    try:
        data = json.loads(usage_file(root).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def record(root: Path, session_id: str, model: str, stats) -> None:
    """Add one task's usage to this session's entry and the all-time total."""
    data = _read(root)
    sessions = [s for s in data.get("sessions", []) if isinstance(s, dict)]
    current = next((s for s in sessions if s.get("id") == session_id), None)
    if current is None:
        current = {"id": session_id, "totals": Totals(started=time.time()).to_dict()}
        sessions.append(current)
    totals = Totals.from_dict(current["totals"])
    totals.add(model, stats)
    current["totals"] = totals.to_dict()

    all_time = Totals.from_dict(data.get("all_time", {}))
    if not all_time.started:
        all_time.started = time.time()
    all_time.add(model, stats)

    path = usage_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"all_time": all_time.to_dict(), "sessions": sessions[-SESSIONS_KEPT:]}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)  # never leave a half-written file behind


def all_time(root: Path) -> Totals:
    return Totals.from_dict(_read(root).get("all_time", {}))


def session(root: Path, session_id: str) -> Totals | None:
    for entry in _read(root).get("sessions", []):
        if isinstance(entry, dict) and entry.get("id") == session_id:
            return Totals.from_dict(entry.get("totals", {}))
    return None


def previous_session(root: Path, session_id: str) -> Totals | None:
    """The most recent session that isn't this one."""
    earlier = [s for s in _read(root).get("sessions", []) if isinstance(s, dict) and s.get("id") != session_id]
    return Totals.from_dict(earlier[-1].get("totals", {})) if earlier else None
