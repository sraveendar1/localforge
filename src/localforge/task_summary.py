"""What a task did, in plain words, shown after every task.

Reported: "after every work there should be a summary of activities performed,
and if there is an error that should also be printed out." A task used to end
with just the answer, so what actually happened (which files, which commands,
which model wrote what, what failed on the way) had to be pieced together from
scrolled-away activity lines, and a failure could be just one red line.

`TaskLog` is filled by `orchestrator._loop` as steps finish -- whether they
worked, failed or were declined -- and hangs off `RunStats.log`, so it reaches
the caller on every ending: an answer, a stop, a cancel, or an error carrying
the stats. `build()` turns it into a plain dict (the desktop app receives it
as-is over the protocol) and `render_lines()` turns that dict into the
terminal text, so the two can't drift apart.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

CHANGED = ("Created ", "Updated ", "Deleted ", "Moved ")
DECLINED = "The user declined"
READ_TOOLS = {"read_file", "list_files", "search"}
WEB_TOOLS = {"web_search", "fetch_url"}
FILE_TOOLS = {"edit_file", "make_dir", "move_path", "delete_path"}
MAX_LISTED = 12  # per section; the rest is counted, not listed


@dataclass
class Step:
    tool: str
    target: str  # the file, command, query... a one-line label of what it was about
    outcome: str  # "ok" | "failed" | "declined"
    detail: str = ""  # the result's first line for a success, the error for a failure
    model: str | None = None  # the model that wrote it, for a delegation
    kind: str = ""  # "file" | "delegate" | "command" | "read" | "web" | "other"


@dataclass
class TaskLog:
    started_at: float = field(default_factory=time.time)
    steps: list[Step] = field(default_factory=list)
    # Trouble that was handled without stopping the task (an orchestrator call
    # that had to be retried, a limit waited out, a step the local model redid).
    notes: list[str] = field(default_factory=list)

    def record(self, tool: str, target: str, outcome: str, detail: str = "", model: str | None = None) -> None:
        self.steps.append(Step(tool, target, outcome, _first_line(detail), model, _kind(tool)))

    def note(self, text: str) -> None:
        if text and text not in self.notes:
            self.notes.append(text)


def _first_line(text: str) -> str:
    text = (text or "").strip()
    return text.splitlines()[0][:300] if text else ""


def _kind(tool: str) -> str:
    if tool.startswith("delegate_") or tool == "generate_image":
        return "delegate"
    if tool == "run_command":
        return "command"
    if tool in FILE_TOOLS:
        return "file"
    if tool in READ_TOOLS:
        return "read"
    if tool in WEB_TOOLS:
        return "web"
    return "other"


def outcome_of(result: str | None, error: str | None) -> str:
    if error is not None:
        return "failed"
    if (result or "").lstrip().startswith(DECLINED):
        return "declined"
    return "ok"


def _command_failed(step: Step) -> bool:
    return step.kind == "command" and step.outcome == "ok" and step.detail.startswith("exit code") and not step.detail.startswith("exit code 0")


def build(log: TaskLog | None, outcome: str = "completed", error: str | None = None) -> dict:
    """The summary as plain data. `outcome` is "completed", "stopped"
    (cancelled, or halted with the work kept) or "failed"; `error` is the
    message that ended it, when one did."""
    steps = list(log.steps) if log else []
    files, delegations, commands, problems, declined = [], [], [], [], []
    reads = web = 0
    for i, step in enumerate(steps):
        if step.outcome == "declined":
            declined.append({"what": step.target or step.tool, "tool": step.tool})
            continue
        if step.outcome == "failed" or _command_failed(step):
            # "recovered" when the very same kind of step succeeded afterwards
            # (a retry, or a different attempt at the same target).
            later_ok = any(s.tool == step.tool and s.target == step.target and s.outcome == "ok" and not _command_failed(s) for s in steps[i + 1 :])
            problems.append({"what": f"{step.tool} {step.target}".strip(), "error": step.detail or "failed", "recovered": later_ok})
            if step.kind == "command":
                commands.append({"command": step.target, "ok": False, "detail": step.detail})
            continue
        if step.kind == "delegate":
            delegations.append({"tool": step.tool, "model": step.model or "", "path": step.target, "detail": step.detail})
            if step.detail.startswith(CHANGED):
                files.append({"action": step.detail.split(" ", 1)[0], "path": step.target or step.detail.split(" ", 1)[-1], "by": step.model or ""})
        elif step.kind == "file" and step.detail.startswith(CHANGED):
            files.append({"action": step.detail.split(" ", 1)[0], "path": step.target, "by": ""})
        elif step.kind == "command":
            commands.append({"command": step.target, "ok": True, "detail": step.detail})
        elif step.kind == "read":
            reads += 1
        elif step.kind == "web":
            web += 1
    for text in log.notes if log else []:
        problems.append({"what": "handled along the way", "error": text, "recovered": True})
    return {
        "outcome": outcome,
        "error": error or None,
        "duration_s": round(time.time() - log.started_at, 1) if log else 0.0,
        "steps": len(steps),
        "files": files,
        "delegations": delegations,
        "commands": commands,
        "reads": reads,
        "web": web,
        "declined": declined,
        "problems": problems,
    }


def is_empty(summary: dict) -> bool:
    """A plain question answered with no tools used has nothing to summarise."""
    return not summary.get("steps") and not summary.get("error") and not summary.get("problems")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _clip(items: list[str], limit: int = MAX_LISTED) -> list[str]:
    return items[:limit] + ([f"... and {len(items) - limit} more"] if len(items) > limit else [])


def render_lines(summary: dict) -> list[str]:
    """Plain text lines (no markup) for the terminal."""
    heading = {"completed": "Done", "stopped": "Stopped", "failed": "Failed"}.get(summary["outcome"], "Finished")
    seconds = summary.get("duration_s") or 0
    took = f" in {int(seconds // 60)}m {int(seconds % 60):02d}s" if seconds >= 60 else f" in {seconds:g}s" if seconds >= 1 else ""
    lines = [f"{heading}{took} · {_plural(summary['steps'], 'step')}"]

    if summary["files"]:
        lines.append("Files changed:")
        lines += _clip([f"  {f['action'].lower()} {f['path']}" + (f"  (written by {f['by']})" if f["by"] else "") for f in summary["files"]])
    if summary["delegations"]:
        by_model: dict[str, int] = {}
        for d in summary["delegations"]:
            by_model[d["model"] or "a local model"] = by_model.get(d["model"] or "a local model", 0) + 1
        lines.append("Delegated to: " + ", ".join(f"{m} ×{n}" for m, n in by_model.items()))
    if summary["commands"]:
        lines.append("Commands run:")
        lines += _clip([f"  {'✓' if c['ok'] else '✗'} {c['command'][:100]}" for c in summary["commands"]])
    looked = []
    if summary["reads"]:
        looked.append(f"looked through the project {_plural(summary['reads'], 'time')}")
    if summary["web"]:
        looked.append(f"searched the web {_plural(summary['web'], 'time')}")
    if looked:
        lines.append("Also " + " and ".join(looked) + ".")
    if summary["declined"]:
        lines.append("You declined:")
        lines += _clip([f"  {d['what'][:100]}" for d in summary["declined"]])
    if not (summary["files"] or summary["commands"] or summary["delegations"]) and summary["steps"] and not looked:
        lines.append("No files were changed and no commands were run.")

    if summary["problems"]:
        lines.append("Problems:")
        lines += _clip(
            [f"  ✗ {p['what']}: {p['error'][:200]}" + (" (recovered)" if p["recovered"] else "") for p in summary["problems"]]
        )
    if summary["error"]:
        lines.append(f"Error: {summary['error']}")
    return lines
