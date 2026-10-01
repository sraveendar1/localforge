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

import re
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



# --- plain English for what went wrong ---------------------------------------------------------
#
# Reported with a screenshot: "Problems: delegate_coding_task: [WARNING: the judge model flagged this
# result (The response is a request for more information...) -- verify before use, or delegate again
# with clearer/simpler instructions]". That is a message between components, not something to show a
# person. `explain_error` turns the known ones into a sentence that says what happened and what it
# means; anything unknown is still cleaned up (tool names and bracketed markers dropped) rather than
# shown raw. The original text is always kept next to it as "details".

_STEP_LABELS = {
    "delegate_coding_task": "The coding model",
    "delegate_docs_task": "The writing model",
    "delegate_general_task": "The general model",
    "generate_image": "The image model",
    "read_file": "Reading a file",
    "list_files": "Listing files",
    "search": "Searching the project",
    "edit_file": "Editing a file",
    "run_command": "Running a command",
    "web_search": "Searching the web",
    "fetch_url": "Opening a web page",
    "make_dir": "Creating a folder",
    "move_path": "Moving a file",
    "delete_path": "Deleting a file",
}


def friendly_step(tool: str, target: str = "") -> str:
    """"delegate_coding_task dia.py" -> "The coding model (dia.py)"."""
    label = _STEP_LABELS.get(tool) or tool.replace("_", " ").strip().capitalize()
    target = (target or "").strip()
    if target and len(target) <= 60 and tool.startswith(("delegate_", "generate_", "edit_", "read_", "move_", "delete_", "make_")):
        return f"{label} ({target})"
    return label


def _modality_word(text: str) -> str:
    m = re.search(r"(?<![a-z])(coding|docs?|general|image)(?![a-z])", text or "")
    return {"docs": "writing", "doc": "writing"}.get(m.group(1), m.group(1)) if m else "local"


def explain_error(text: str, tool: str = "") -> str:
    """A person-readable version of an internal error or warning. Never longer than a couple of
    sentences; unknown text is cleaned, not hidden."""
    raw = (text or "").strip()
    if not raw:
        return "Something went wrong, but no details were given."
    one = re.sub(r"\s+", " ", raw)
    low = one.lower()
    who = _modality_word(tool) if tool.startswith(("delegate_", "generate_")) else _modality_word(one)

    if m := re.search(r"the judge model flagged this (?:result|file) \((.*?)\)\s*-- verify", one):
        return (
            f"The model that double-checks the work did not accept the {who} model's reply: {m.group(1).rstrip('.')}. "
            "The orchestrator has been told, and will ask again with clearer instructions."
        )
    if m := re.search(r"this (\w+) result may be unreliable \((.*?)\)", one):
        why = m.group(2).split(";")[0]
        return f"The {_modality_word(m.group(1))} model's reply looked unusable ({why}). It was asked again; if you see this, double-check the result."
    if "hit its" in low and "token limit" in low and "cut off" in low:
        return f"The {who} model ran out of room in the middle of its answer, so the reply was cut off and not used. Ask for a smaller change, or split it into steps."
    if "doesn't parse" in low or "does not parse" in low:
        return f"The file the {who} model wrote has a syntax error, even after a second try, so nothing was saved. The orchestrator can ask again or write a smaller piece."
    if "is too large" in low and "rewrite" in low:
        return "That file is too big for the local model to rewrite in one go. Ask for a change to one part of it, or split the file."
    if "memory is short" in low or "out of memory" in low:
        return "The computer ran low on memory while the local model was working. Close other apps, or choose a smaller local model in Models."
    if "instructions contain" in low and "lines of code" in low:
        return "The orchestrator tried to write the code itself instead of describing it. It was told to describe the change and let the local model write it."
    if "edit_file failed" in low and "small fix-ups" in low:
        return "The orchestrator tried to make a large edit directly. Large changes are written by the local model instead, so it was sent back to delegate."
    if "no image model is chosen" in low:
        return "Image generation isn't set up. Choose an image model in Models (it needs an OpenAI or Gemini key)."
    if "would pass" in low and "limit" in low and "nothing was generated or billed" in low:
        return "That image would go over your monthly image budget, so it was not generated or billed. You can raise the limit in Models."
    if low.startswith("the user declined"):
        return "You declined this step, so it was not done."
    if "usage limit" in low or "session limit" in low or "rate limit" in low or "spend limit" in low:
        return f"The AI provider's usage limit was reached ({one[:160]}). Wait for it to reset, or switch models."
    if "needs a" in low and "api key" in low:
        return one  # already written for people (see serve._friendly_failure)
    if "timed out" in low or "timeout" in low:
        return f"The step took too long and was stopped ({one[:160]}). Try again, or break the request into smaller steps."
    if m := re.match(r"exit code (\d+)", one):
        return f"The command failed (exit code {m.group(1)}). Its output is under details."

    # unknown: drop only the machine markers; the words (and any error class) stay, they help to debug
    cleaned = re.sub(r"^\[(?:WARNING|ERROR):\s*", "", one)
    cleaned = re.sub(r"\s*\]$", "", cleaned)
    cleaned = re.sub(r"\s*--\s*verify before use.*$", "", cleaned)
    cleaned = re.sub(r"^\w+ failed: ", "", cleaned)
    return cleaned[:300] or "Something went wrong."


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
            problems.append({
                "what": f"{step.tool} {step.target}".strip(), "error": step.detail or "failed", "recovered": later_ok,
                "plain": f"{friendly_step(step.tool, step.target)}: {explain_error(step.detail or 'failed', step.tool)}",
            })
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
        problems.append({"what": "handled along the way", "error": text, "recovered": True, "plain": explain_error(text)})
    return {
        "outcome": outcome,
        "error": error or None,
        "plain_error": explain_error(error) if error else None,
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
            [f"  ✗ {p.get('plain') or p['what'] + ': ' + p['error'][:200]}" + (" (recovered)" if p["recovered"] else "") for p in summary["problems"]]
        )
    if summary["error"]:
        lines.append(f"Error: {summary.get('plain_error') or summary['error']}")
    return lines
