import { useState } from "react";
import { SessionNote } from "./sessionNote";
import type { MemoryState } from "./state";

const CLAMP = 220;  // characters of the goal shown before "Show more"

// Kept at the top of the sidebar on purpose (requested): the project's goal is the first
// thing worth orienting on, above the plan, active models or usage. The goal is what the
// project is for: AGENTS.md's own "What this project is" section once there is one, else
// what you first asked for, else (an older project) the goal from the last session's note.
// Recent progress is the last few lines of AGENTS.md's "Progress log". "Last session" is the
// memory model's note, read into collapsible sections instead of shown as raw markdown.
// The plan (below) is different: it is only the current task's steps.
export function GoalPanel({ memory, onItem }: { memory: MemoryState; onItem?: (action: "done" | "reopen", index: number) => void }) {
  const [more, setMore] = useState(false);
  const long = memory.goal.length > CLAMP;
  const goal = long && !more ? memory.goal.slice(0, CLAMP).replace(/\s+\S*$/, "") + "…" : memory.goal;
  return (
    <div className="space-y-2" data-testid="goal-panel">
      {memory.goal ? (
        <div>
          <p className="whitespace-pre-wrap break-words text-mx-bright" data-testid="goal-text">{goal}</p>
          {long && (
            <button type="button" onClick={() => setMore(m => !m)} className="mt-0.5 text-[11px] text-mx-green underline hover:text-mx-bright">
              {more ? "Show less" : "Show more"}
            </button>
          )}
        </div>
      ) : (
        <p className="text-mx-dim italic">No goal yet. What you ask for first becomes this project's goal and is kept in AGENTS.md.</p>
      )}
      {memory.progress.length > 0 && (
        <div data-testid="goal-progress">
          <h3 className="text-[11px] uppercase tracking-wide text-mx-mid">Recent progress</h3>
          <ul className="mt-1 space-y-1">
            {[...memory.progress].reverse().map((p, i) => (
              <li key={i} className="break-words text-mx-dim">{p}</li>
            ))}
          </ul>
        </div>
      )}
      {memory.narrative && (
        <div>
          <h3 className="mb-1 text-[11px] uppercase tracking-wide text-mx-mid">Last session</h3>
          <SessionNote note={memory.narrative} onItem={onItem} />
        </div>
      )}
    </div>
  );
}
