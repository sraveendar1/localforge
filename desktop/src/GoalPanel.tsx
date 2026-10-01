import type { MemoryState } from "./state";

// Kept at the top of the sidebar on purpose (requested): the project's goal is the first
// thing worth orienting on, above the plan, active models or usage. The goal is what the
// project is for: AGENTS.md's own "What this project is" section once there is one, and
// until then what you first asked for (the first request typed in a project is recorded as
// its goal). Recent progress is the last few lines of AGENTS.md's "Progress log", which
// grows by a line each time a task changes something. The plan (below) is different: it is
// only the current task's steps. The outer title/header lives in the Curtain wrapper.
export function GoalPanel({ memory }: { memory: MemoryState }) {
  return (
    <div className="space-y-2" data-testid="goal-panel">
      {memory.goal ? (
        <p className="whitespace-pre-wrap text-mx-bright" data-testid="goal-text">{memory.goal}</p>
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
          <h3 className="text-[11px] uppercase tracking-wide text-mx-mid">Last session</h3>
          <p className="text-mx-dim">{memory.narrative}</p>
        </div>
      )}
    </div>
  );
}
