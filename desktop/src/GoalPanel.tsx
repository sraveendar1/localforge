import type { MemoryState } from "./state";

// Kept at the top of the sidebar on purpose (requested): the project's
// "what this project is" (from AGENTS.md's own goals section) is the first
// thing worth orienting on, above the plan, active models or usage. The
// outer title/header lives in the Curtain wrapper (App.tsx), not here.
export function GoalPanel({ memory }: { memory: MemoryState }) {
  return (
    <>
      {memory.goal ? (
        <p className="text-mx-mid whitespace-pre-wrap">{memory.goal}</p>
      ) : (
        <p className="text-mx-dim italic">No AGENTS.md yet for this project — run <span className="text-mx-mid">/goals</span> to draft one from what's here.</p>
      )}
      {memory.narrative && (
        <>
          <h3 className="mt-2 text-[11px] uppercase tracking-wide text-mx-mid">Last session</h3>
          <p className="text-mx-dim">{memory.narrative}</p>
        </>
      )}
    </>
  );
}
