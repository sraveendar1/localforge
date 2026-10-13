import { useState } from "react";
import { SessionNote } from "./sessionNote";
import type { MemoryState } from "./state";

const CLAMP = 220;  // characters of the goal shown before "Show more"
const PROGRESS_SHOWN = 3;  // recent-progress entries shown before "Show all"

// "2026-10-09 — I want this packaged (created requirements.txt, created run.sh, +1 more)" ->
// a headline, a short date, and "3 files". The wall of file names is what made this list unreadable.
export function summarizeProgress(line: string): { date: string; title: string; files: number } {
  const m = line.match(/^(\d{4})-(\d{2})-(\d{2})\s*[—-]\s*(.*)$/s);
  let rest = m ? m[4] : line;
  const date = m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "";
  let files = 0;
  const tail = rest.match(/\s*\(((?:created|updated|deleted|moved)[^)]*)\)\s*$/);
  if (tail) {
    rest = rest.slice(0, tail.index).trim();
    const parts = tail[1].split(",").map(x => x.trim()).filter(Boolean);
    const more = parts.map(x => x.match(/^\+(\d+) more$/)?.[1]).find(Boolean);
    files = parts.filter(x => !/^\+\d+ more$/.test(x)).length + (more ? Number(more) : 0);
  }
  return { date, title: rest.replace(/\s+/g, " ").trim(), files };
}

// Kept at the top of the sidebar on purpose (requested): the project's goal is the first
// thing worth orienting on, above the plan, active models or usage. The goal is what the
// project is for: AGENTS.md's own "What this project is" section once there is one, else
// what you first asked for, else (an older project) the goal from the last session's note.
// Recent progress is the last few lines of AGENTS.md's "Progress log". "Last session" is the
// memory model's note, read into collapsible sections instead of shown as raw markdown.
// The plan (below) is different: it is only the current task's steps.
export function GoalPanel({ memory, onItem, onGoal }: { memory: MemoryState; onItem?: (action: "done" | "reopen", index: number) => void; onGoal?: (text: string) => void }) {
  const [more, setMore] = useState(false);
  const [allProgress, setAllProgress] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const long = memory.goal.length > CLAMP;
  const goal = long && !more ? memory.goal.slice(0, CLAMP).replace(/\s+\S*$/, "") + "…" : memory.goal;
  return (
    <div className="space-y-2" data-testid="goal-panel">
      {editing ? (
        <div data-testid="goal-editor" className="space-y-1">
          <textarea
            value={draft}
            onChange={e => setDraft(e.target.value)}
            rows={4}
            maxLength={600}
            autoFocus
            placeholder="What is this project for, in a sentence or two?"
            data-testid="goal-input"
            className="w-full resize-y rounded-sm border border-mx-mid bg-mx-panel px-2 py-1 text-mx-bright placeholder:text-mx-muted"
          />
          <div className="flex gap-2">
            <button type="button" disabled={draft.trim().length < 8} data-testid="goal-save"
              onClick={() => { onGoal?.(draft.trim()); setEditing(false); }}
              className="rounded-sm border border-mx-mid px-2 py-0.5 text-[11px] text-mx-green hover:border-mx-bright disabled:opacity-50">Save</button>
            <button type="button" onClick={() => setEditing(false)} className="rounded-sm border border-mx-dim px-2 py-0.5 text-[11px] text-mx-mid hover:text-mx-bright">Cancel</button>
          </div>
        </div>
      ) : memory.goal ? (
        <div>
          <p className="whitespace-pre-wrap break-words text-mx-bright" data-testid="goal-text">{goal}</p>
          <div className="mt-0.5 flex gap-3">
            {long && (
              <button type="button" onClick={() => setMore(m => !m)} className="text-[11px] text-mx-green underline hover:text-mx-bright">
                {more ? "Show less" : "Show more"}
              </button>
            )}
            {onGoal && <button type="button" data-testid="goal-edit" onClick={() => { setDraft(memory.goal); setEditing(true); }} className="text-[11px] text-mx-muted underline hover:text-mx-bright">Edit</button>}
          </div>
        </div>
      ) : (
        <div>
          <p className="text-mx-muted italic">No goal yet. Say what the project is for, or write it yourself.</p>
          {onGoal && <button type="button" data-testid="goal-edit" onClick={() => { setDraft(""); setEditing(true); }} className="text-[11px] text-mx-green underline hover:text-mx-bright">Write the goal</button>}
        </div>
      )}
      {memory.progress.length > 0 && (
        <div data-testid="goal-progress">
          <h3 className="text-[11px] uppercase tracking-wide text-mx-mid">Recent progress</h3>
          <ul className="mt-1 space-y-1.5">
            {[...memory.progress].reverse().slice(0, allProgress ? undefined : PROGRESS_SHOWN).map((p, i) => {
              const { date, title, files } = summarizeProgress(p);
              return (
                <li key={i} className="break-words" title={p}>
                  <div className="line-clamp-2 text-mx-bright">{title}</div>
                  <div className="text-[11px] text-mx-muted">{[date, files > 0 ? `${files} file${files === 1 ? "" : "s"}` : ""].filter(Boolean).join(" · ")}</div>
                </li>
              );
            })}
          </ul>
          {memory.progress.length > PROGRESS_SHOWN && (
            <button type="button" onClick={() => setAllProgress(a => !a)} className="mt-1 text-[11px] text-mx-green underline hover:text-mx-bright" data-testid="progress-toggle">
              {allProgress ? "Show fewer" : `Show all (${memory.progress.length})`}
            </button>
          )}
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
