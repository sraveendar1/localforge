import type { ReactNode } from "react";

// The note the memory model writes at the end of a session is markdown (`## Goal`, bullets with
// **bold** and `code`). Shown raw it was a wall of text with stray asterisks; this reads it into
// sections and draws them (no markdown library needed for this small subset).

export type NoteSection = { title: string; items: string[]; text: string };

const BULLET = /^\s*(?:[*\-•]|\d+[.)])\s+(.*)$/;

export function parseNote(note: string): NoteSection[] {
  // tolerate a note whose line breaks were lost: "## A text ## B * x * y"
  let text = (note ?? "").replace(/\r/g, "").trim();
  if (!text.includes("\n")) text = text.replace(/\s+(##\s)/g, "\n$1").replace(/\s+([*•])\s+(?=\S)/g, "\n$1 ");
  const sections: NoteSection[] = [];
  let cur: NoteSection | null = null;
  const start = (title: string) => { cur = { title, items: [], text: "" }; sections.push(cur); };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    if (!line.trim()) continue;
    const h = line.match(/^#{1,3}\s+(.*)$/);
    if (h) { start(h[1].trim()); continue; }
    if (!cur) start("");
    const b = line.match(BULLET);
    if (b) cur!.items.push(b[1].trim());
    else cur!.text += (cur!.text ? " " : "") + line.trim();
  }
  return sections.filter(s => s.items.length || s.text);
}

// **bold** and `code`, nothing else
export function Inline({ text }: { text: string }): ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**") && p.endsWith("**") && p.length > 4 ? <strong key={i} className="font-semibold text-mx-mid"><Inline text={p.slice(2, -2)} /></strong>
        : p.startsWith("`") && p.endsWith("`") && p.length > 2 ? <code key={i} className="break-all rounded-sm bg-mx-dim/40 px-1 font-mono text-[11px] text-mx-green">{p.slice(1, -1)}</code>
        : <span key={i}>{p}</span>)}
    </>
  );
}

// "Goal" is shown above as the project's goal; the rest are collapsible, closed by default.
const SKIP = /^(goal|objective|project goal)$/i;

const kindOf = (title: string): "open" | "done" | null => {
  const t = title.replace(/\s*\(.*\)\s*$/, "").trim().toLowerCase();
  return t === "open items" || t === "open" ? "open" : t === "completed items" || t === "completed" || t === "done" ? "done" : null;
};

// `onItem` lets a person tick an open item off (or reopen a finished one) when localforge hasn't
// noticed it was done; the same move happens by itself after a task that completes an item.
export function SessionNote({ note, onItem }: { note: string; onItem?: (action: "done" | "reopen", index: number) => void }) {
  const sections = parseNote(note).filter(s => !SKIP.test(s.title));
  if (sections.length === 0) return null;
  return (
    <div className="space-y-1" data-testid="session-note">
      {sections.map((s, i) => {
        const count = s.items.length;
        return (
          <details key={i} open={kindOf(s.title) === "open" && count > 0 ? true : undefined} className="rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1" data-testid={`note-${kindOf(s.title) ?? "section"}`}>
            <summary className="cursor-pointer select-none text-mx-mid" title={s.title}>
              {(s.title || "Notes").replace(/\s*\(.*\)\s*$/, "")}{count > 0 && <span className="text-mx-dim"> · {count}</span>}
            </summary>
            {s.text && <p className="mt-1 break-words text-mx-dim"><Inline text={s.text} /></p>}
            {count > 0 && (
              <ul className="mt-1 space-y-1">
                {s.items.map((it, j) => {
                  const kind = kindOf(s.title);
                  return (
                    <li key={j} className="flex gap-1.5 break-words text-mx-dim">
                      <span aria-hidden className={kind === "done" ? "text-mx-green" : "text-mx-mid"}>{kind === "done" ? "✓" : "›"}</span>
                      <span className={"min-w-0 flex-1 " + (kind === "done" ? "line-through decoration-mx-dim" : "")}><Inline text={it} /></span>
                      {onItem && kind && (
                        <button
                          type="button"
                          title={kind === "open" ? "Mark this as done" : "Move it back to open items"}
                          aria-label={kind === "open" ? "Mark done" : "Reopen"}
                          data-testid={kind === "open" ? "item-done" : "item-reopen"}
                          onClick={() => onItem(kind === "open" ? "done" : "reopen", j)}
                          className="h-5 shrink-0 rounded-sm border border-mx-dim px-1 text-[11px] text-mx-mid hover:border-mx-mid hover:text-mx-bright"
                        >
                          {kind === "open" ? "✓ Done" : "↺"}
                        </button>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </details>
        );
      })}
    </div>
  );
}
