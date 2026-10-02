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

export function SessionNote({ note }: { note: string }) {
  const sections = parseNote(note).filter(s => !SKIP.test(s.title));
  if (sections.length === 0) return null;
  return (
    <div className="space-y-1" data-testid="session-note">
      {sections.map((s, i) => {
        const count = s.items.length;
        return (
          <details key={i} className="rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1">
            <summary className="cursor-pointer select-none text-mx-mid" title={s.title}>
              {(s.title || "Notes").replace(/\s*\(.*\)\s*$/, "")}{count > 0 && <span className="text-mx-dim"> · {count}</span>}
            </summary>
            {s.text && <p className="mt-1 break-words text-mx-dim"><Inline text={s.text} /></p>}
            {count > 0 && (
              <ul className="mt-1 space-y-1">
                {s.items.map((it, j) => (
                  <li key={j} className="flex gap-1.5 break-words text-mx-dim">
                    <span aria-hidden className="text-mx-mid">›</span>
                    <span className="min-w-0"><Inline text={it} /></span>
                  </li>
                ))}
              </ul>
            )}
          </details>
        );
      })}
    </div>
  );
}
