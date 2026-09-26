import { useState } from "react";
import type { ReactNode } from "react";

// A collapsible sidebar section ("curtain pull"): the title bar is always
// visible and toggles the body open/closed, so the sidebar can stay compact
// without losing any section entirely -- requested after the sidebar grew
// to five stacked panels (goal, plan, active LLMs, usage, system).
export function Curtain({ title, defaultOpen = true, children }: { title: string; defaultOpen?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="w-full text-xs">
      <button
        type="button"
        className="flex w-full items-center justify-between border-b border-mx-dim pb-1 uppercase tracking-wide text-mx-bright glow"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span>{title}</span>
        <span className={"text-mx-mid transition-transform " + (open ? "rotate-90" : "")}>▸</span>
      </button>
      {open && <div className="space-y-2 pt-2">{children}</div>}
    </section>
  );
}
