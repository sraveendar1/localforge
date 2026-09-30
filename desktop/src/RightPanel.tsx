import type { ReactNode } from "react";

// The right-hand sidebar (goal, plan, active models, usage), collapsible from a
// chevron tab on its left edge -- the mirror of the left panel's tab.
export function RightPanel({ open, onToggle, children }: { open: boolean; onToggle: () => void; children: ReactNode }) {
  return (
    <div className="flex h-full shrink-0">
      <button
        type="button"
        title={open ? "Hide goal, plan, models & usage" : "Show goal, plan, models & usage"}
        aria-label={open ? "Hide right panel" : "Show right panel"}
        aria-expanded={open}
        onClick={onToggle}
        className="group flex w-6 shrink-0 items-center justify-center border-l border-mx-dim bg-mx-panel2 hover:bg-mx-dim"
      >
        <span className="flex h-14 w-5 items-center justify-center rounded-full border border-mx-mid bg-mx-panel text-base font-bold text-mx-green glow group-hover:border-mx-bright group-hover:text-mx-bright">
          {open ? "›" : "‹"}
        </span>
      </button>
      {open && (
        <aside className="w-72 shrink-0 space-y-3 overflow-y-auto border-l border-mx-dim bg-mx-panel p-3" data-testid="right-panel">
          {children}
        </aside>
      )}
    </div>
  );
}
