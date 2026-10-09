import type { ReactNode } from "react";
import { ExpandButton, ResizeHandle } from "./PanelResize";
import type { PanelLayout } from "./PanelResize";

// The right-hand sidebar (goal, plan, active models, usage), collapsible from a
// chevron tab on its left edge -- the mirror of the left panel's tab.
export function RightPanel({ open, onToggle, layout, children }: { open: boolean; onToggle: () => void; layout: PanelLayout; children: ReactNode }) {
  const { expanded } = layout;
  return (
    <div className={"flex h-full " + (expanded ? "min-w-0 flex-1" : "shrink-0")}>
      {!expanded && open && <ResizeHandle side="right" width={layout.width} onWidth={layout.onWidth} />}
      {!expanded && <button
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
      </button>}
      {open && (
        <aside
          className={"overflow-y-auto border-l border-mx-dim bg-mx-panel p-3 " + (expanded ? "min-w-0 flex-1" : "shrink-0")}
          style={expanded ? undefined : { width: layout.width }}
          data-testid="right-panel"
          data-expanded={expanded}
        >
          <div className="mb-2 flex justify-end"><ExpandButton expanded={expanded} onToggle={layout.onToggleExpand} label="the side panel" /></div>
          {/* Expanded, the sections flow into columns so a wide window is used, not stretched. */}
          <div className={expanded ? "gap-6 [column-width:24rem] [&>*]:mb-4 [&>*]:break-inside-avoid" : "space-y-3"}>{children}</div>
        </aside>
      )}
    </div>
  );
}
