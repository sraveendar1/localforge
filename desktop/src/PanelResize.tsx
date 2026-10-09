import { useRef } from "react";
import { PANEL_WIDTH, clampWidth } from "./panelPrefs";

// What a side panel needs to be resizable and expandable (owned by App, which remembers it).
export type PanelLayout = {
  width: number;
  expanded: boolean;
  onWidth: (width: number, commit: boolean) => void;  // commit = the drag ended: remember it
  onToggleExpand: () => void;
};

const STEP = 16;

// A thin draggable divider on a side panel's inner edge. Drag it, use the arrow keys when it has
// focus, or double-click to put the width back. `grows` is the direction that widens the panel.
export function ResizeHandle({ side, width, onWidth }: { side: "left" | "right"; width: number; onWidth: PanelLayout["onWidth"] }) {
  const start = useRef<{ x: number; w: number } | null>(null);
  const sign = side === "left" ? 1 : -1;  // the left panel widens as you drag right; the right one as you drag left
  const move = (clientX: number, commit: boolean) => {
    if (!start.current) return;
    onWidth(clampWidth(side, start.current.w + sign * (clientX - start.current.x)), commit);
  };
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={`Resize ${side} panel`}
      aria-valuemin={PANEL_WIDTH[side].min}
      aria-valuemax={PANEL_WIDTH[side].max}
      aria-valuenow={width}
      tabIndex={0}
      title="Drag to resize · double-click to reset"
      data-testid={`resize-${side}`}
      className="group relative z-10 w-1.5 shrink-0 cursor-col-resize touch-none bg-transparent hover:bg-mx-mid focus:bg-mx-mid focus:outline-none"
      onPointerDown={e => {
        start.current = { x: e.clientX, w: width };
        e.currentTarget.setPointerCapture(e.pointerId);
        document.body.style.userSelect = "none";
      }}
      onPointerMove={e => move(e.clientX, false)}
      onPointerUp={e => { move(e.clientX, true); start.current = null; document.body.style.userSelect = ""; }}
      onPointerCancel={() => { start.current = null; document.body.style.userSelect = ""; }}
      onDoubleClick={() => onWidth(PANEL_WIDTH[side].initial, true)}
      onKeyDown={e => {
        const dir = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!dir) return;
        e.preventDefault();
        onWidth(clampWidth(side, width + dir * sign * STEP), true);
      }}
    >
      <span className="pointer-events-none absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-mx-dim group-hover:bg-mx-bright" />
    </div>
  );
}

// "Expand" fills the whole window with this panel (the chat steps aside); "Restore" brings it back.
export function ExpandButton({ expanded, onToggle, label }: { expanded: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      title={expanded ? "Back to the normal layout (Esc)" : `Expand ${label} to fill the window`}
      aria-label={expanded ? `Restore ${label}` : `Expand ${label}`}
      data-testid="expand-panel"
      className="flex shrink-0 items-center gap-1 rounded-sm border border-mx-dim bg-mx-panel2 px-1.5 py-0.5 text-[11px] text-mx-mid hover:border-mx-mid hover:text-mx-bright"
    >
      <span aria-hidden>{expanded ? "⤡" : "⤢"}</span>
      {expanded ? "Restore" : "Expand"}
    </button>
  );
}
