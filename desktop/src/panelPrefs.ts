// Whether each side panel is open is a per-viewer convenience (like the recent
// folders), so it lives in localStorage -- guarded, since storage can be blocked
// or empty and the app has to work without it.
const KEY = "localforge.panel.";

export function loadPanelOpen(name: "left" | "right", fallback: boolean): boolean {
  try {
    const raw = localStorage.getItem(KEY + name);
    return raw === "open" ? true : raw === "closed" ? false : fallback;
  } catch {
    return fallback;
  }
}

export function savePanelOpen(name: "left" | "right", open: boolean): void {
  try {
    localStorage.setItem(KEY + name, open ? "open" : "closed");
  } catch {
    /* not remembered; fine */
  }
}

// With no saved choice, the right panel starts closed on a narrow window: the app
// opens at 800px, and two side panels would leave the chat almost no room.
export const NARROW_PX = 1100;
export function defaultRightOpen(): boolean {
  try {
    return window.innerWidth >= NARROW_PX;
  } catch {
    return true;
  }
}

// How wide each side panel is, and whether one fills the window, are per-viewer
// conveniences too. Widths are only saved when a drag ends (or a key is pressed).
export const PANEL_WIDTH = { left: { initial: 256, min: 200, max: 640 }, right: { initial: 288, min: 220, max: 720 } } as const;

export function clampWidth(name: "left" | "right", width: number): number {
  const { min, max } = PANEL_WIDTH[name];
  let limit: number = max;
  try {
    limit = Math.min(max, Math.floor(window.innerWidth * 0.6));  // never push the chat off the window
  } catch {
    /* no window: use the plain max */
  }
  return Math.max(min, Math.min(limit, Math.round(width)));
}

export function loadPanelWidth(name: "left" | "right"): number {
  try {
    const raw = Number(localStorage.getItem(KEY + name + ".width"));
    if (Number.isFinite(raw) && raw > 0) return clampWidth(name, raw);
  } catch {
    /* fall through */
  }
  return PANEL_WIDTH[name].initial;
}

export function savePanelWidth(name: "left" | "right", width: number): void {
  try {
    localStorage.setItem(KEY + name + ".width", String(Math.round(width)));
  } catch {
    /* not remembered; fine */
  }
}
