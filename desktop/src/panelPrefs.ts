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
