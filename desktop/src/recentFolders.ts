// Per-viewer convenience only (which folders you've opened before, on this
// machine) -- never shared state, so localStorage is the right place for
// it rather than anything that goes through the backend.
const KEY = "localforge.recentFolders";
const MAX = 10;

export function loadRecentFolders(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((f): f is string => typeof f === "string") : [];
  } catch {
    return [];
  }
}

export function addRecentFolder(path: string, existing: string[]): string[] {
  const next = [path, ...existing.filter(f => f !== path)].slice(0, MAX);
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Private window, disabled storage, etc. -- the folder still opens, it just won't be remembered.
  }
  return next;
}
