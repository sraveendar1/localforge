import { useEffect, useState } from "react";
import type { Activity } from "./state";

// The desktop counterpart of the CLI's live status line (background.py's
// TaskRunner.toolbar):
//
//   🔨 Forging with qwen2.5-coder:7b… (2m 14s · ↓ 3.1k tokens · writing coding, 41 tok/s)
//
// Reported: while the GUI worked there was nothing like it -- just a pulsing
// dot -- so a slow step looked like a hang. One verb, and the model actually
// working right now, so it always answers "who is doing something?" at a glance.

// Same threshold as the CLI's QUIET_AFTER_SECONDS: past this, say how long it's been silent.
const QUIET_AFTER_MS = 20_000;

export function elapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

export function thousands(n: number): string {
  return n < 1000 ? String(n) : `${(n / 1000).toFixed(1)}k`;
}

export function ForgingIndicator({
  activity,
  waitingFor,
  queued,
}: {
  activity: Activity | null;
  waitingFor?: string;
  queued: number;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, []);

  if (waitingFor) {
    return (
      <div className="flex items-center gap-2 border-t border-mx-amber bg-mx-panel px-3 py-1 text-xs text-mx-amber" role="status">
        <span aria-hidden>⏸</span>
        <span>Waiting for you: {waitingFor}</span>
      </div>
    );
  }
  if (!activity) return null;

  const tokens = activity.finishedTokens + activity.localTokens + Math.ceil(activity.answerChars / 4);
  const parts = [elapsed(now - activity.startedAt), `↓ ${thousands(tokens)} tokens`];
  let detail = activity.detail;
  if (activity.localStartedAt !== null && activity.localTokens > 0) {
    const rate = activity.localTokens / Math.max((now - activity.localStartedAt) / 1000, 0.001);
    detail += `, ${Math.round(rate)} tok/s`;
  }
  const quietFor = now - activity.lastEventAt;
  if (quietFor > QUIET_AFTER_MS) detail += `, quiet for ${elapsed(quietFor)}`;
  parts.push(detail);

  return (
    <div className="flex items-center gap-2 border-t border-mx-dim bg-mx-panel px-3 py-1 text-xs text-mx-green" role="status" aria-live="polite">
      <span className="forge-hammer" aria-hidden>🔨</span>
      <span className="forge-spark" aria-hidden>✦</span>
      <span className="truncate">
        Forging with <span className="text-mx-bright">{activity.who || "a model"}</span>…{" "}
        <span className="text-mx-mid">({parts.join(" · ")})</span>
        {queued > 0 && <span className="text-mx-dim"> · queue: {queued}</span>}
      </span>
    </div>
  );
}
