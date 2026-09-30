import { useEffect, useState } from "react";

const SLOW_MS = 12_000;
const STUCK_MS = 40_000;

// What to show between "folder chosen" and the backend's `ready`, and when it
// never gets there. Before this an empty, disabled chat was the only sign, so a
// backend that failed to start (or was just slow to unpack on first launch) looked
// like a dead app with no trust prompt and no explanation.
export function BackendStatus({
  folder,
  log,
  exited,
  startedAt,
  trustDeclined,
  onRetry,
}: {
  folder: string | null;
  log: string[];
  exited: boolean;
  startedAt: number | null;
  trustDeclined: boolean;
  onRetry: () => void;
}) {
  const [now, setNow] = useState(() => Date.now());
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);
  const waited = startedAt ? now - startedAt : 0;
  const tail = log.slice(-15).join("\n");

  if (trustDeclined) {
    return (
      <div className="mx-auto flex h-full max-w-lg flex-col items-center justify-center gap-3 px-6 text-center" data-testid="backend-declined">
        <p className="text-sm text-mx-mid">You chose not to trust this folder, so nothing was started and no files were touched.</p>
        <button type="button" onClick={onRetry} className="rounded-sm border border-mx-mid px-4 py-2 text-sm text-mx-green hover:border-mx-bright hover:text-mx-bright">
          Open it again
        </button>
      </div>
    );
  }

  if (exited) {
    const details = `localforge backend stopped before it was ready\nfolder: ${folder ?? ""}\n\n${tail || "(it printed nothing)"}`;
    return (
      <div className="mx-auto flex h-full max-w-xl flex-col justify-center gap-3 px-6" role="alert" data-testid="backend-failed">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-mx-red">localforge didn't start</h2>
        <p className="text-sm text-mx-mid">
          The part of the app that does the work stopped before it was ready, so there was nothing to ask about trusting this folder or setting up a model.
        </p>
        <pre className="max-h-56 overflow-auto whitespace-pre-wrap rounded-sm border border-mx-red bg-mx-panel2 p-2 font-mono text-[11px] text-mx-red" data-testid="backend-log">
          {tail || "(it printed nothing)"}
        </pre>
        <p className="text-xs text-mx-dim">
          Try again first. If it keeps happening after a fresh download, please send the details above: they say what stopped it.
        </p>
        <div className="flex gap-2">
          <button type="button" onClick={onRetry} className="rounded-sm border border-mx-mid px-4 py-1.5 text-sm text-mx-green hover:border-mx-bright hover:text-mx-bright">Try again</button>
          <button
            type="button"
            onClick={() => { navigator.clipboard?.writeText(details).then(() => setCopied(true)).catch(() => {}); }}
            className="rounded-sm border border-mx-dim px-4 py-1.5 text-sm text-mx-mid hover:border-mx-mid hover:text-mx-bright"
          >
            {copied ? "Copied" : "Copy details"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex h-full max-w-lg flex-col items-center justify-center gap-2 px-6 text-center" role="status" data-testid="backend-starting">
      <p className="animate-pulse text-sm text-mx-green">Starting localforge…</p>
      <p className="max-w-full truncate text-xs text-mx-dim" title={folder ?? ""}>{folder}</p>
      {waited > SLOW_MS && waited <= STUCK_MS && (
        <p className="text-xs text-mx-dim" data-testid="backend-slow">Still starting. The first launch can take a few seconds while the app unpacks.</p>
      )}
      {waited > STUCK_MS && (
        <div className="mt-2 space-y-2" data-testid="backend-stuck">
          <p className="text-xs text-mx-amber">This is taking much longer than normal, so the backend may not be starting.</p>
          {tail && <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-sm border border-mx-dim bg-mx-panel2 p-2 text-left font-mono text-[11px] text-mx-mid">{tail}</pre>}
          <button type="button" onClick={onRetry} className="rounded-sm border border-mx-mid px-4 py-1.5 text-sm text-mx-green hover:border-mx-bright hover:text-mx-bright">Try again</button>
        </div>
      )}
    </div>
  );
}
