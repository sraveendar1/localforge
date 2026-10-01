import { SystemStats } from "./state";

// A labelled bar: how much of something is in use.
function Meter({ label, value, text, testId }: { label: string; value: number; text: string; testId: string }) {
  const pct = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  const tone = pct >= 90 ? "bg-mx-red" : pct >= 70 ? "bg-mx-amber" : "bg-mx-green";
  return (
    <div data-testid={testId}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-mx-mid">{label}</span>
        <span className="tabular-nums text-mx-bright">{text}</span>
      </div>
      <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-sm bg-mx-dim/50" role="progressbar" aria-label={label} aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <div className={"h-full transition-all " + tone} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[4.5rem_minmax(0,1fr)] items-baseline gap-2">
      <dt className="text-mx-mid">{label}</dt>
      <dd className="break-words text-right text-mx-bright">{children}</dd>
    </div>
  );
}

// What this computer has and how busy it is, refreshed every few seconds while
// the app is connected.
export function SystemPanel({ stats }: { stats: SystemStats | null }) {
  if (!stats) {
    return <div className="text-mx-dim italic" data-testid="system-waiting">Reading this computer…</div>;
  }
  const hw = stats.hardware ?? ({} as SystemStats["hardware"]);
  const gpus = Array.isArray(hw.gpus) ? hw.gpus : [];
  const ramPct = stats.ramTotalGb > 0 ? (stats.ramUsedGb / stats.ramTotalGb) * 100 : 0;
  return (
    <div className="space-y-3" data-testid="system-panel">
      <Meter label="CPU" value={stats.cpuPercent} text={`${stats.cpuPercent.toFixed(0)}%`} testId="system-cpu" />
      <Meter label="Memory" value={ramPct} text={`${stats.ramUsedGb.toFixed(1)} of ${stats.ramTotalGb.toFixed(1)} GB`} testId="system-ram" />
      <dl className="space-y-1 border-t border-mx-dim pt-2">
        <Row label="System">{[hw.os, hw.arch].filter(Boolean).join(" ") || "unknown"}</Row>
        <Row label="CPU cores">{hw.cpu_cores ?? "?"}</Row>
        <Row label="GPU">{gpus.length ? gpus.map(g => `${g.name} (${g.vram_gb} GB)`).join(", ") : "none"}</Row>
        {typeof hw.free_disk_gb === "number" && <Row label="Free disk">{hw.free_disk_gb} GB</Row>}
      </dl>
    </div>
  );
}
