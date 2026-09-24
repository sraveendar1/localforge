import { SystemStats } from "./state";

export function SystemPanel({ stats }: { stats: SystemStats | null }) {
  if (!stats) {
    return (
      <section className="w-full space-y-2 text-xs">
        <div className="text-mx-dim">No system data yet.</div>
      </section>
    );
  }

  return (
    <section className="w-full space-y-2 text-xs">
      <h2 className="uppercase tracking-wide text-mx-bright border-b border-mx-dim pb-1">System</h2>
      <div className="flex justify-between gap-2">
        <span className="text-mx-mid">OS</span>
        <span className="text-mx-green tabular-nums">{`${stats.hardware.os} ${stats.hardware.arch}`}</span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-mx-mid">CPU cores</span>
        <span className="text-mx-green tabular-nums">{stats.hardware.cpu_cores}</span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-mx-mid">CPU usage</span>
        <span className="text-mx-green tabular-nums">{stats.cpuPercent.toFixed(0)}%</span>
      </div>
      <div className="flex justify-between gap-2">
        <span className="text-mx-mid">RAM</span>
        <span className="text-mx-green tabular-nums">{`${stats.ramUsedGb.toFixed(1)} / ${stats.ramTotalGb.toFixed(1)} GB`}</span>
      </div>
      {stats.hardware.gpus && stats.hardware.gpus.length > 0 ? (
        <div className="flex justify-between gap-2">
          <span className="text-mx-mid">GPU</span>
          <span className="text-mx-green tabular-nums">{stats.hardware.gpus.join(', ')}</span>
        </div>
      ) : (
        <div className="flex justify-between gap-2">
          <span className="text-mx-mid">GPU</span>
          <span className="text-mx-green tabular-nums">None</span>
        </div>
      )}
    </section>
  );
}
