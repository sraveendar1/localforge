import type { PaidModel } from "./paidModels";
import { PaidUsage, Usage, UsageTotals } from "./state";

function HistoryRow({ label, totals }: { label: string; totals: UsageTotals }) {
  const share = totals.localTokensGenerated * 100 / Math.max(totals.localTokensGenerated + totals.frontierPromptTokens + totals.frontierCompletionTokens, 1);
  const cost = totals.frontierCostUsd ? `$${totals.frontierCostUsd.toFixed(2)}` : totals.subscriptionCostUsd ? `~$${totals.subscriptionCostUsd.toFixed(2)} subscription` : "-";
  return (
    <div className="flex justify-between gap-2">
      <span className="text-mx-mid">{label}</span>
      <span className="text-mx-green tabular-nums">{totals.tasks} task{totals.tasks !== 1 ? "s" : ""} · {share.toFixed(0)}% local · {cost}</span>
    </div>
  );
}

function cost(u: PaidUsage | undefined): string {
  if (!u) return "-";
  if (u.costUsd) return `$${u.costUsd.toFixed(4)}`;
  if (u.viaSubscription || u.notionalCostUsd) return "included in subscription";
  return u.runs ? "$0.0000" : "-";
}

// Usage and cost: every paid model in play with what it has used this
// session, then this project's usage history. It used to be one "Frontier
// model" block for the orchestrator, plus a lump for paid delegates -- but the
// orchestrator is no longer the only paid model (any task type can go to one,
// and image generation always does), so each is listed by name with its role:
// "claude-opus-5 -- orchestrator, coding", "gpt-image-1 -- image". The local
// side (which models, active/run counts) is in ActiveModelsPanel.tsx.
export function UsagePanel({
  usage,
  configuredPaid,
  usageHistory,
}: {
  usage: Usage;
  configuredPaid: PaidModel[];
  usageHistory: { previousSession: UsageTotals | null; allTime: UsageTotals | null };
}) {
  // Configured ones first (even before they've been used), then any that were
  // used but are no longer configured (a model switched away from mid-session).
  const rows: { name: string; roles: string[]; how: string }[] = configuredPaid.map(m => ({ ...m }));
  for (const [name, u] of Object.entries(usage.paidModels)) {
    const row = rows.find(r => r.name === name);
    if (row) row.roles = [...new Set([...row.roles, ...u.roles])];
    else rows.push({ name, roles: u.roles, how: "" });
  }
  return (
    <>
      <section>
        <h3 className="mb-2 border-b border-mx-dim pb-1 uppercase tracking-wide text-mx-mid">Paid models</h3>
        {rows.length === 0 ? (
          <p className="text-mx-dim italic">None in use: everything runs on local models.</p>
        ) : (
          <div className="space-y-2" data-testid="paid-models">
            {rows.map(r => {
              const u = usage.paidModels[r.name];
              return (
                <div key={r.name} data-testid={`paid-${r.name}`}>
                  <div className="break-words text-mx-bright">{r.name}</div>
                  <div className="text-mx-dim">{r.roles.join(" · ")}{r.how && ` · ${r.how}`}</div>
                  {u && u.promptTokens > 0 ? (
                    <>
                      <div className="flex justify-between gap-2">
                        <span className="text-mx-mid">Prompt</span>
                        <span className="text-mx-green tabular-nums">{u.promptTokens.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between gap-2">
                        <span className="text-mx-mid">Completion</span>
                        <span className="text-mx-green tabular-nums">{u.completionTokens.toLocaleString()}</span>
                      </div>
                    </>
                  ) : (
                    u && u.completionTokens > 0 && (
                      <div className="flex justify-between gap-2">
                        <span className="text-mx-mid">Tokens</span>
                        <span className="text-mx-green tabular-nums">{u.completionTokens.toLocaleString()}</span>
                      </div>
                    )
                  )}
                  <div className="flex justify-between gap-2">
                    <span className="text-mx-mid">{u && u.runs > 0 && !u.promptTokens && !u.completionTokens ? `Cost (${u.runs} call${u.runs === 1 ? "" : "s"})` : "Cost"}</span>
                    <span className="text-mx-green tabular-nums">{cost(u)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
      {(usageHistory.previousSession || usageHistory.allTime) && (
        <section>
          <h3 className="mb-2 border-b border-mx-dim pb-1 uppercase tracking-wide text-mx-mid">History for this project</h3>
          {usageHistory.previousSession && <HistoryRow label="Previous session" totals={usageHistory.previousSession} />}
          {usageHistory.allTime && <HistoryRow label="All time" totals={usageHistory.allTime} />}
        </section>
      )}
    </>
  );
}
