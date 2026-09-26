import { Usage, UsageTotals } from "./state";

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

// Usage and cost: the orchestrator's own tokens/spend, plus this project's
// usage history. The local-model side (which models, active/run counts)
// moved to ActiveModelsPanel.tsx, per the requested sidebar order: Overall
// goal -> Pending task -> Active LLMs -> Usage and cost.
export function UsagePanel({ usage, model, usageHistory }: { usage: Usage; model: string; usageHistory: { previousSession: UsageTotals | null; allTime: UsageTotals | null } }) {
  return (
    <>
      <section>
        <h3 className="mb-2 border-b border-mx-dim pb-1 uppercase tracking-wide text-mx-mid">Frontier model</h3>
        <div className="text-mx-bright">
          {model || "—"}
        </div>
        <div className="flex justify-between gap-2">
          <span className="text-mx-mid">Prompt</span>
          <span className="text-mx-green tabular-nums">{usage.frontierPromptTokens.toLocaleString()}</span>
        </div>
        <div className="flex justify-between gap-2">
          <span className="text-mx-mid">Completion</span>
          <span className="text-mx-green tabular-nums">{usage.frontierCompletionTokens.toLocaleString()}</span>
        </div>
        <div className="flex justify-between gap-2">
          <span className="text-mx-mid">Total</span>
          <span className="text-mx-green tabular-nums">{(usage.frontierPromptTokens + usage.frontierCompletionTokens).toLocaleString()}</span>
        </div>
        <div className="flex justify-between gap-2">
          <span className="text-mx-mid">Cost</span>
          <span className="text-mx-green tabular-nums">
            {usage.frontierViaSubscription ? "included in subscription" : `$${usage.frontierCostUsd.toFixed(4)}`}
          </span>
        </div>
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
