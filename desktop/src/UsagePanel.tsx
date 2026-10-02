import type { PaidModel } from "./paidModels";
import { PaidUsage, Usage, UsageTotals } from "./state";

const SECTION = "mb-1 text-[10px] uppercase tracking-wide text-mx-dim";
const CARD = "rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1.5";

// 1,381 -> "1.4k": the exact figure is in the tooltip.
function compact(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

function Stat({ label, value, title, tone = "text-mx-bright" }: { label: string; value: string; title?: string; tone?: string }) {
  return (
    <div className="min-w-0" title={title}>
      <div className="text-[10px] uppercase tracking-wide text-mx-dim">{label}</div>
      <div className={"truncate tabular-nums " + tone}>{value}</div>
    </div>
  );
}

// What a paid model cost this session. A subscription's usage is never shown as money charged:
// it is "included", with what it would have cost kept for the history.
function costText(u: PaidUsage | undefined): { text: string; tone: string } {
  if (!u || (!u.runs && !u.costUsd && !u.notionalCostUsd)) return { text: "–", tone: "text-mx-dim" };
  if (u.costUsd) return { text: `$${u.costUsd.toFixed(u.costUsd < 1 ? 4 : 2)}`, tone: "text-mx-amber" };
  if (u.viaSubscription || u.notionalCostUsd) return { text: "in plan", tone: "text-mx-green" };
  return { text: "$0.00", tone: "text-mx-bright" };
}

function PaidCard({ name, roles, how, u }: { name: string; roles: string[]; how: string; u: PaidUsage | undefined }) {
  const used = !!u && (u.promptTokens > 0 || u.completionTokens > 0 || u.runs > 0);
  const c = costText(u);
  return (
    <li className={CARD} data-testid={`paid-${name}`}>
      <div className="break-words font-medium text-mx-bright">{name}</div>
      <div className="mt-1 flex flex-wrap items-center gap-1">
        {roles.map(r => (
          <span key={r} className="rounded-sm border border-mx-amber/60 px-1 text-[10px] uppercase tracking-wide text-mx-amber">{r}</span>
        ))}
        {how && <span className="text-[11px] text-mx-dim">{how}</span>}
      </div>
      {used ? (
        <div className="mt-1.5 grid grid-cols-3 gap-2 border-t border-mx-dim pt-1.5">
          {u!.promptTokens > 0 ? (
            <>
              <Stat label="Prompt" value={compact(u!.promptTokens)} title={`${u!.promptTokens.toLocaleString()} tokens`} />
              <Stat label="Reply" value={compact(u!.completionTokens)} title={`${u!.completionTokens.toLocaleString()} tokens`} />
            </>
          ) : (
            <>
              <Stat label="Calls" value={String(u!.runs)} />
              <Stat label="Tokens" value={u!.completionTokens ? compact(u!.completionTokens) : "–"} title={`${u!.completionTokens.toLocaleString()} tokens`} />
            </>
          )}
          <Stat label="Cost" value={c.text} tone={c.tone} title={c.text === "in plan" ? "Included in your subscription: nothing is charged per use" : undefined} />
        </div>
      ) : (
        <div className="mt-1 text-[11px] italic text-mx-dim">Not used yet this session.</div>
      )}
    </li>
  );
}

function HistoryCard({ label, totals }: { label: string; totals: UsageTotals }) {
  const frontier = totals.frontierPromptTokens + totals.frontierCompletionTokens;
  const total = totals.localTokensGenerated + frontier;
  const share = total > 0 ? (totals.localTokensGenerated * 100) / total : 0;
  const money = totals.frontierCostUsd
    ? { text: `$${totals.frontierCostUsd.toFixed(2)}`, note: "billed", tone: "text-mx-amber" }
    : totals.subscriptionCostUsd
      ? { text: `~$${totals.subscriptionCostUsd.toFixed(2)}`, note: "in your plan, not charged", tone: "text-mx-green" }
      : { text: "–", note: "", tone: "text-mx-dim" };
  return (
    <div className={CARD} data-testid={`history-${label.toLowerCase().replace(/\s+/g, "-")}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wide text-mx-mid">{label}</span>
        <span className="tabular-nums text-mx-bright">{totals.tasks} task{totals.tasks === 1 ? "" : "s"}</span>
      </div>
      <div className="mt-1.5">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[11px] text-mx-dim">Written locally</span>
          <span className="tabular-nums text-mx-bright">{share.toFixed(0)}%</span>
        </div>
        <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-sm bg-mx-dim/50" role="progressbar" aria-label="Share written locally" aria-valuenow={Math.round(share)} aria-valuemin={0} aria-valuemax={100}>
          <div className="h-full bg-mx-green" style={{ width: `${share}%` }} />
        </div>
      </div>
      <div className="mt-1.5 flex items-baseline justify-between gap-2">
        <span className="text-[11px] text-mx-dim">Cost{money.note && ` · ${money.note}`}</span>
        <span className={"tabular-nums " + money.tone}>{money.text}</span>
      </div>
    </div>
  );
}

// Usage and cost: every paid model in play with what it has used this session, then this
// project's history. Each is a small card like Active LLMs: the model, its role(s) and how it's
// reached, then its numbers in a row. The local side (which models, active/run counts) is in
// ActiveModelsPanel.tsx.
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
    <div className="space-y-3" data-testid="usage-panel">
      <section>
        <h3 className={SECTION}>Paid models · this session</h3>
        {rows.length === 0 ? (
          <p className="text-mx-dim italic">None in use.</p>
        ) : (
          <ul className="space-y-1.5" data-testid="paid-models">
            {rows.map(r => <PaidCard key={r.name} name={r.name} roles={r.roles} how={r.how} u={usage.paidModels[r.name]} />)}
          </ul>
        )}
      </section>
      {(usageHistory.previousSession || usageHistory.allTime) && (
        <section>
          <h3 className={SECTION}>History for this project</h3>
          <div className="space-y-1.5">
            {usageHistory.previousSession && <HistoryCard label="Previous session" totals={usageHistory.previousSession} />}
            {usageHistory.allTime && <HistoryCard label="All time" totals={usageHistory.allTime} />}
          </div>
        </section>
      )}
    </div>
  );
}
