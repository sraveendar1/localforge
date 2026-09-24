import { Usage } from "./state";

export function UsagePanel({ usage, model }: { usage: Usage; model: string }) {
  return (
    <section className="w-full space-y-4 text-xs">
      <h2 className="uppercase tracking-wide text-mx-bright glow">Session usage</h2>
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
      <section>
        <h3 className="mb-2 border-b border-mx-dim pb-1 uppercase tracking-wide text-mx-mid">Local models</h3>
        <div className="flex justify-between gap-2">
          <span className="text-mx-mid">Tokens generated</span>
          <span className="text-mx-green tabular-nums">{usage.localTokensGenerated.toLocaleString()}</span>
        </div>
        {Object.keys(usage.localModels).length === 0 ? (
          <div className="text-mx-dim italic">No delegations yet.</div>
        ) : (
          <ul className="space-y-1 mt-2">
            {Object.entries(usage.localModels)
              .sort((a, b) => b[1].tokens - a[1].tokens)
              .map(([name, { tokens, active, runs }]) => (
                <li
                  key={name}
                  className={`flex items-center gap-2${active ? " text-mx-bright" : " text-mx-mid"}`}
                >
                  {active && <span className="inline-block h-2 w-2 rounded-full bg-mx-green animate-pulse"></span>}
                  <span className="text-mx-bright">{name}</span>
                  <span className="text-mx-mid">{runs} run{runs !== 1 ? "s" : ""}</span>
                  <span className="text-mx-green tabular-nums">{tokens.toLocaleString()}</span>
                </li>
              ))}
          </ul>
        )}
      </section>
    </section>
  );
}
