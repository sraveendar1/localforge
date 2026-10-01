import type { PaidModel } from "./paidModels";
import { ConfiguredModel, LocalModelTarget, Usage } from "./state";
import { splitDescription } from "./describe";

const MODALITY_ORDER = ["coding", "docs", "general"];

const SECTION = "mb-1 text-[10px] uppercase tracking-wide text-mx-dim";
const CARD = "rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1.5";

// "Active LLMs at play", split by what each costs: the paid models in use
// (the orchestrator, and any task type -- image always -- sent to a paid model,
// each with its role) and the free local ones (with a pulsing dot on the one
// generating, and this session's run counts). Live status only -- choosing the
// models lives in the centre-of-window model screen ("Change models…"), so this
// one never has settings mixed into it. Each model is its own small card: the
// name on its own line, then what it's for and how it's reached, so long names
// wrap cleanly instead of pushing columns around.
export function ActiveModelsPanel({
  usage,
  configuredModels,
  localModelTargets,
  configuredPaid,
  orchestrator,
}: {
  usage: Usage;
  configuredModels: ConfiguredModel[];
  localModelTargets: { [modality: string]: LocalModelTarget };
  configuredPaid: PaidModel[];
  orchestrator: string;
}) {
  const byModality = Object.fromEntries(configuredModels.map(m => [m.modality, m]));
  const isPaidTarget = (t: string) => t.startsWith("api:") || t.startsWith("cli:");
  // Task types that stay local (automatic or a pinned local model); the ones
  // sent to a paid model are listed under Paid instead.
  const local = MODALITY_ORDER.filter(m => localModelTargets[m] && !isPaidTarget(localModelTargets[m].target));
  const localOrchestrator = orchestrator.startsWith("ollama/") || orchestrator.startsWith("ollama_chat/");

  return (
    <div className="space-y-3" data-testid="active-models">
      <div>
        <div className={SECTION}>Paid models</div>
        {configuredPaid.length === 0 ? (
          <div className="text-mx-dim italic">None in use.</div>
        ) : (
          <ul className="space-y-1.5" data-testid="active-paid">
            {configuredPaid.map(m => (
              <li key={m.name} className={CARD}>
                <div className="break-words font-medium text-mx-bright">{m.name}</div>
                <div className="mt-1 flex flex-wrap items-center gap-1">
                  {m.roles.map(r => (
                    <span key={r} className="rounded-sm border border-mx-amber/60 px-1 text-[10px] uppercase tracking-wide text-mx-amber">{r}</span>
                  ))}
                  {m.how && <span className="text-[11px] text-mx-dim">{m.how}</span>}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <div className={SECTION}>Local models (free)</div>
        {local.length > 0 || localOrchestrator ? (
          <ul className="space-y-1.5" data-testid="active-local">
            {localOrchestrator && (
              <li className={CARD}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] uppercase tracking-wide text-mx-mid">orchestrator</span>
                </div>
                <div className="break-words font-medium text-mx-bright">{orchestrator.replace(/^ollama(_chat)?\//, "")}</div>
              </li>
            )}
            {local.map(modality => {
              const target = localModelTargets[modality];
              const configured = byModality[modality];
              const live = configured ? usage.localModels[configured.name] : undefined;
              const { name, note } = target.autoModel
                ? { name: target.autoModel, note: `auto, ${target.installed ? "installed" : "not installed yet"}` }
                : splitDescription(target.description);
              return (
                <li key={modality} className={CARD} title={target.description}>
                  <div className="flex items-center gap-2">
                    {live?.active && <span className="inline-block h-2 w-2 shrink-0 animate-pulse rounded-full bg-mx-green" aria-label="working now" />}
                    <span className="text-[10px] uppercase tracking-wide text-mx-mid">{modality}</span>
                    {live && <span className="ml-auto shrink-0 tabular-nums text-[11px] text-mx-dim">{live.runs} run{live.runs !== 1 ? "s" : ""}</span>}
                  </div>
                  <div className="break-words font-medium text-mx-bright">{name}</div>
                  {note && <div className="text-[11px] text-mx-dim">{note}</div>}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="text-mx-dim italic">{Object.keys(localModelTargets).length ? "None: the task types use paid models." : "No fitting local model found for this machine."}</div>
        )}
      </div>

      <div className="flex items-baseline justify-between gap-2 border-t border-mx-dim pt-2">
        <span className="text-mx-mid">Tokens generated this session</span>
        <span className="tabular-nums text-mx-green">{usage.localTokensGenerated.toLocaleString()}</span>
      </div>
    </div>
  );
}
