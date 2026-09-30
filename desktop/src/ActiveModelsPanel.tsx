import type { PaidModel } from "./paidModels";
import { ConfiguredModel, LocalModelTarget, Usage } from "./state";

const MODALITY_ORDER = ["coding", "docs", "general"];

// "Active LLMs at play", split by what each costs: the paid models in use
// (the orchestrator, and any task type -- image always -- sent to a paid model,
// each with its role) and the free local ones (with a pulsing dot on the one
// generating, and this session's run counts). Live status only -- choosing the
// models lives in the left panel's "Models" section (ModelsPanel), so this one
// never has settings mixed into it.
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
    <>
      <div>
        <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">Paid models</div>
        {configuredPaid.length === 0 ? (
          <div className="text-mx-dim italic">None in use.</div>
        ) : (
          <ul className="space-y-1" data-testid="active-paid">
            {configuredPaid.map(m => (
              <li key={m.name}>
                <div className="break-words"><span className="text-mx-amber">{m.roles.join(" · ")}</span> <span className="text-mx-bright">{m.name}</span></div>
                {m.how && <div className="text-mx-dim">{m.how}</div>}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div>
        <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">Local models (free)</div>
        {local.length > 0 || localOrchestrator ? (
          <ul className="space-y-1">
            {localOrchestrator && (
              <li className="flex items-center gap-2">
                <span className="text-mx-mid">orchestrator</span>
                <span className="min-w-0 break-words text-mx-bright">{orchestrator.replace(/^ollama(_chat)?\//, "")}</span>
              </li>
            )}
            {local.map(modality => {
              const target = localModelTargets[modality];
              const configured = byModality[modality];
              const live = configured ? usage.localModels[configured.name] : undefined;
              return (
                <li key={modality} className="flex items-center gap-2">
                  {live?.active && <span className="inline-block h-2 w-2 rounded-full bg-mx-green animate-pulse"></span>}
                  <span className="text-mx-mid">{modality}</span>
                  <span className="min-w-0 break-words text-mx-bright" title={target.description}>{target.description}</span>
                  {live && <span className="text-mx-dim tabular-nums ml-auto shrink-0">{live.runs} run{live.runs !== 1 ? "s" : ""}</span>}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="text-mx-dim italic">{Object.keys(localModelTargets).length ? "None: the task types use paid models." : "No fitting local model found for this machine."}</div>
        )}
      </div>
      <div className="flex justify-between gap-2 pt-1">
        <span className="text-mx-mid">Tokens generated this session</span>
        <span className="text-mx-green tabular-nums">{usage.localTokensGenerated.toLocaleString()}</span>
      </div>
    </>
  );
}
