import { ConfiguredModel, Usage } from "./state";

// "Active LLMs at play": which open-weighted model is configured for each
// modality on this machine/project, and which one is actually generating
// right now (the pulsing dot) plus this session's run/token counts per
// model. Split out of UsagePanel (which now covers frontier tokens/cost
// only) so it can sit directly under the plan, per the requested order:
// Overall goal -> Pending task -> Active LLMs -> Usage and cost.
export function ActiveModelsPanel({ usage, configuredModels }: { usage: Usage; configuredModels: ConfiguredModel[] }) {
  return (
    <>
      {configuredModels.length > 0 ? (
        <ul className="space-y-0.5">
          {configuredModels.map(m => {
            const live = usage.localModels[m.name];
            return (
              <li key={m.modality} className="flex items-center gap-2">
                {live?.active && <span className="inline-block h-2 w-2 rounded-full bg-mx-green animate-pulse"></span>}
                <span className="text-mx-mid">{m.modality}</span>
                <span className="text-mx-bright">{m.name}</span>
                {live && <span className="text-mx-dim tabular-nums ml-auto">{live.runs} run{live.runs !== 1 ? "s" : ""}</span>}
              </li>
            );
          })}
        </ul>
      ) : (
        <div className="text-mx-dim italic">No fitting local model found for this machine.</div>
      )}
      <div className="flex justify-between gap-2 pt-1">
        <span className="text-mx-mid">Tokens generated this session</span>
        <span className="text-mx-green tabular-nums">{usage.localTokensGenerated.toLocaleString()}</span>
      </div>
    </>
  );
}
