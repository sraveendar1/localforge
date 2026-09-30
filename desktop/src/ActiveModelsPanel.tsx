import { ConfiguredModel, LocalModelTarget, Usage } from "./state";

const MODALITY_ORDER = ["coding", "docs", "general"];

// "Active LLMs at play": which model handles each kind of work right now, and
// which open-weighted one is actually generating (the pulsing dot) plus this
// session's run/token counts per model. Live status only -- choosing the
// models (frontier and per kind of work) lives in the left panel's "Models"
// section (ModelsPanel), so this one never has settings mixed into it.
export function ActiveModelsPanel({
  usage,
  configuredModels,
  localModelTargets,
}: {
  usage: Usage;
  configuredModels: ConfiguredModel[];
  localModelTargets: { [modality: string]: LocalModelTarget };
}) {
  const byModality = Object.fromEntries(configuredModels.map(m => [m.modality, m]));
  const modalities = MODALITY_ORDER.filter(m => localModelTargets[m]);

  return (
    <>
      {modalities.length > 0 ? (
        <ul className="space-y-1">
          {modalities.map(modality => {
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
        <div className="text-mx-dim italic">No fitting local model found for this machine.</div>
      )}
      <div className="flex justify-between gap-2 pt-1">
        <span className="text-mx-mid">Tokens generated this session</span>
        <span className="text-mx-green tabular-nums">{usage.localTokensGenerated.toLocaleString()}</span>
      </div>
    </>
  );
}
