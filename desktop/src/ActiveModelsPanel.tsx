import { useState } from "react";
import { ConfiguredModel, DelegateOptions, LocalModelTarget, Usage } from "./state";

const MODALITY_ORDER = ["coding", "docs", "general"];

function DelegatePicker({
  options,
  onPick,
  onClose,
}: {
  options: DelegateOptions | undefined;
  onPick: (target: string) => void;
  onClose: () => void;
}) {
  if (!options) {
    return <div className="mt-1 px-2 py-1 text-mx-dim italic">Loading options…</div>;
  }
  return (
    <div className="mt-1 rounded-sm border border-mx-dim bg-mx-panel2 p-2">
      <button
        className={"block w-full rounded-sm px-1.5 py-0.5 text-left hover:bg-mx-dim/50 " + (options.current === "auto" ? "text-mx-green" : "text-mx-mid")}
        onClick={() => { onPick("auto"); onClose(); }}
      >
        Auto (best-fitting installed local model)
      </button>
      {options.local.length > 0 && (
        <div className="mt-1 border-t border-mx-dim pt-1">
          <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">Local (free)</div>
          {options.local.map(m => (
            <button
              key={m.name}
              className={"block w-full rounded-sm px-1.5 py-0.5 text-left hover:bg-mx-dim/50 " + (options.current === `ollama:${m.name}` ? "text-mx-green" : "text-mx-mid")}
              onClick={() => { onPick(m.name); onClose(); }}
            >
              {m.name} {m.installed ? "" : `(will download, ~${m.disk_gb}GB)`}
            </button>
          ))}
        </div>
      )}
      {options.cloud.length > 0 && (
        <div className="mt-1 border-t border-mx-dim pt-1">
          <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">Cloud (costs money)</div>
          {options.cloud.map(c => {
            const value = `${c.kind}:${c.provider}:${c.model}`;
            return (
              <button
                key={value}
                className={"block w-full rounded-sm px-1.5 py-0.5 text-left hover:bg-mx-dim/50 " + (options.current === value ? "text-mx-amber" : "text-mx-mid")}
                onClick={() => { onPick(value); onClose(); }}
              >
                {c.model} ({c.provider}, via {c.kind === "api" ? "API key" : "CLI login"})
              </button>
            );
          })}
        </div>
      )}
      <button className="mt-1 block w-full rounded-sm px-1.5 py-0.5 text-left text-mx-red hover:bg-mx-dim/50" onClick={onClose}>
        Cancel
      </button>
    </div>
  );
}

// "Active LLMs at play": which open-weighted model is configured for each
// modality on this machine/project, and which one is actually generating
// right now (the pulsing dot) plus this session's run/token counts per
// model. A "Change" action per row opens a picker (local catalog models, or
// a paid cloud model via an API key/CLI login you already have) that sends
// set_delegate_target -- the same override /local-model sets from the chat
// input or the terminal REPL, kept in sync since both write the same place.
export function ActiveModelsPanel({
  usage,
  configuredModels,
  localModelTargets,
  delegateOptions,
  send,
}: {
  usage: Usage;
  configuredModels: ConfiguredModel[];
  localModelTargets: { [modality: string]: LocalModelTarget };
  delegateOptions: { [modality: string]: DelegateOptions };
  send: (obj: object) => void;
}) {
  const [openModality, setOpenModality] = useState<string | null>(null);
  const byModality = Object.fromEntries(configuredModels.map(m => [m.modality, m]));
  const modalities = MODALITY_ORDER.filter(m => localModelTargets[m]);

  function toggle(modality: string) {
    if (openModality === modality) {
      setOpenModality(null);
      return;
    }
    setOpenModality(modality);
    if (!delegateOptions[modality]) send({ type: "delegate_options_request", modality });
  }

  return (
    <>
      {modalities.length > 0 ? (
        <ul className="space-y-1">
          {modalities.map(modality => {
            const target = localModelTargets[modality];
            const configured = byModality[modality];
            const live = configured ? usage.localModels[configured.name] : undefined;
            return (
              <li key={modality}>
                <div className="flex items-center gap-2">
                  {live?.active && <span className="inline-block h-2 w-2 rounded-full bg-mx-green animate-pulse"></span>}
                  <span className="text-mx-mid">{modality}</span>
                  <span className="truncate text-mx-bright" title={target.description}>{target.description}</span>
                  {live && <span className="text-mx-dim tabular-nums ml-auto shrink-0">{live.runs} run{live.runs !== 1 ? "s" : ""}</span>}
                  <button
                    className="shrink-0 rounded-sm border border-mx-dim px-1.5 py-0 text-[10px] text-mx-dim hover:border-mx-mid hover:text-mx-bright"
                    onClick={() => toggle(modality)}
                  >
                    Change
                  </button>
                </div>
                {openModality === modality && (
                  <DelegatePicker
                    options={delegateOptions[modality]}
                    onPick={target => send({ type: "set_delegate_target", modality, target })}
                    onClose={() => setOpenModality(null)}
                  />
                )}
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
