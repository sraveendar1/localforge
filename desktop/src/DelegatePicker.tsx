import type { DelegateOptions } from "./state";

// The choices for one task type's model: Auto, a local model (free), or a
// paid cloud model through a key/login you already have. Sends nothing
// itself -- the caller turns a pick into set_delegate_target.
export function DelegatePicker({
  modality,
  options,
  onPick,
  onClose,
}: {
  modality?: string;
  options: DelegateOptions | undefined;
  onPick: (target: string) => void;
  onClose: () => void;
}) {
  if (!options) {
    return <div className="mt-1 px-2 py-1 text-mx-dim italic">Loading options…</div>;
  }
  // Image generation has no local model: it is off, or a paid cloud image model.
  const isImage = modality === "image";
  return (
    <div className="mt-1 rounded-sm border border-mx-dim bg-mx-panel2 p-2">
      <button
        className={"block w-full rounded-sm px-1.5 py-0.5 text-left hover:bg-mx-dim/50 " + (options.current === "auto" ? "text-mx-green" : "text-mx-mid")}
        onClick={() => { onPick("auto"); onClose(); }}
      >
        {isImage ? "Off (no image generation)" : "Auto (best-fitting installed local model)"}
      </button>
      {isImage && options.cloud.length === 0 && (
        <p className="mt-1 border-t border-mx-dim px-1.5 pt-1 text-mx-dim">
          No image provider is set up. Image generation needs an OpenAI or Gemini API key: run <span className="font-mono">localforge setup</span> in a terminal.
        </p>
      )}
      {options.local.length > 0 && (
        <div className="mt-1 border-t border-mx-dim pt-1">
          <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">Local (free)</div>
          {options.local.map(m => (
            <div key={m.name}>
              <button
                disabled={!!m.problem}
                title={m.problem ?? undefined}
                className={"block w-full rounded-sm px-1.5 py-0.5 text-left " + (m.problem ? "cursor-not-allowed text-mx-dim line-through" : "hover:bg-mx-dim/50 " + (options.current === `ollama:${m.name}` ? "text-mx-green" : "text-mx-mid"))}
                onClick={() => { onPick(m.name); onClose(); }}
              >
                {m.name} {m.problem ? "" : m.installed ? "" : `(will download, ~${m.disk_gb}GB)`}
              </button>
              {m.problem && <p className="px-1.5 pb-1 text-[10px] text-mx-red">Can't run here: {m.problem}.</p>}
            </div>
          ))}
        </div>
      )}
      {options.cloud.length > 0 && (
        <div className="mt-1 border-t border-mx-dim pt-1">
          <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">{isImage ? "Image models (costs money per image)" : "Cloud (costs money)"}</div>
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
