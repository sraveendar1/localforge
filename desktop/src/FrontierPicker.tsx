import { useState } from "react";
import type { OrchestratorOption } from "./state";

// The choices for the orchestrator (the frontier model that plans and
// reviews). Same list `/model` shows in the terminal. No "Auto": the
// orchestrator is always an explicit choice.
export function FrontierPicker({
  current,
  options,
  onPick,
  onClose,
  onSetup,
}: {
  current: string;
  options: OrchestratorOption[];
  onPick: (model: string) => void;
  onClose: () => void;
  onSetup: () => void;
}) {
  const [other, setOther] = useState("");
  const known = options;  // only what can actually run here (never a list of models with no key behind them)
  const groups: { [group: string]: OrchestratorOption[] } = {};
  for (const o of known) (groups[o.group] ??= []).push(o);

  function pick(model: string) {
    const m = model.trim();
    if (!m) return;
    if (m !== current) onPick(m);
    onClose();
  }

  return (
    <div className="mt-1 rounded-sm border border-mx-dim bg-mx-panel2 p-2">
      {options.length === 0 && (
        <p className="px-1.5 pb-1 text-mx-dim">
          Nothing to choose from yet: no API key, login or downloaded local model.{" "}
          <button type="button" className="text-mx-green underline hover:text-mx-bright" onClick={() => { onSetup(); onClose(); }}>Set up…</button>
        </p>
      )}
      {Object.entries(groups).map(([group, items], i) => (
        <div key={group} className={i > 0 ? "mt-1 border-t border-mx-dim pt-1" : ""}>
          <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">{group}</div>
          {items.map(o => (
            <div key={o.id}>
              <button
                title={o.problem ?? o.label}
                disabled={!!o.problem}
                className={"block w-full truncate rounded-sm px-1.5 py-0.5 text-left " + (o.problem ? "cursor-not-allowed text-mx-dim line-through" : "hover:bg-mx-dim/50 " + (o.id === current ? "text-mx-green" : "text-mx-mid"))}
                onClick={() => pick(o.id)}
              >
                {o.id}
              </button>
              {o.problem && <p className="px-1.5 pb-1 text-[10px] text-mx-red">Can't run here: {o.problem}.</p>}
            </div>
          ))}
        </div>
      ))}
      <div className="mt-1 border-t border-mx-dim pt-1">
        <input
          type="text"
          value={other}
          placeholder="Other model id…"
          aria-label="Other model id"
          onChange={e => setOther(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") pick(other); }}
          className="block w-full rounded-sm border border-mx-dim bg-mx-panel px-1.5 py-0.5 text-mx-mid"
        />
      </div>
      <button className="mt-1 block w-full rounded-sm px-1.5 py-0.5 text-left text-mx-red hover:bg-mx-dim/50" onClick={onClose}>
        Cancel
      </button>
    </div>
  );
}
