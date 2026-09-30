import { useState } from "react";
import type { OrchestratorOption } from "./state";

// Shown until the backend has said what is actually usable on this machine
// (or when it can't: no Ollama, no keys). The real list -- every model already
// in Ollama plus each provider with an API key or CLI login -- arrives as an
// `orchestrator_options` event and replaces this.
const FALLBACK_OPTIONS: OrchestratorOption[] = [
  { id: "claude-opus-5", label: "claude-opus-5", group: "Anthropic" },
  { id: "claude-sonnet-5", label: "claude-sonnet-5", group: "Anthropic" },
  { id: "claude-haiku-4-5-20251001", label: "claude-haiku-4-5-20251001", group: "Anthropic" },
  { id: "claude-fable-5-1", label: "claude-fable-5-1", group: "Anthropic" },
  { id: "gpt-5", label: "gpt-5", group: "OpenAI" },
  { id: "gemini/gemini-2.5-pro", label: "gemini/gemini-2.5-pro", group: "Gemini" },
];

// The choices for the orchestrator (the frontier model that plans and
// reviews). Same list `/model` shows in the terminal. No "Auto": the
// orchestrator is always an explicit choice.
export function FrontierPicker({
  current,
  options,
  onPick,
  onClose,
}: {
  current: string;
  options: OrchestratorOption[];
  onPick: (model: string) => void;
  onClose: () => void;
}) {
  const [other, setOther] = useState("");
  const known = options.length > 0 ? options : FALLBACK_OPTIONS;
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
