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

const OTHER = "__other__";

export function ModelPicker({
  value,
  options,
  onChange,
  onOpen,
  disabled = false,
}: {
  value: string;
  options: OrchestratorOption[];
  onChange: (model: string) => void;
  onOpen?: () => void;  // asked to refresh the list, e.g. after pulling a model
  disabled?: boolean;
}) {
  const known = options.length > 0 ? options : FALLBACK_OPTIONS;
  const listed = known.some(o => o.id === value);
  const [isOther, setIsOther] = useState(false);
  const [otherModel, setOtherModel] = useState("");

  // Group in first-seen order, and keep whatever is selected now visible even
  // when it isn't in the list (a typed id, or a provider without a key).
  const groups: { [group: string]: OrchestratorOption[] } = {};
  for (const o of known) (groups[o.group] ??= []).push(o);
  const showCurrent = !!value && !listed;

  return (
    <div className="flex items-center gap-2">
      <select
        value={isOther ? OTHER : value || ""}
        onChange={e => {
          const selected = e.target.value;
          if (selected === OTHER) { setIsOther(true); return; }
          setIsOther(false);
          onChange(selected);
        }}
        onFocus={onOpen}
        onMouseDown={onOpen}
        disabled={disabled}
        aria-label="Orchestrator model"
        className={`block w-full max-w-xs rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1 text-sm ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
      >
        {showCurrent && (
          <optgroup label="Current">
            <option value={value}>{value}</option>
          </optgroup>
        )}
        {Object.entries(groups).map(([group, items]) => (
          <optgroup key={group} label={group}>
            {items.map(o => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </optgroup>
        ))}
        <option value={OTHER}>Other…</option>
      </select>
      {isOther && (
        <input
          type="text"
          value={otherModel}
          placeholder="model id"
          onChange={e => setOtherModel(e.target.value)}
          onBlur={() => { const m = otherModel.trim(); if (m) onChange(m); }}
          onKeyDown={e => { if (e.key === "Enter") { const m = otherModel.trim(); if (m) { onChange(m); setIsOther(false); } } }}
          disabled={disabled}
          className={`block w-full rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1 text-sm ${disabled ? "cursor-not-allowed opacity-50" : ""}`}
        />
      )}
    </div>
  );
}
