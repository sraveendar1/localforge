import { useState } from "react";
import { DelegatePicker } from "./DelegatePicker";
import { FrontierPicker } from "./FrontierPicker";
import type { BudgetRow, DelegateOptions, LocalModelTarget, OrchestratorOption } from "./state";

// Video has a row so its choice has somewhere to live, but there's no video
// generation yet: it can't be changed.
// This month's paid image spending and the monthly limit. A subscription doesn't
// cover API use, so the limit is what keeps a paid image model from running past
// what the user meant to spend (localforge refuses, before anything is billed).
function BudgetLine({ row, send }: { row: BudgetRow; send: (obj: object) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  const save = () => {
    const usd = Number(value);
    if (Number.isFinite(usd) && usd > 0) { send({ type: "set_budget", provider: row.provider, usd }); setEditing(false); }
  };
  return (
    <div className="mt-0.5 text-[11px] text-mx-dim" data-testid="budget-line">
      <span>
        {row.provider} this month: <span className="text-mx-mid">${row.spent.toFixed(2)}</span>
        {row.limit ? <> of <span className="text-mx-mid">${row.limit.toFixed(2)}</span></> : " · no limit"}
        {row.images > 0 && ` · ${row.images} image${row.images === 1 ? "" : "s"}`}
      </span>{" "}
      {!editing ? (
        <>
          <button type="button" className="text-mx-green underline hover:text-mx-bright" onClick={() => { setValue(row.limit ? String(row.limit) : ""); setEditing(true); }}>
            {row.limit ? "change limit" : "set a monthly limit"}
          </button>
          {row.limit && (
            <> · <button type="button" className="text-mx-mid underline hover:text-mx-bright" onClick={() => send({ type: "set_budget", provider: row.provider, usd: null })}>remove</button></>
          )}
        </>
      ) : (
        <form className="mt-1 flex items-center gap-1" onSubmit={e => { e.preventDefault(); save(); }}>
          <span>$</span>
          <input type="number" min="0" step="1" value={value} onChange={e => setValue(e.target.value)} aria-label="Monthly limit in dollars" autoFocus
            className="w-16 rounded-sm border border-mx-dim bg-mx-panel px-1 py-0.5 text-mx-mid" />
          <button type="submit" className="rounded-sm border border-mx-mid px-2 py-0.5 text-mx-green hover:text-mx-bright">Save</button>
          <button type="button" className="text-mx-dim underline" onClick={() => setEditing(false)}>cancel</button>
        </form>
      )}
    </div>
  );
}

const MODALITIES = ["coding", "docs", "general", "image", "video"];
const NOT_CHANGEABLE = new Set(["video"]);

// Every model choice in one place: the frontier model that plans and
// reviews, and the model that writes each kind of work (coding / docs /
// general). It used to be split between a header dropdown (frontier) and a
// "Change" button per row in the right-hand panel (the rest), so the whole
// setup was never visible, or changeable, together.
//
//   Automatic  localforge picks the best local model for each task type (rows
//              are read-only, but always name the real model). The frontier is
//              still yours to choose: it has no "auto".
//   Advanced   every row can be changed: a local model, or a paid cloud model
//              through a key/login you already have. Image is off unless a paid
//              image model is chosen; Video isn't available yet.
//
// Advanced is simply "at least one task type is pinned, or you asked for it":
// there is no separate saved mode to fall out of sync with `/advanced-model`
// typed in the terminal or the chat. Switching back to Automatic clears the pins.
export function ModelsPanel({
  frontier,
  frontierOptions,
  targets,
  delegateOptions,
  connected,
  source,
  budgets,
  onSetup,
  onManageAccounts,
  send,
}: {
  frontier: string;
  frontierOptions: OrchestratorOption[];
  targets: { [modality: string]: LocalModelTarget };
  delegateOptions: { [modality: string]: DelegateOptions };
  connected: boolean;
  source: "project" | "defaults" | null;
  budgets: BudgetRow[];
  onSetup: () => void;
  onManageAccounts: () => void;
  send: (obj: object) => void;
}) {
  const [wantAdvanced, setWantAdvanced] = useState(false);
  const [open, setOpen] = useState<string | null>(null);  // "frontier" | a modality | null
  const rows = MODALITIES.filter(m => targets[m]);
  const pinned = rows.filter(m => targets[m].target !== "auto");
  const advanced = wantAdvanced || pinned.length > 0;

  function toggle(which: string) {
    if (open === which) return setOpen(null);
    setOpen(which);
    // Asked each time it opens, so a model pulled or a key added since is there.
    if (which === "frontier") send({ type: "orchestrator_options_request" });
    else send({ type: "delegate_options_request", modality: which });
  }

  function goAutomatic() {
    setWantAdvanced(false);
    setOpen(null);
    for (const m of pinned) send({ type: "set_delegate_target", modality: m, target: "auto" });
  }

  const changeButton = (which: string, label: string) => (
    <button
      type="button"
      disabled={!connected}
      aria-label={label}
      aria-expanded={open === which}
      className="shrink-0 rounded-sm border border-mx-dim px-1.5 py-0 text-[10px] text-mx-dim hover:border-mx-mid hover:text-mx-bright disabled:opacity-50"
      onClick={() => toggle(which)}
    >
      Change
    </button>
  );

  return (
    <div className="space-y-2" data-testid="models-panel">
      <div className="flex rounded-sm border border-mx-dim text-[11px]" role="group" aria-label="Model mode">
        <button
          type="button"
          aria-pressed={!advanced}
          onClick={goAutomatic}
          className={"flex-1 px-2 py-0.5 " + (!advanced ? "bg-mx-dim/60 text-mx-bright" : "text-mx-mid hover:text-mx-bright")}
        >
          Automatic
        </button>
        <button
          type="button"
          aria-pressed={advanced}
          onClick={() => setWantAdvanced(true)}
          className={"flex-1 border-l border-mx-dim px-2 py-0.5 " + (advanced ? "bg-mx-dim/60 text-mx-bright" : "text-mx-mid hover:text-mx-bright")}
        >
          Advanced
        </button>
      </div>
      <p className="text-mx-dim">
        {advanced
          ? "Choose the model for each kind of work. Paid models cost money per use. Image generation is off until you pick a paid image model."
          : "localforge picks the best local model for each kind of work. Switch to Advanced to choose."}
      </p>

      {source && (
        <p className="text-mx-dim" data-testid="models-source">
          {source === "project"
            ? "Saved for this project (.localforge/models.json). Changes update it."
            : "Using your defaults. The first change is saved for this project."}
        </p>
      )}

      <div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-mx-mid">Frontier</span>
          {changeButton("frontier", "Change frontier model")}
        </div>
        <div className="break-words text-mx-bright">{frontier || "not set"}</div>
        {open === "frontier" && (
          <FrontierPicker
            current={frontier}
            options={frontierOptions}
            onPick={model => send({ type: "set_model", model })}
            onClose={() => setOpen(null)}
            onSetup={onSetup}
          />
        )}
      </div>

      {rows.length === 0 ? (
        <div className="text-mx-dim italic">No fitting local model found for this machine.</div>
      ) : (
        rows.map(modality => (
          <div key={modality}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-mx-mid">{modality}</span>
              {advanced && !NOT_CHANGEABLE.has(modality) && changeButton(modality, `Change ${modality} model`)}
            </div>
            <div className={"break-words " + (NOT_CHANGEABLE.has(modality) || (modality === "image" && targets[modality].target === "auto") ? "text-mx-dim italic" : "text-mx-bright")}>{targets[modality].description}</div>
            {modality === "image" && targets.image?.target.startsWith("api:") && (() => {
              const row = budgets.find(b => b.provider === targets.image.target.split(":")[1]);
              return row ? <BudgetLine row={row} send={send} /> : null;
            })()}
            {advanced && open === modality && (
              <DelegatePicker
                modality={modality}
                options={delegateOptions[modality]}
                onPick={target => send({ type: "set_delegate_target", modality, target })}
                onClose={() => setOpen(null)}
              />
            )}
          </div>
        ))
      )}
      <button
        type="button"
        disabled={!connected}
        onClick={onManageAccounts}
        data-testid="manage-accounts"
        className="w-full rounded-sm border border-mx-dim px-2 py-1 text-left text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:opacity-50"
      >
        Accounts &amp; keys…
        <span className="block text-[10px] text-mx-dim">Add an API key or sign in (Claude, GPT, Gemini)</span>
      </button>
    </div>
  );
}
