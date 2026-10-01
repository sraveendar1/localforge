import { useState } from "react";
import type { BudgetRow } from "./state";

// This month's paid image spending and the monthly limit. A subscription doesn't
// cover API use, so the limit is what keeps a paid image model from running past
// what the user meant to spend (localforge refuses, before anything is billed).
export function BudgetLine({ row, send }: { row: BudgetRow; send: (obj: object) => void }) {
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
