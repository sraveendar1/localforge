import { splitDescription } from "./describe";
import type { LocalModelTarget } from "./state";

const MODALITIES: [string, string][] = [
  ["coding", "Coding"],
  ["docs", "Writing & docs"],
  ["general", "General"],
  ["image", "Images"],
  ["video", "Video"],
];

const LABEL = "text-[10px] uppercase tracking-wide text-mx-mid";
const CARD = "rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1.5";

// A compact, read-only summary of the models in use. Choosing and changing them
// (and every refusal or download that goes with it) happens in the centre of the
// window, in the same guided screen a new project starts with: a narrow side panel
// is no place to read why a model can't be used. Each model is a small card: what it's
// for, the model's name on its own line, then how it's reached.
export function ModelsPanel({
  frontier,
  targets,
  connected,
  source,
  onChange,
  onManageAccounts,
}: {
  frontier: string;
  targets: { [modality: string]: LocalModelTarget };
  connected: boolean;
  source: "project" | "defaults" | null;
  onChange: (which: "frontier" | string | null) => void;  // opens the model screen in the centre, at that model
  onManageAccounts: () => void;
}) {
  const rows = MODALITIES.filter(([m]) => targets[m]);
  const changeButton = (which: string, label: string) => (
    <button
      type="button"
      disabled={!connected}
      aria-label={label}
      onClick={() => onChange(which)}
      className="shrink-0 rounded-sm border border-mx-dim px-1.5 py-0 text-[10px] text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:opacity-50"
    >
      Change
    </button>
  );
  return (
    <div className="space-y-1.5" data-testid="models-panel">
      <div className={CARD}>
        <div className="flex items-center justify-between gap-2">
          <span className={LABEL}>Frontier</span>
          {changeButton("frontier", "Change the frontier model")}
        </div>
        <div className="break-words font-medium text-mx-bright">{frontier || "not set"}</div>
      </div>

      {rows.length === 0 ? (
        <div className="text-mx-dim italic">No fitting local model found for this machine.</div>
      ) : (
        rows.map(([modality, label]) => {
          const t = targets[modality];
          const off = modality === "video" || (modality === "image" && t.target === "auto");
          const { name, note } = t.autoModel
            ? { name: t.autoModel, note: `auto, ${t.installed ? "installed" : "not installed yet"}` }
            : splitDescription(t.description);
          return (
            <div key={modality} className={CARD} title={t.description}>
              <div className="flex items-center justify-between gap-2">
                <span className={LABEL}>{label}</span>
                {modality !== "video" && changeButton(modality, `Change the ${label.toLowerCase()} model`)}
              </div>
              <div className={"break-words " + (off ? "italic text-mx-dim" : "font-medium text-mx-bright")}>{off ? name.replace(/\s+--.*$/, "") : name}</div>
              {!off && note && <div className="text-[11px] text-mx-dim">{note.replace(/via your CLI login/, "via your login")}</div>}
              {off && modality === "image" && <div className="text-[11px] text-mx-dim">pick a paid image model with Change</div>}
            </div>
          );
        })
      )}

      {source && (
        <p className="px-0.5 text-[11px] text-mx-dim" data-testid="models-source">
          {source === "project" ? "Saved for this project (.localforge/models.json)." : "Using your defaults; not saved for this project yet."}
        </p>
      )}
      <div className="grid grid-cols-1 gap-1.5 pt-0.5">
        <button
          type="button"
          disabled={!connected}
          onClick={() => onChange(null)}
          data-testid="change-models"
          className="rounded-sm border border-mx-mid px-2 py-1.5 text-center text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:opacity-50"
        >
          Change models…
        </button>
        <button
          type="button"
          disabled={!connected}
          onClick={onManageAccounts}
          data-testid="manage-accounts"
          className="rounded-sm border border-mx-dim px-2 py-1 text-center text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:opacity-50"
        >
          Accounts &amp; keys…
          <span className="block text-[10px] text-mx-dim">Add an API key or sign in</span>
        </button>
      </div>
    </div>
  );
}
