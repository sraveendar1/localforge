import type { LocalModelTarget } from "./state";

const MODALITIES: [string, string][] = [
  ["coding", "Coding"],
  ["docs", "Writing & docs"],
  ["general", "General"],
  ["image", "Images"],
  ["video", "Video"],
];

// A compact, read-only summary of the models in use. Choosing and changing them
// (and every refusal or download that goes with it) happens in the centre of the
// window, in the same guided screen a new project starts with: a narrow side panel
// is no place to read why a model can't be used.
export function ModelsPanel({
  frontier,
  targets,
  connected,
  source,
  onChangeModels,
  onManageAccounts,
}: {
  frontier: string;
  targets: { [modality: string]: LocalModelTarget };
  connected: boolean;
  source: "project" | "defaults" | null;
  onChangeModels: () => void;
  onManageAccounts: () => void;
}) {
  const rows = MODALITIES.filter(([m]) => targets[m]);
  return (
    <div className="space-y-2" data-testid="models-panel">
      <div>
        <div className="text-mx-mid">Frontier</div>
        <div className="break-words text-mx-bright">{frontier || "not set"}</div>
      </div>
      {rows.length === 0 ? (
        <div className="text-mx-dim italic">No fitting local model found for this machine.</div>
      ) : (
        rows.map(([modality, label]) => (
          <div key={modality}>
            <div className="text-mx-mid">{label}</div>
            <div className={"break-words " + (modality === "video" || (modality === "image" && targets[modality].target === "auto") ? "text-mx-dim italic" : "text-mx-bright")}>
              {targets[modality].description}
            </div>
          </div>
        ))
      )}
      {source && (
        <p className="text-[11px] text-mx-dim" data-testid="models-source">
          {source === "project" ? "Saved for this project (.localforge/models.json)." : "Using your defaults; not saved for this project yet."}
        </p>
      )}
      <button
        type="button"
        disabled={!connected}
        onClick={onChangeModels}
        data-testid="change-models"
        className="w-full rounded-sm border border-mx-mid px-2 py-1 text-left text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:opacity-50"
      >
        Change models…
        <span className="block text-[10px] text-mx-dim">Opens in the centre of the window</span>
      </button>
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
