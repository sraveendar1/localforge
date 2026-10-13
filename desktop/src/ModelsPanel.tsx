import { splitDescription } from "./describe";
import { Fragment, useEffect, useState } from "react";
import type { DecisionState, LocalModelTarget, PullState, Suggestion, UpgradeState } from "./state";

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
  upgrade,
  decision,
  localModels,
  onDecision,
  pulls,
  onUpgrade,
}: {
  frontier: string;
  targets: { [modality: string]: LocalModelTarget };
  connected: boolean;
  source: "project" | "defaults" | null;
  onChange: (which: "frontier" | string | null) => void;  // opens the model screen in the centre, at that model
  onManageAccounts: () => void;
  upgrade?: UpgradeState;
  decision?: DecisionState;
  localModels?: string[];
  onDecision?: (action: "set" | "download", model: string) => void;
  pulls?: { [model: string]: PullState };
  onUpgrade?: (action: "check" | "apply" | "research" | "adopt", extra?: { [key: string]: unknown }) => void;
}) {
  // Disabled between a click and its answer; the answer arrives as a new plan/result.
  const [busy, setBusy] = useState<"" | "check" | "apply">("");
  useEffect(() => setBusy(""), [upgrade?.plan, upgrade?.result]);
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
            <Fragment key={modality}>
            <div className={CARD} title={t.description}>
              <div className="flex items-center justify-between gap-2">
                <span className={LABEL}>{label}</span>
                {modality !== "video" && changeButton(modality, `Change the ${label.toLowerCase()} model`)}
              </div>
              <div className={"break-words " + (off ? "italic text-mx-dim" : "font-medium text-mx-bright")}>{off ? name.replace(/\s+--.*$/, "") : name}</div>
              {!off && note && <div className="text-[11px] text-mx-dim">{note.replace(/via your CLI login/, "via your login")}</div>}
              {off && modality === "image" && <div className="text-[11px] text-mx-dim">pick a paid image model with Change</div>}
            </div>
            {modality === "general" && onDecision && decision && (
              <DecisionRow decision={decision} localModels={localModels ?? []} pulls={pulls} connected={connected} onDecision={onDecision} />
            )}
            </Fragment>
          );
        })
      )}

      {onUpgrade && (
        <UpgradeCard
          upgrade={upgrade}
          pulls={pulls}
          busy={busy}
          connected={connected}
          onCheck={() => { setBusy("check"); onUpgrade("check"); }}
          onApply={() => { setBusy("apply"); onUpgrade("apply"); }}
          onResearch={() => onUpgrade("research")}
          onAdopt={(s, deleteOld) => onUpgrade("adopt", { model: s.name, modality: s.modality, replace: s.replaces, delete_old: deleteOld })}
        />
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

// "Is there a better open-weight model for this machine?" Compares what is installed with the catalog's
// balanced pick for this hardware (it doesn't search the internet), shows what would be downloaded and
// removed, and does it on a click. Only models LocalForge itself installed are ever removed.
function UpgradeCard({
  upgrade, pulls, busy, connected, onCheck, onApply, onResearch, onAdopt,
}: {
  upgrade?: UpgradeState;
  pulls?: { [model: string]: PullState };
  busy: "" | "check" | "apply";
  connected: boolean;
  onCheck: () => void;
  onApply: () => void;
  onResearch: () => void;
  onAdopt: (s: Suggestion, deleteOld: boolean) => void;
}) {
  const research = upgrade?.research;
  const plan = upgrade?.plan ?? null;
  const result = upgrade?.result ?? null;
  const downloading = Object.entries(pulls ?? {}).filter(([, p]) => p.ctx === "upgrade" && !p.done);
  const pct = (p: PullState) => (p.total && p.completed != null ? Math.round((p.completed / p.total) * 100) : null);
  const canApply = !!plan?.ok && !plan.empty && !plan.blocked;
  return (
    <div className={CARD} data-testid="upgrade-card">
      <div className="flex items-center justify-between gap-2">
        <span className={LABEL}>Local model upgrades</span>
        <button
          type="button"
          disabled={!connected || busy !== ""}
          onClick={onCheck}
          data-testid="upgrade-check"
          className="shrink-0 rounded-sm border border-mx-dim px-1.5 py-0 text-[10px] text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:opacity-50"
        >
          {busy === "check" ? "Checking…" : "Check for better"}
        </button>
      </div>
      {!plan && !result && downloading.length === 0 && (
        <div className="text-[11px] text-mx-dim">Checks whether a better open-weight model fits this machine.</div>
      )}
      {plan && !plan.ok && <div className="text-[11px] text-mx-amber" data-testid="upgrade-problem">{plan.message}</div>}
      {plan?.ok && plan.empty && !plan.blocked && <div className="text-[11px] text-mx-green">Your local models are already the best fit for this machine.</div>}
      {plan?.ok && plan.blocked && <div className="text-[11px] text-mx-amber">A better model fits, but the upgrade {plan.blocked}.</div>}
      {plan?.ok && !plan.empty && (
        <ul className="mt-1 space-y-0.5 text-[11px] text-mx-mid" data-testid="upgrade-plan">
          {plan.upgrades.map(u => (
            <li key={u.new}>{u.modalities.join(", ")}: {u.old} → <span className="text-mx-bright">{u.new}</span> (~{u.disk_gb} GB)</li>
          ))}
          {plan.remove.map(n => <li key={n} className="text-mx-dim">then remove {n}</li>)}
          {plan.freed_gb > 0 && <li className="text-mx-dim">frees ~{plan.freed_gb} GB afterwards</li>}
        </ul>
      )}
      {plan?.ok && plan.unused_own.length > 0 && (
        <div className="mt-1 text-[11px] text-mx-dim">Not installed by LocalForge, so left alone: {plan.unused_own.join(", ")}</div>
      )}
      {downloading.map(([name, p]) => (
        <div key={name} className="mt-1 text-[11px] text-mx-mid">Downloading {name}{pct(p) !== null ? ` ${pct(p)}%` : "…"}</div>
      ))}
      {canApply && (
        <button
          type="button"
          disabled={!connected || busy !== ""}
          onClick={onApply}
          data-testid="upgrade-apply"
          className="mt-1.5 w-full rounded-sm border border-mx-mid px-2 py-1 text-center text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:opacity-50"
        >
          {busy === "apply" ? "Upgrading…" : "Upgrade now"}
        </button>
      )}
      {result && <div className={"mt-1 text-[11px] " + (result.ok ? "text-mx-green" : "text-mx-red")} data-testid="upgrade-result">{result.message}</div>}

      <div className="mt-2 border-t border-mx-dim pt-1.5" data-testid="research">
        <button
          type="button"
          disabled={!connected || !!research?.running}
          onClick={onResearch}
          data-testid="research-start"
          title="Your orchestrator searches the web for newer open-weight models; LocalForge then checks every name against Ollama's registry and your machine's memory and disk. It uses the orchestrator for a few steps."
          className="w-full rounded-sm border border-mx-dim px-2 py-1 text-center text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:opacity-50"
        >
          {research?.running ? "Researching…" : "Research newer models online"}
        </button>
        {research?.running && <div className="mt-1 break-words text-[11px] text-mx-dim" data-testid="research-progress">{research.progress}</div>}
        {research && !research.running && research.ok === false && <div className="mt-1 text-[11px] text-mx-red" data-testid="research-problem">{research.message}</div>}
        {research && !research.running && research.ok && research.suggestions.length === 0 && <div className="mt-1 text-[11px] text-mx-dim">{research.message}</div>}
        {research?.suggestions.map(s => (
          <ResearchRow key={s.name} s={s} outcome={research.adopted[s.name]} pull={pulls?.[s.name]} onAdopt={onAdopt} />
        ))}
        {research && research.rejected.length > 0 && (
          <details className="mt-1 text-[11px] text-mx-dim">
            <summary className="cursor-pointer">Dropped by the checks ({research.rejected.length})</summary>
            <ul className="mt-0.5 space-y-0.5">{research.rejected.map(r => <li key={r.name}>{r.name}: {r.reason}</li>)}</ul>
          </details>
        )}
      </div>
    </div>
  );
}

// One verified suggestion: what it is, why, how big, what it would replace, and a tick to remove that one
// afterwards (on by default for a model LocalForge installed; off, and labelled, for one the user installed).
function ResearchRow({ s, outcome, pull, onAdopt }: { s: Suggestion; outcome?: { ok: boolean; message: string }; pull?: PullState; onAdopt: (s: Suggestion, deleteOld: boolean) => void }) {
  const [deleteOld, setDeleteOld] = useState(s.replaces_managed);
  const downloading = !!pull && !pull.done && pull.ctx === "research";
  return (
    <div className="mt-1.5 rounded-sm border border-mx-dim px-1.5 py-1" data-testid="research-row">
      <div className="break-words text-mx-bright">{s.name} <span className="text-[10px] text-mx-dim">· {s.modality} · ~{s.size_gb} GB</span></div>
      {s.why && <div className="text-[11px] text-mx-mid">{s.why}</div>}
      {s.replaces && (
        <label className="mt-0.5 flex items-start gap-1 text-[11px] text-mx-dim">
          <input type="checkbox" checked={deleteOld} onChange={e => setDeleteOld(e.target.checked)} data-testid="research-delete-old" />
          <span>Remove {s.replaces} afterwards to free space{s.replaces_managed ? "" : " (you installed it yourself)"}</span>
        </label>
      )}
      {!outcome && (
        <button
          type="button"
          disabled={downloading}
          onClick={() => onAdopt(s, deleteOld)}
          data-testid="research-adopt"
          className="mt-1 w-full rounded-sm border border-mx-mid px-2 py-0.5 text-center text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:opacity-50"
        >
          {downloading ? `Downloading ${pull?.total && pull.completed != null ? Math.round((pull.completed / pull.total) * 100) + "%" : "…"}` : `Use for ${s.modality}`}
        </button>
      )}
      {!outcome && pull?.done && !pull.ok && <div className="mt-0.5 text-[11px] text-mx-red">{pull.message}</div>}
      {outcome && <div className={"mt-0.5 text-[11px] " + (outcome.ok ? "text-mx-green" : "text-mx-red")} data-testid="research-outcome">{outcome.message}</div>}
    </div>
  );
}

// The Decision model: the small local model that makes the pass/fail call on delegated work (a Jev-style
// decision model, or any judge). Pick one that is installed, or type a name to download it (the download is
// checked against this machine first, like any other).
export function DecisionRow({ decision, localModels, pulls, connected, onDecision }: {
  decision: DecisionState;
  localModels: string[];
  pulls?: { [model: string]: PullState };
  connected: boolean;
  onDecision: (action: "set" | "download", model: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [typed, setTyped] = useState("");
  const pulling = Object.entries(pulls ?? {}).filter(([, p]) => p.ctx === "decision");
  const auto = decision.model === "auto";
  return (
    <div className={CARD} data-testid="decision-row">
      <div className="flex items-center justify-between gap-2">
        <span className={LABEL}>Decision (System 1)</span>
        <button
          type="button"
          disabled={!connected}
          onClick={() => setEditing(e => !e)}
          data-testid="decision-change"
          className="shrink-0 rounded-sm border border-mx-dim px-1.5 py-0 text-[10px] text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:opacity-50"
        >
          {editing ? "Close" : "Change"}
        </button>
      </div>
      <div className={"break-words " + (auto ? "italic text-mx-dim" : "font-medium text-mx-bright")} data-testid="decision-current">
        {auto ? "automatic" : decision.model}
      </div>
      <div className="text-[11px] text-mx-dim">
        {auto ? "a dedicated judge if installed, else another local model" : decision.installed ? "installed" : "not installed"}
      </div>
      {editing && (
        <div className="mt-1.5 space-y-1.5">
          <select
            value={auto ? "auto" : decision.model}
            onChange={e => onDecision("set", e.target.value)}
            data-testid="decision-select"
            className="w-full rounded-sm border border-mx-dim bg-mx-panel px-1 py-0.5 text-[11px] text-mx-bright"
          >
            <option value="auto">automatic</option>
            {localModels.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
          <div className="flex gap-1">
            <input
              value={typed}
              onChange={e => setTyped(e.target.value)}
              placeholder="a model to download, e.g. atla/selene-mini"
              data-testid="decision-typed"
              className="min-w-0 flex-1 rounded-sm border border-mx-dim bg-mx-panel px-1 py-0.5 text-[11px] text-mx-bright placeholder:text-mx-dim"
            />
            <button
              type="button"
              disabled={!typed.trim()}
              onClick={() => { onDecision("download", typed.trim()); setTyped(""); }}
              data-testid="decision-download"
              className="shrink-0 rounded-sm border border-mx-mid px-1.5 py-0.5 text-[11px] text-mx-green hover:border-mx-bright disabled:opacity-50"
            >
              Download &amp; use
            </button>
          </div>
        </div>
      )}
      {pulling.map(([name, p]) => (
        <div key={name} className={"mt-1 text-[11px] " + (p.done ? (p.ok ? "text-mx-green" : "text-mx-red") : "text-mx-mid")} data-testid="decision-pull">
          {p.done ? p.message : `Downloading ${name}${p.total && p.completed != null ? " " + Math.round((p.completed / p.total) * 100) + "%" : "…"}`}
        </div>
      ))}
      {decision.result && <div className={"mt-1 text-[11px] " + (decision.result.ok ? "text-mx-green" : "text-mx-red")} data-testid="decision-result">{decision.result.message}</div>}
    </div>
  );
}
