import type { ReactNode } from "react";
import type { SetupCheck, SetupChecks } from "./state";

// Small pieces of the first-run experience, shared by the model wizard and the getting-started
// card. Ideas borrowed from Paperclip's onboarding (one question per screen, a labelled progress
// strip, provider tiles that carry how you'd connect, one state-aware primary button, a plain
// environment check), drawn in localforge's own Matrix style.

// ---- a labelled, segmented progress strip --------------------------------------------------
export function StepStrip({ titles, step }: { titles: readonly string[]; step: number }) {
  return (
    <div data-testid="step-strip">
      <ol className="flex gap-1.5" aria-label="Steps">
        {titles.map((t, i) => (
          <li key={t} aria-current={i === step ? "step" : undefined} className="min-w-0 flex-1">
            <div className={"h-1 rounded-full " + (i < step ? "bg-mx-green" : i === step ? "bg-mx-bright glow" : "bg-mx-dim/60")} />
            <div className={"mt-1 truncate text-[11px] " + (i === step ? "text-mx-bright" : i < step ? "text-mx-green" : "text-mx-dim")}>
              <span aria-hidden>{i < step ? "✓ " : `${i + 1}. `}</span>{t}
            </div>
          </li>
        ))}
      </ol>
      <p className="sr-only">Step {step + 1} of {titles.length}</p>
    </div>
  );
}

// ---- provider tiles --------------------------------------------------------------------------
export type Tile = { id: string; mark: string; label: string; tag: string; status: "ready" | "active" | "off"; statusText: string };

// A row of tiles, one per place the planning model can come from. Choosing one shows only that
// provider's details below, instead of every provider's form stacked on one long page.
export function ProviderTiles({ tiles, picked, onPick }: { tiles: Tile[]; picked: string; onPick: (id: string) => void }) {
  return (
    <div role="radiogroup" aria-label="Where the planning model comes from" className="grid grid-cols-2 gap-2 sm:grid-cols-4" data-testid="provider-tiles">
      {tiles.map(t => {
        const on = t.id === picked;
        return (
          <button
            key={t.id}
            type="button"
            role="radio"
            aria-checked={on}
            data-testid={`tile-${t.id}`}
            onClick={() => onPick(t.id)}
            className={
              "flex flex-col items-center gap-1 rounded-sm border px-2 py-3 text-center transition-colors " +
              (on ? "border-mx-bright bg-mx-dim/30" : "border-mx-dim bg-mx-panel2 hover:border-mx-mid")
            }
          >
            <span aria-hidden className={"flex h-8 w-8 items-center justify-center rounded-sm border font-mono text-sm " + (on ? "border-mx-bright text-mx-bright" : "border-mx-mid text-mx-green")}>{t.mark}</span>
            <span className={"text-xs font-semibold " + (on ? "text-mx-bright" : "text-mx-green")}>{t.label}</span>
            <span className="text-[10px] text-mx-dim">{t.tag}</span>
            <span
              data-testid={`tile-status-${t.id}`}
              className={"text-[10px] " + (t.status === "active" ? "text-mx-bright" : t.status === "ready" ? "text-mx-green" : "text-mx-dim")}
            >
              {t.status === "active" ? "● " : t.status === "ready" ? "✓ " : "○ "}{t.statusText}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ---- the setup check -------------------------------------------------------------------------
const ICON: Record<SetupCheck["state"], [string, string]> = { ok: ["✓", "text-mx-green"], warn: ["!", "text-mx-amber"], fail: ["✕", "text-mx-red"] };

export function ChecksList({ checks, onFix, loading }: { checks: SetupChecks | null; onFix?: (fix: "planner" | "writers") => void; loading?: boolean }) {
  if (!checks) return <p className="text-xs text-mx-dim italic" data-testid="checks-loading">{loading === false ? "" : "Checking this computer…"}</p>;
  return (
    <ul className="space-y-1.5" data-testid="checks-list">
      {checks.checks.map(c => {
        const [icon, color] = ICON[c.state];
        return (
          <li key={c.id} data-state={c.state} data-testid={`check-${c.id}`} className="flex items-start gap-2 text-xs">
            <span aria-hidden className={"mt-px w-3 shrink-0 text-center font-semibold " + color}>{icon}</span>
            <span className="sr-only">{c.state === "ok" ? "Passed:" : c.state === "warn" ? "Heads up:" : "Needs attention:"}</span>
            <span className="min-w-0 flex-1">
              <span className="text-mx-bright">{c.label}</span>
              {c.detail && <span className="block break-words text-mx-dim">{c.detail}</span>}
            </span>
            {c.fix && onFix && c.state !== "ok" && (
              <button type="button" onClick={() => onFix(c.fix!)} className="shrink-0 rounded-sm border border-mx-dim px-1.5 py-0.5 text-[11px] text-mx-mid hover:border-mx-mid hover:text-mx-bright">
                {c.fix === "planner" ? "Fix" : "Choose"}
              </button>
            )}
          </li>
        );
      })}
    </ul>
  );
}

// ---- a primary action that says what it will do -----------------------------------------------
export function Primary({ children, disabled, onClick, testId, busy }: { children: ReactNode; disabled?: boolean; onClick: () => void; testId?: string; busy?: boolean }) {
  return (
    <button
      type="button"
      data-testid={testId}
      disabled={disabled}
      onClick={onClick}
      className="rounded-sm border border-mx-mid bg-mx-dim/20 px-4 py-1.5 text-xs font-semibold text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:bg-transparent disabled:font-normal disabled:text-mx-dim"
    >
      {busy ? "Working…" : children}
    </button>
  );
}
