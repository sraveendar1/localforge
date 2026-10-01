import { useEffect, useRef, useState } from "react";
import { BudgetLine } from "./BudgetLine";
import { Link, ProviderCard } from "./SetupScreen";
import type { BudgetRow, ChatState, DelegateOptions, PullState } from "./state";

const TASKS: [string, string, string][] = [
  ["coding", "Coding", "Writes and edits source code"],
  ["docs", "Writing & docs", "READMEs, comments, documentation"],
  ["general", "General", "Everything else that's text"],
  ["image", "Images", "Paid image models only (no local one yet)"],
  ["video", "Video", "Not available yet"],
];

const STEP_TITLES = ["The model that plans", "Who does the work", "Review"];

function gb(n: number | null | undefined): string {
  return typeof n === "number" ? `${n >= 10 ? n.toFixed(0) : n.toFixed(1)} GB` : "";
}

// One download's progress or its outcome. A refusal (the model isn't recommended for
// this machine) reads as a refusal, in red, and nothing was downloaded.
function PullStatus({ pull }: { pull: PullState | undefined }) {
  if (!pull) return null;
  if (pull.done) {
    return (
      <p role={pull.ok ? "status" : "alert"} className={"mt-1 text-xs " + (pull.ok ? "text-mx-green" : "text-mx-red")} data-testid="pull-result">
        {pull.message}
      </p>
    );
  }
  const pct = pull.total ? Math.min(100, Math.round(((pull.completed ?? 0) * 100) / pull.total)) : null;
  return (
    <div className="mt-1" data-testid="pull-progress">
      <div className="h-1.5 w-full overflow-hidden rounded-sm bg-mx-dim/50">
        <div className="h-full bg-mx-green transition-all" style={{ width: `${pct ?? 3}%` }} />
      </div>
      <div className="mt-0.5 text-[11px] text-mx-dim">
        {pull.status === "starting" || !pull.status ? "Starting the download…" : pull.status}
        {pct !== null && ` · ${pct}%`}
        {pull.total ? ` · ${gb((pull.completed ?? 0) / 1e9)} of ${gb(pull.total / 1e9)}` : ""}
      </div>
    </div>
  );
}

const isPulling = (p: PullState | undefined) => !!p && !p.done;
// A download's state, but only where it was asked for: a refusal under the orchestrator list
// mustn't turn up under a task row for the same model.
const pullOf = (pulls: ChatState["pulls"], name: string, ctx: string): PullState | undefined => (pulls[name]?.ctx === ctx ? pulls[name] : undefined);

// A text box for a model name, with a button that downloads it (if needed) and uses it.
// The tag the backend downloads under (it adds :latest, and ignores an ollama/ prefix).
const normalizeName = (raw: string) => {
  const n = raw.trim().replace(/^ollama(_chat)?\//, "");
  return n.includes(":") ? n : `${n}:latest`;
};

function TypedModel({
  placeholder,
  label,
  buttonLabel,
  disabled,
  onSubmit,
  pulls,
  ctx = "",
}: {
  placeholder: string;
  label: string;
  buttonLabel: string;
  disabled: boolean;
  onSubmit: (name: string) => void;
  pulls?: ChatState["pulls"];  // when the name is a download, its progress and outcome show underneath
  ctx?: string;
}) {
  const [name, setName] = useState("");
  const [submitted, setSubmitted] = useState("");
  return (
    <div>
      <form
        className="flex items-center gap-2"
        onSubmit={e => { e.preventDefault(); if (name.trim() && !disabled) { setSubmitted(name.trim()); onSubmit(name.trim()); } }}
      >
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder={placeholder}
          aria-label={label}
          spellCheck={false}
          autoComplete="off"
          className="min-w-0 flex-1 rounded-sm border border-mx-dim bg-mx-panel px-2 py-1 text-xs text-mx-mid"
        />
        <button
          type="submit"
          disabled={disabled || !name.trim()}
          className="shrink-0 rounded-sm border border-mx-mid px-3 py-1 text-xs text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim"
        >
          {buttonLabel}
        </button>
      </form>
      {pulls && submitted && <PullStatus pull={pullOf(pulls, normalizeName(submitted), ctx)} />}
    </div>
  );
}

// The models a provider offers, directly under its card: pick the one that plans. Appears once the
// provider can be used, and scrolls into view when it first does so it's never missed.
function ProviderModels({
  provider,
  label,
  options,
  current,
  disabled,
  placeholder,
  onChoose,
  onTyped,
}: {
  provider: string;
  label: string;
  options: ChatState["orchestratorOptions"];
  current: string;
  disabled: boolean;
  placeholder: string;
  onChoose: (id: string) => void;
  onTyped: (id: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, []);
  const vias = [...new Set(options.map(o => o.via))];
  return (
    <div ref={ref} className="ml-3 mt-1 rounded-sm border-l-2 border-mx-mid bg-mx-panel2 p-3" data-testid={`models-${provider}`}>
      <div className="font-semibold text-mx-bright">Pick the model that plans</div>
      {vias.map(via => (
        <div key={via} className="mt-1">
          {vias.length > 1 && <div className="text-[10px] uppercase tracking-wide text-mx-dim">{via === "login" ? "With your login" : "With your API key"}</div>}
          {options.filter(o => o.via === via).map(o => (
            <button
              key={o.id}
              type="button"
              disabled={disabled}
              aria-pressed={o.id === current}
              onClick={() => onChoose(o.id)}
              className={"flex w-full items-center gap-2 rounded-sm px-2 py-1 text-left text-xs hover:bg-mx-dim/40 " + (o.id === current ? "text-mx-green" : "text-mx-mid")}
            >
              <span aria-hidden>{o.id === current ? "●" : "○"}</span>
              <span className="min-w-0 flex-1 truncate">{o.id}</span>
            </button>
          ))}
        </div>
      ))}
      <div className="mt-2">
        <TypedModel placeholder={`Other ${label.split(" ")[0]} model id, e.g. ${placeholder}`} label={`Other ${label.split(" ")[0]} model id`} buttonLabel="Use" disabled={disabled} onSubmit={onTyped} />
      </div>
    </div>
  );
}

type Props = {
  chat: ChatState;
  send: (obj: object) => void;
  mode: "new" | "edit";
  onClose: () => void;   // the ✕ / Close: leave without saving anything more
  onFinish: () => void;  // Start / Save: write the project's models file and carry on
};

// The guided model setup, in the centre of the window: shown for a project that has no
// saved models yet (right after "Trust this folder"), and again whenever "Change models…"
// is clicked. Everything about choosing a model -- including why one is refused and
// downloads with their progress -- reads here, not in a side panel.
export function ModelWizard({ chat, send, mode, onClose, onFinish }: Props) {
  const setup = chat.setup;
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [wantAdvanced, setWantAdvanced] = useState(false);
  const [tried, setTried] = useState<string | null>(null);  // an installed model that was picked but can't run here: say why, right under it
  const [open, setOpen] = useState<string | null>(null);  // the task row whose choices are showing
  const result = chat.setupResult;
  useEffect(() => { setBusy(null); }, [result?.n]);

  const connected = chat.connected;
  const anyPull = Object.values(chat.pulls).some(isPulling);
  const disabled = !connected || busy !== null;
  const targets = chat.localModelTargets;
  const pinned = TASKS.filter(([m]) => targets[m] && targets[m].target !== "auto");
  const advanced = wantAdvanced || pinned.length > 0;

  // Fresh data whenever the wizard opens (a model pulled or key added since).
  useEffect(() => {
    send({ type: "setup_status_request" });
    send({ type: "orchestrator_options_request" });
    send({ type: "advanced_model_request" });
    send({ type: "budget_request" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!setup) return <div className="p-6 text-sm text-mx-dim">Looking at this computer…</div>;

  const needsOrchestrator = setup.needsSetup;
  const ollama = setup.ollama;

  const pull = (model: string, use_as: object | undefined, ctx: string) => send({ type: "pull_model", model, use_as, ctx });

  function chooseMode(next: "auto" | "advanced") {
    setOpen(null);
    if (next === "advanced") return setWantAdvanced(true);
    setWantAdvanced(false);
    for (const [m] of pinned) send({ type: "set_delegate_target", modality: m, target: "auto", inline: true });
  }

  function toggleRow(modality: string) {
    if (open === modality) return setOpen(null);
    setOpen(modality);
    send({ type: "delegate_options_request", modality });  // asked each time, so a new download or key is in the list
  }

  const stepper = (
    <ol className="flex items-center gap-2 text-xs" aria-label="Steps">
      {STEP_TITLES.map((t, i) => (
        <li key={t} className={"flex items-center gap-1 " + (i === step ? "text-mx-bright" : i < step ? "text-mx-green" : "text-mx-dim")} aria-current={i === step ? "step" : undefined}>
          <span className={"flex h-5 w-5 items-center justify-center rounded-full border text-[10px] " + (i === step ? "border-mx-bright" : "border-mx-dim")}>{i < step ? "✓" : i + 1}</span>
          <span className="hidden sm:inline">{t}</span>
          {i < STEP_TITLES.length - 1 && <span className="mx-1 text-mx-dim">—</span>}
        </li>
      ))}
    </ol>
  );

  // ---- step 1: the orchestrator -------------------------------------------------
  // A provider's models appear right under its own card once it can be used (a key is saved or
  // its login chosen), so choosing one is the next thing you see, not a list further down.
  const optionsFor = (provider: string) => chat.orchestratorOptions.filter(o => o.provider === provider);
  const localOptions = optionsFor("local");
  const modelPlaceholder: { [provider: string]: string } = { anthropic: "claude-sonnet-5-5", openai: "gpt-5", gemini: "gemini/gemini-2.5-pro" };
  const planning = !needsOrchestrator;

  const stepOrchestrator = (
    <div className="space-y-4">
      <p className="text-sm text-mx-mid">
        One model plans your work and checks it. It can be a paid one (sign in or paste a key, then pick its model right under it) or an
        open-weight model that runs on this computer for free. The models that write the code are chosen in the next step.
      </p>

      <div className="space-y-3" data-testid="wizard-providers">
        {setup.providers.map(p => {
          const opts = optionsFor(p.id);
          return (
            <div key={p.id}>
              <ProviderCard
                provider={p}
                busy={busy}
                disabled={disabled}
                onKey={key => { setBusy(`key:${p.id}`); send({ type: "save_api_key", provider: p.id, key }); }}
                onLogin={() => { setBusy(`login:${p.id}`); send({ type: "setup_use_login", provider: p.id }); }}
              />
              {opts.length > 0 && (
                <ProviderModels
                  provider={p.id}
                  label={p.label}
                  options={opts}
                  current={planning ? chat.model : ""}
                  disabled={disabled}
                  placeholder={modelPlaceholder[p.id] ?? "model id"}
                  onChoose={id => { setBusy(`orch:${id}`); send({ type: "setup_choose_orchestrator", model: id }); }}
                  onTyped={id => { setBusy("orch:typed"); send({ type: "setup_choose_orchestrator", model: id, typed: true }); }}
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="rounded-sm border border-mx-dim bg-mx-panel2 p-3" data-testid="wizard-local">
        <div className="font-semibold text-mx-bright">Open-weight models (run on this computer, free)</div>
        {!ollama.installed ? (
          <p className="mt-1 text-xs text-mx-dim">
            They run through <Link url="https://ollama.com/download">Ollama</Link>, which isn't installed. Install it, open it once, then come back.
          </p>
        ) : !ollama.running ? (
          <p className="mt-1 text-xs text-mx-dim">Ollama is installed but not running. Open the Ollama app (or run <span className="font-mono">ollama serve</span>), then come back.</p>
        ) : (
          <>
            <p className="mt-1 text-xs text-mx-dim">
              Small models are often unreliable at planning; a paid model is the safer choice. A model that isn't recommended for this
              computer won't be downloaded; you'll be told why if you pick one.
            </p>
            {localOptions.length > 0 && (
              <div className="mt-2" data-testid="wizard-local-installed">
                <div className="mb-0.5 text-[10px] uppercase tracking-wide text-mx-dim">Already on this computer</div>
                {localOptions.map(o => (
                  <div key={o.id}>
                    <button
                      type="button"
                      disabled={disabled}
                      aria-pressed={o.id === chat.model}
                      onClick={() => {
                        if (o.problem) return setTried(o.id);  // say why, don't pretend it works
                        setTried(null);
                        setBusy(`orch:${o.id}`);
                        send({ type: "setup_choose_orchestrator", model: o.id });
                      }}
                      className={"flex w-full items-center gap-2 rounded-sm px-2 py-1 text-left text-xs hover:bg-mx-dim/40 " + (o.id === chat.model && planning ? "text-mx-green" : "text-mx-mid")}
                    >
                      <span aria-hidden>{o.id === chat.model && planning ? "●" : "○"}</span>
                      <span className="min-w-0 flex-1 truncate">{o.id.replace(/^ollama(_chat)?\//, "")}</span>
                    </button>
                    {tried === o.id && o.problem && <p role="alert" className="px-2 pb-1 text-[11px] text-mx-red">Can't use it here: {o.problem}.</p>}
                  </div>
                ))}
              </div>
            )}
            {ollama.suggested.length > 0 && (
              <div className="mt-3">
                <div className="mb-1 text-[10px] uppercase tracking-wide text-mx-dim">Download one</div>
                <ul className="space-y-2 text-xs">
                  {ollama.suggested.map(m => {
                    const p = pullOf(chat.pulls, m.name, "orchestrator");
                    return (
                      <li key={m.name} data-testid={`suggested-${m.name}`}>
                        <div className="flex items-center gap-2">
                          <span className="min-w-0 flex-1 truncate text-mx-mid">
                            {m.name} <span className="text-mx-dim">· {gb(m.diskGb)}</span>
                          </span>
                          <button
                            type="button"
                            disabled={disabled || isPulling(p)}
                            onClick={() => pull(m.name, { orchestrator: true }, "orchestrator")}
                            className="shrink-0 rounded-sm border border-mx-mid px-2 py-0.5 text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim"
                          >
                            {isPulling(p) ? "Downloading…" : "Download & use"}
                          </button>
                        </div>
                        <PullStatus pull={p} />
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
            <div className="mt-3">
              <TypedModel
                placeholder="Or type a model name, e.g. llama3.1:8b"
                label="Open-weight model name"
                buttonLabel="Download & use"
                disabled={disabled}
                onSubmit={name => pull(name, { orchestrator: true }, "orchestrator")}
                pulls={chat.pulls}
                ctx="orchestrator"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );

  // ---- step 2: the writers --------------------------------------------------------
  const rowDescription = (modality: string): React.ReactNode => {
    const t = targets[modality];
    if (!t) return null;
    return <span className={modality === "video" || (modality === "image" && t.target === "auto") ? "text-mx-dim italic" : "text-mx-bright"}>{t.description}</span>;
  };

  const stepWriters = (
    <div className="space-y-4">
      <p className="text-sm text-mx-mid">
        Each kind of work goes to its own model. Let localforge pick the best open-weight model that fits this computer, or choose them yourself.
      </p>
      <div className="flex rounded-sm border border-mx-dim text-xs" role="group" aria-label="Model mode">
        <button type="button" aria-pressed={!advanced} onClick={() => chooseMode("auto")} data-testid="mode-auto"
          className={"flex-1 px-3 py-2 text-left " + (!advanced ? "bg-mx-dim/60 text-mx-bright" : "text-mx-mid hover:text-mx-bright")}>
          <span className="block font-semibold">Automatic</span>
          <span className="text-mx-dim">localforge picks, and downloads what's missing</span>
        </button>
        <button type="button" aria-pressed={advanced} onClick={() => chooseMode("advanced")} data-testid="mode-advanced"
          className={"flex-1 border-l border-mx-dim px-3 py-2 text-left " + (advanced ? "bg-mx-dim/60 text-mx-bright" : "text-mx-mid hover:text-mx-bright")}>
          <span className="block font-semibold">Advanced</span>
          <span className="text-mx-dim">choose any model for each kind of work</span>
        </button>
      </div>

      <div className="space-y-2">
        {TASKS.filter(([m]) => targets[m]).map(([modality, label, blurb]) => {
          const t = targets[modality];
          const autoPull = t.autoModel ? pullOf(chat.pulls, t.autoModel, "auto") : undefined;
          return (
            <div key={modality} className="rounded-sm border border-mx-dim bg-mx-panel2 p-3" data-testid={`task-${modality}`}>
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="font-semibold text-mx-bright">{label}</div>
                  <div className="text-[11px] text-mx-dim">{blurb}</div>
                </div>
                {advanced && modality !== "video" && (
                  <button type="button" disabled={!connected} aria-expanded={open === modality} onClick={() => toggleRow(modality)}
                    className="shrink-0 rounded-sm border border-mx-dim px-2 py-0.5 text-xs text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:opacity-50">
                    {open === modality ? "Done" : "Change"}
                  </button>
                )}
              </div>
              <div className="mt-1 break-words text-xs">{rowDescription(modality)}</div>

              {!advanced && t.autoModel && t.installed === false && (
                <div className="mt-2">
                  <button type="button" disabled={disabled || isPulling(autoPull)} onClick={() => pull(t.autoModel!, undefined, "auto")}
                    className="rounded-sm border border-mx-mid px-2 py-0.5 text-xs text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim">
                    {isPulling(autoPull) ? "Downloading…" : `Download now (${gb(t.diskGb)})`}
                  </button>
                  <span className="ml-2 text-[11px] text-mx-dim">or it asks the first time it's needed</span>
                  <PullStatus pull={autoPull} />
                </div>
              )}
              {modality === "image" && !advanced && <p className="mt-1 text-[11px] text-mx-dim">Choose Advanced to turn on paid image generation.</p>}
              {modality === "image" && t.target.startsWith("api:") && (() => {
                const row: BudgetRow | undefined = chat.budgets.find(b => b.provider === t.target.split(":")[1]);
                return row ? <BudgetLine row={row} send={send} /> : null;
              })()}

              {advanced && open === modality && (
                <TaskChoices
                  modality={modality}
                  options={chat.delegateOptions[modality]}
                  pulls={chat.pulls}
                  disabled={disabled}
                  onPick={target => { send({ type: "set_delegate_target", modality, target, inline: true }); }}
                  onPull={(name) => pull(name, { modality }, modality)}
                  onNeedKey={() => setStep(0)}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );

  // ---- step 3: review ----------------------------------------------------------------
  const stepReview = (
    <div className="space-y-4">
      <p className="text-sm text-mx-mid">This is what {mode === "edit" ? "this project uses" : "will be saved for this project"}. You can change it any time from Models in the left panel.</p>
      <dl className="space-y-2 rounded-sm border border-mx-dim bg-mx-panel2 p-3 text-xs" data-testid="wizard-review">
        <div>
          <dt className="text-mx-mid">Plans and checks the work</dt>
          <dd className="text-mx-bright">{needsOrchestrator ? "nothing chosen yet" : chat.model}</dd>
        </div>
        {TASKS.filter(([m]) => targets[m]).map(([m, label]) => (
          <div key={m}>
            <dt className="text-mx-mid">{label}</dt>
            <dd className="break-words">{rowDescription(m)}</dd>
          </div>
        ))}
      </dl>
      {anyPull && <p className="text-xs text-mx-amber">A download is still running. You can start now; it carries on in the background.</p>}
    </div>
  );

  const steps = [stepOrchestrator, stepWriters, stepReview];

  return (
    <div className="mx-auto flex h-full max-w-2xl flex-col gap-4 overflow-y-auto px-6 pb-0 pt-0" data-testid="model-wizard">
      <div className="sticky top-0 z-10 flex items-start justify-between gap-3 bg-mx-bg pb-2 pt-6">
        <div>
          <h1 className="text-lg font-semibold uppercase tracking-widest text-mx-bright glow">{mode === "edit" ? "Change models" : "Choose your models"}</h1>
          <div className="mt-2">{stepper}</div>
        </div>
        <button type="button" onClick={onClose} aria-label="Close" title="Close" data-testid="wizard-x"
          className="shrink-0 rounded-sm border border-mx-dim px-2 text-mx-mid hover:border-mx-mid hover:text-mx-bright">✕</button>
      </div>

      {steps[step]}

      {result && step < 2 && (
        <p role={result.ok ? "status" : "alert"} data-testid="setup-result"
          className={"whitespace-pre-wrap rounded-sm border px-3 py-2 text-sm " + (result.ok ? "border-mx-green text-mx-green" : "border-mx-red text-mx-red")}>
          {result.message}
        </p>
      )}
      {step === 0 && needsOrchestrator && setup.orchestrator.reason && (
        <p className="text-xs text-mx-amber" data-testid="setup-reason">Right now: {setup.orchestrator.reason}.</p>
      )}

      <div className="sticky bottom-0 z-10 mt-auto flex items-center justify-between gap-2 border-t border-mx-dim bg-mx-bg py-3">
        <button type="button" onClick={onClose} className="rounded-sm border border-mx-dim px-3 py-1 text-xs text-mx-mid hover:border-mx-mid hover:text-mx-bright">
          {mode === "edit" ? "Close" : "Skip for now"}
        </button>
        <div className="flex gap-2">
          {step > 0 && (
            <button type="button" onClick={() => { setOpen(null); setStep(step - 1); }} className="rounded-sm border border-mx-dim px-3 py-1 text-xs text-mx-mid hover:border-mx-mid hover:text-mx-bright">Back</button>
          )}
          {step < 2 ? (
            <button type="button" data-testid="wizard-next" disabled={step === 0 && needsOrchestrator}
              title={step === 0 && needsOrchestrator ? "Choose the model that plans first" : undefined}
              onClick={() => { setOpen(null); setStep(step + 1); if (step === 0) send({ type: "advanced_model_request" }); }}
              className="rounded-sm border border-mx-mid px-4 py-1 text-xs text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim">
              Next
            </button>
          ) : (
            <button type="button" data-testid="wizard-finish" onClick={onFinish}
              className="rounded-sm border border-mx-mid px-4 py-1 text-xs text-mx-green hover:border-mx-bright hover:text-mx-bright">
              {mode === "edit" ? "Save" : "Start"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// The choices for one task type: Auto, a local model (installed ones are used as they
// are, the rest downloaded first if they fit this computer), a paid model, or a name
// typed in.
function TaskChoices({
  modality,
  options,
  pulls,
  disabled,
  onPick,
  onPull,
  onNeedKey,
}: {
  modality: string;
  options: DelegateOptions | undefined;
  pulls: ChatState["pulls"];
  disabled: boolean;
  onPick: (target: string) => void;
  onPull: (name: string) => void;
  onNeedKey: () => void;
}) {
  const [tried, setTried] = useState<string | null>(null);  // an installed model that was picked but can't run here
  if (!options) return <div className="mt-2 text-xs italic text-mx-dim">Loading options…</div>;
  const isImage = modality === "image";
  const row = "block w-full rounded-sm px-2 py-1 text-left text-xs ";
  return (
    <div className="mt-2 space-y-2 border-t border-mx-dim pt-2" data-testid={`choices-${modality}`}>
      <button type="button" disabled={disabled} onClick={() => onPick("auto")}
        className={row + "hover:bg-mx-dim/40 " + (options.current === "auto" ? "text-mx-green" : "text-mx-mid")}>
        {isImage ? "Off (no image generation)" : "Automatic (best-fitting local model)"}
      </button>

      {isImage && options.cloud.length === 0 && (
        <p className="px-2 text-xs text-mx-dim">
          Image generation needs an OpenAI or Gemini API key.{" "}
          <button type="button" className="text-mx-green underline hover:text-mx-bright" onClick={onNeedKey}>Add one in step 1</button>.
        </p>
      )}

      {options.local.length > 0 && (
        <div>
          <div className="mb-0.5 px-2 text-[10px] uppercase tracking-wide text-mx-dim">Open-weight, on this computer (free)</div>
          {options.local.map(m => {
            const p = pullOf(pulls, m.name, modality);
            const current = options.current === `ollama:${m.name}`;
            return (
              <div key={m.name}>
                <div className="flex items-center gap-2">
                  <button type="button" disabled={disabled || !m.installed}
                    onClick={() => { if (m.problem) return setTried(m.name); setTried(null); onPick(m.name); }}
                    className={row + "min-w-0 flex-1 truncate " + (m.installed ? "hover:bg-mx-dim/40 " + (current ? "text-mx-green" : "text-mx-mid") : "cursor-default text-mx-dim")}>
                    {current ? "● " : ""}{m.name} <span className="text-mx-dim">· {m.installed ? "installed" : `${gb(m.disk_gb)} to download`}</span>
                  </button>
                  {!m.installed && (
                    <button type="button" disabled={disabled || isPulling(p)} onClick={() => onPull(m.name)}
                      className="shrink-0 rounded-sm border border-mx-mid px-2 py-0.5 text-xs text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim">
                      {isPulling(p) ? "Downloading…" : "Download & use"}
                    </button>
                  )}
                </div>
                {tried === m.name && m.problem && <p role="alert" className="px-2 pb-1 text-[11px] text-mx-red">Can't use it here: {m.problem}.</p>}
                <PullStatus pull={p} />
              </div>
            );
          })}
        </div>
      )}

      {options.cloud.length > 0 && (
        <div>
          <div className="mb-0.5 px-2 text-[10px] uppercase tracking-wide text-mx-dim">{isImage ? "Image models (billed per image to your API key)" : "Paid (costs money)"}</div>
          {options.cloud.map(c => {
            const value = `${c.kind}:${c.provider}:${c.model}`;
            return (
              <button key={value} type="button" disabled={disabled} onClick={() => onPick(value)}
                className={row + "break-words hover:bg-mx-dim/40 " + (options.current === value ? "text-mx-amber" : "text-mx-mid")}>
                {c.provider} · {c.model.replace(/^gemini\//, "")} <span className="text-mx-dim">· {c.kind === "api" ? "API key" : "login"}</span>
                {typeof c.priceUsd === "number" && <span className="text-mx-dim"> · ~${c.priceUsd.toFixed(3)}/image</span>}
              </button>
            );
          })}
        </div>
      )}

      {!isImage && (
        <div className="px-2">
          <TypedModel
            placeholder="Type a model name, e.g. qwen2.5-coder:7b"
            label={`${modality} model name`}
            buttonLabel="Use"
            disabled={disabled}
            onSubmit={name => onPull(name)}
            pulls={pulls}
            ctx={modality}
          />
        </div>
      )}
    </div>
  );
}
