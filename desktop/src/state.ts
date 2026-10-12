export type ToolItem = { kind: "tool"; name: string; summary: string; result?: string };
// `runtime` is how the model is reached: "ollama" (on this computer) or "api"/"cli" (a paid model).
export type DelegateItem = { kind: "delegate"; modality: string; model: string; runtime: string; provider: string; output: string; done: boolean; tokens?: number; seconds?: number };
export type NoteItem = { kind: "note"; text: string };
export type Item = ToolItem | DelegateItem | NoteItem;
// What a task did and what went wrong, sent after every task (task_summary.py).
export type TaskSummary = {
  outcome: "completed" | "stopped" | "failed";
  error: string | null;
  durationS: number;
  steps: number;
  files: { action: string; path: string; by: string }[];
  delegations: { tool: string; model: string; path: string; detail: string }[];
  commands: { command: string; ok: boolean; detail: string }[];
  reads: number;
  web: number;
  declined: { what: string; tool: string }[];
  problems: { what: string; error: string; recovered: boolean; plain?: string }[];
  plainError?: string | null;
};
export type Message = { role: "user" | "assistant" | "error" | "system" | "summary"; text: string; items: Item[]; round?: number; imageDataUrl?: string; summary?: TaskSummary; detail?: string; plan?: boolean };
export type Approval = { id: string; kind: string; title: string; detail: string };
export type Todo = { content: string; status: "pending" | "in_progress" | "completed" };
export type Gpu = { name: string; vram_gb: number; backend: string };
export type SystemStats = {
  hardware: {
    os: string;
    arch: string;
    cpu_cores: number;
    ram_gb: number;
    free_disk_gb: number;
    gpus: Gpu[];
    memory_bandwidth_gbps?: number | null;
  };
  cpuPercent: number;
  ramUsedGb: number;
  ramTotalGb: number;
};
export type MemoryFact = { name: string; description?: string; content?: string; type?: string };
export type MemoryState = { facts: MemoryFact[]; narrative: string; goal: string; progress: string[] };
export type ScratchFile = { path: string; size: number };
export type UsageTotals = {
  frontierPromptTokens: number;
  frontierCompletionTokens: number;
  frontierCostUsd: number;
  localTokensGenerated: number;
  subscriptionCostUsd: number;
  delegateTokensGenerated: number;
  delegateCostUsd: number;
  delegateNotionalCostUsd: number;
  tasks: number;
  models: string[];
};
export type ConfiguredModel = { modality: string; name: string; runtime: string; qualityTier: number };
// `autoModel`/`installed`/`diskGb` are only set when the target is "auto" and a local model
// fits: the model "auto" means here, and whether it still has to be downloaded.
export type LocalModelTarget = { target: string; description: string; autoModel?: string; installed?: boolean; diskGb?: number };
export type DelegateCloudOption = { kind: "api" | "cli"; provider: string; model: string; priceUsd?: number | null };
export type DelegateLocalOption = { name: string; quality_tier: number; disk_gb: number; installed: boolean; problem?: string | null };
export type DelegateOptions = { local: DelegateLocalOption[]; cloud: DelegateCloudOption[]; current: string };
// What first-run setup needs to know (no secret in it): whether the orchestrator
// can run, which providers have a key or an installed CLI, and Ollama.
export type SetupProvider = {
  id: string;
  label: string;
  keySet: boolean;
  keyUrl: string | null;
  cli: { command: string; installed: boolean; installHint: string; loginHint: string } | null;
};
export type SetupStatus = {
  needsSetup: boolean;
  orchestrator: { model: string; ready: boolean; reason: string };
  providers: SetupProvider[];
  ollama: {
    installed: boolean;
    running: boolean;
    models: { name: string; problem: string | null }[];
    // Open-weight models not on disk yet: those that fit this machine, then some that don't
    // (with the reason, and they can't be downloaded from the app).
    suggested: { name: string; diskGb: number; qualityTier: number; modality: string; problem: string | null }[];
  };
};
// One line of "Check my setup": ok, warn (works, but something happens on first use) or fail
// (a task would stop). `fix` names the wizard step that resolves it.
export type SetupCheck = { id: string; label: string; state: "ok" | "warn" | "fail"; detail: string; fix: "planner" | "writers" | null };
export type SetupChecks = { checks: SetupCheck[]; ok: boolean };
// A model download in progress, or its outcome. `blocked` = refused before any download
// because the model isn't recommended for this machine.
// `ctx` says where the download was asked for ("orchestrator", a task type, "auto"), so its progress or refusal shows only there.
export type PullState = { ctx: string; status: string; completed: number | null; total: number | null; done: boolean; ok?: boolean; message?: string; blocked?: boolean };
// The guided model setup in the centre of the window: "unknown" until the project's models
// are known, "active" while it is on screen, "done" once finished or closed.
export type WizardState = "unknown" | "active" | "done";
// The answer to a setup action; `n` counts them so a screen can tell a new one.
export type SetupResult = { ok: boolean; message: string; n: number };
// This month's paid image spending per provider, and the user's optional monthly limit.
export type BudgetRow = { provider: string; spent: number; limit: number | null; images: number };
export type OrchestratorOption = { id: string; label: string; group: string; provider: string; via: string; problem?: string | null };
export type ChatState = {
  messages: Message[];
  approvals: Approval[];
  todos: Todo[];
  status: string;
  connected: boolean;
  running: boolean;
  model: string;
  autoApprove: boolean;
  usage: Usage;
  usageHistory: { previousSession: UsageTotals | null; allTime: UsageTotals | null };
  configuredModels: ConfiguredModel[];
  // Every orchestrator usable here (installed Ollama models + cloud providers
  // with a key or CLI login), for the header dropdown. Empty until fetched.
  orchestratorOptions: OrchestratorOption[];
  // The backend process, for a start that goes wrong: what it printed, whether it
  // has exited, and when we began waiting for it. Without these a backend that
  // fails to start looked like an empty, dead app.
  backendLog: string[];
  backendExited: boolean;
  backendStartedAt: number | null;
  trustDeclined: boolean;
  projectBusy: string | null;  // another window already has this folder open (the backend refused to start)
  budgets: BudgetRow[];
  setup: SetupStatus | null;
  setupChecks: SetupChecks | null;
  setupResult: SetupResult | null;
  pulls: { [model: string]: PullState };
  wizard: WizardState;
  // Per-modality delegate target (see /advanced-model): auto by default, or a
  // pinned local/cloud override. delegateOptions is fetched lazily, per
  // modality, only when the "Change" picker for that row is opened.
  // Where the model choices come from: this project's own file
  // (.localforge/models.json) or the user's defaults (none saved for it yet).
  modelsSource: "project" | "defaults" | null;
  localModelTargets: { [modality: string]: LocalModelTarget };
  delegateOptions: { [modality: string]: DelegateOptions };
  systemStats: SystemStats | null;
  memory: MemoryState;
  scratchFiles: ScratchFile[];
  queue: string[];  // New queue field
  planPending: boolean;  // a proposed plan is on screen waiting to be approved
  streamOutput: boolean;  // New ChatState field
  trustRequired: string | null;  // folder path awaiting a trust decision, or null
  // What the running task is doing right now, for the live "Forging with X…"
  // strip (the desktop counterpart of the CLI's status line). null when idle.
  activity: Activity | null;
};

// Times are epoch ms. tokens shown = finishedTokens + localTokens (the
// delegation in flight, estimated from streamed characters until it
// finishes and reports the exact count) + the orchestrator's answer so far.
export type Activity = {
  startedAt: number;
  lastEventAt: number;
  who: string;                     // whichever model is working right now
  detail: string;                  // "planning (round 2)", "writing coding", "downloading x", ...
  finishedTokens: number;
  localTokens: number;
  localStartedAt: number | null;   // when the current delegation began, for tok/s
  answerChars: number;
};
export const initialState: ChatState = {
  messages: [],
  approvals: [],
  todos: [],
  status: "",
  connected: false,
  running: false,
  model: "",
  autoApprove: false,
  usage: {
    frontierPromptTokens: 0,
    frontierCompletionTokens: 0,
    frontierCostUsd: 0,
    localTokensGenerated: 0,
    frontierViaSubscription: false,
    localModels: {},
    delegateTokensGenerated: 0,
    delegateCostUsd: 0,
    delegateNotionalCostUsd: 0,
    paidModels: {}
  },
  usageHistory: { previousSession: null, allTime: null },
  configuredModels: [],
  orchestratorOptions: [],
  backendLog: [],
  backendExited: false,
  backendStartedAt: null,
  trustDeclined: false,
  projectBusy: null,
  budgets: [],
  setup: null,
  setupChecks: null,
  setupResult: null,
  pulls: {},
  wizard: "unknown",
  modelsSource: null,
  localModelTargets: {},
  delegateOptions: {},
  systemStats: null,
  memory: { facts: [], narrative: "", goal: "", progress: [] },
  scratchFiles: [],
  queue: [],  // Initialize queue to an empty array
  planPending: false,
  streamOutput: false,  // Initialize streamOutput to false
  trustRequired: null,
  activity: null
};

function toUsageTotals(raw: any): UsageTotals | null {
  if (!raw || typeof raw !== "object") return null;
  return {
    frontierPromptTokens: Number(raw.frontier_prompt_tokens ?? 0),
    frontierCompletionTokens: Number(raw.frontier_completion_tokens ?? 0),
    frontierCostUsd: Number(raw.frontier_cost_usd ?? 0),
    localTokensGenerated: Number(raw.local_tokens_generated ?? 0),
    subscriptionCostUsd: Number(raw.subscription_cost_usd ?? 0),
    delegateTokensGenerated: Number(raw.delegate_tokens_generated ?? 0),
    delegateCostUsd: Number(raw.delegate_cost_usd ?? 0),
    delegateNotionalCostUsd: Number(raw.delegate_notional_cost_usd ?? 0),
    tasks: Number(raw.tasks ?? 0),
    models: Array.isArray(raw.models) ? raw.models.map(String) : [],
  };
}

// Apply fn to the last assistant message, returning a new array.
function updateLastAssistant(messages: Message[], fn: (m: Message) => Message): Message[] {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      return messages.map((m, j) => (j === i ? fn(m) : m));
    }
  }
  return messages;
}

function updateAssistant(state: ChatState, fn: (m: Message) => Message): ChatState {
  return { ...state, messages: updateLastAssistant(state.messages, fn) };
}

// Replace the last item matching pred in the last assistant message.
function updateLastItem(state: ChatState, pred: (it: Item) => boolean, fn: (it: Item) => Item): ChatState {
  return updateAssistant(state, m => {
    for (let k = m.items.length - 1; k >= 0; k--) {
      if (pred(m.items[k])) {
        return { ...m, items: m.items.map((it, j) => (j === k ? fn(it) : it)) };
      }
    }
    return m;
  });
}

function addItem(state: ChatState, item: Item): ChatState {
  return updateAssistant(state, m => ({ ...m, items: [...m.items, item] }));
}

// Accumulate per-run frontier stats from the Python RunStats dataclass (snake_case).
// local_tokens_generated is ignored here: it is already counted on delegate_finished.
function withStats(state: ChatState, stats: any): ChatState {
  if (!stats || typeof stats !== "object") return state;
  // Per paid model: the orchestrator (under the model it ran as), plus each
  // paid delegate the run reported by name.
  const paid = { ...state.usage.paidModels };
  const bump = (name: string, add: Partial<PaidUsage> & { role: string }) => {
    const cur = paid[name] ?? { roles: [], promptTokens: 0, completionTokens: 0, costUsd: 0, notionalCostUsd: 0, viaSubscription: false, runs: 0 };
    paid[name] = {
      roles: cur.roles.includes(add.role) ? cur.roles : [...cur.roles, add.role],
      promptTokens: cur.promptTokens + (add.promptTokens ?? 0),
      completionTokens: cur.completionTokens + (add.completionTokens ?? 0),
      costUsd: cur.costUsd + (add.costUsd ?? 0),
      notionalCostUsd: cur.notionalCostUsd + (add.notionalCostUsd ?? 0),
      viaSubscription: cur.viaSubscription || !!add.viaSubscription,
      runs: cur.runs + (add.runs ?? 0),
    };
  };
  const promptTokens = Number(stats.frontier_prompt_tokens ?? 0);
  const completionTokens = Number(stats.frontier_completion_tokens ?? 0);
  if (state.model && (promptTokens || completionTokens || Number(stats.frontier_cost_usd ?? 0))) {
    const subscription = Boolean(stats.frontier_via_subscription);
    bump(state.model, {
      role: "orchestrator", promptTokens, completionTokens,
      costUsd: subscription ? 0 : Number(stats.frontier_cost_usd ?? 0),
      notionalCostUsd: subscription ? Number(stats.frontier_cost_usd ?? 0) : 0,
      viaSubscription: subscription, runs: 1,
    });
  }
  for (const [name, m] of Object.entries<any>(stats.delegate_models && typeof stats.delegate_models === "object" ? stats.delegate_models : {})) {
    const roles: string[] = Array.isArray(m?.roles) && m.roles.length ? m.roles.map(String) : ["delegate"];
    roles.forEach((role, i) =>
      bump(name, {
        role,
        // counted once, under the first role: the numbers are per model, not per role
        completionTokens: i === 0 ? Number(m?.tokens ?? 0) : 0,
        costUsd: i === 0 ? Number(m?.cost_usd ?? 0) : 0,
        notionalCostUsd: i === 0 ? Number(m?.notional_cost_usd ?? 0) : 0,
        viaSubscription: m?.kind === "cli",
        runs: i === 0 ? Number(m?.runs ?? 0) : 0,
      }),
    );
  }
  return {
    ...state,
    usage: {
      ...state.usage,
      frontierPromptTokens: state.usage.frontierPromptTokens + promptTokens,
      frontierCompletionTokens: state.usage.frontierCompletionTokens + completionTokens,
      frontierCostUsd: state.usage.frontierCostUsd + Number(stats.frontier_cost_usd ?? 0),
      frontierViaSubscription: Boolean(stats.frontier_via_subscription),
      delegateTokensGenerated: state.usage.delegateTokensGenerated + Number(stats.delegate_tokens_generated ?? 0),
      delegateCostUsd: state.usage.delegateCostUsd + Number(stats.delegate_cost_usd ?? 0),
      delegateNotionalCostUsd: state.usage.delegateNotionalCostUsd + Number(stats.delegate_notional_cost_usd ?? 0),
      paidModels: paid,
    }
  };
}

const CHARS_PER_TOKEN = 4;  // rough, for the live count only; delegate_finished carries the exact number

function brief(text: string, max = 60): string {
  const oneLine = text.replace(/\s+/g, " ").trim();
  return oneLine.length > max ? oneLine.slice(0, max - 1) + "…" : oneLine;
}

// Keeps `activity` in step with the run: started on run_started, refreshed by
// every event that means progress, cleared when the run ends. Layered over the
// main reducer rather than woven into its cases, so the two stay independent.
function trackActivity(next: ChatState, ev: any): ChatState {
  const now = Date.now();
  switch (ev.type) {
    case "run_started":
      return {
        ...next,
        activity: {
          startedAt: now, lastEventAt: now, who: next.model, detail: "starting",
          finishedTokens: 0, localTokens: 0, localStartedAt: null, answerChars: 0,
        },
      };
    case "run_finished":
    case "run_cancelled":
    case "error":
      return next.activity ? { ...next, activity: null } : next;
  }
  const a = next.activity;
  if (!a) return next;
  const set = (patch: Partial<Activity>): ChatState => ({ ...next, activity: { ...a, lastEventAt: now, ...patch } });
  switch (ev.type) {
    case "frontier_round":
      return set({ who: next.model, detail: `planning (round ${ev.round ?? "?"})`, localStartedAt: null, localTokens: 0 });
    case "text_delta":
      return set({ who: next.model, detail: "writing the answer", answerChars: a.answerChars + String(ev.text ?? "").length });
    case "tool_call_started":
      return set({ who: next.model, detail: brief(`${ev.name ?? "tool"} ${ev.summary ?? ""}`) });
    case "delegate_started":
      return set({ who: String(ev.model ?? "a model"), detail: `writing ${ev.modality ?? "code"}`, localStartedAt: now, localTokens: 0 });
    case "delegate_token":
      return set({ localTokens: a.localTokens + Math.ceil(String(ev.text ?? "").length / CHARS_PER_TOKEN) });
    case "delegate_finished":
      return set({
        who: next.model, detail: "reviewing the result", localStartedAt: null, localTokens: 0,
        finishedTokens: a.finishedTokens + Number(ev.tokens ?? 0),
      });
    case "model_pull": {
      const p = ev.progress;
      const pct = p && p.total ? ` ${Math.round((Number(p.completed ?? 0) * 100) / Number(p.total))}%` : "";
      return set({ who: String(ev.model ?? "a model"), detail: `downloading${pct}` });
    }
    case "tool_call_finished":
    case "todos_updated":
    case "approval_request":
    case "approval_auto":
      return set({});
    default:
      return next;
  }
}

export function applyEvent(state: ChatState, ev: any): ChatState {
  return trackActivity(applyEventBase(state, ev), ev);
}

function applyEventBase(state: ChatState, ev: any): ChatState {
  switch (ev.type) {
    case "ready":
      return { ...state, connected: true, model: ev.model ?? "", autoApprove: !!ev.auto_approve, streamOutput: ev.stream_output ?? state.streamOutput };
    case "trust_required":
      return { ...state, trustRequired: String(ev.folder ?? "") };
    case "trust_result":
      // A "no" ends the session -- the backend closes and localforge-exit
      // (App.tsx) sets connected: false, so there's nothing to clear here
      // beyond the prompt itself.
      return { ...state, trustRequired: null, trustDeclined: !ev.trusted };
    case "run_started":
      return { ...state, running: true, planPending: false, status: "", messages: [...state.messages, { role: "assistant", text: "", items: [] }] };
    case "plan_approved":
      return addSystemMessage({ ...state, planPending: false }, "Plan approved — building it now.");
    case "plan_cancelled":
      return { ...state, planPending: false };
    case "frontier_round":
      return updateAssistant(state, m => ({ ...m, round: ev.round }));
    case "text_delta":
      return updateAssistant(state, m => ({ ...m, text: m.text + String(ev.text ?? "") }));
    case "tool_call_started":
      return addItem(state, { kind: "tool", name: String(ev.name ?? ""), summary: String(ev.summary ?? "") });
    case "tool_call_finished":
      return updateLastItem(
        state,
        it => it.kind === "tool" && it.name === ev.name && it.result === undefined,
        it => (it.kind === "tool" ? { ...it, result: String(ev.result ?? "") } : it)
      );
    case "delegate_started": {
      const name = String(ev.model ?? "");
      const runtime = String(ev.runtime ?? "ollama");
      const started = addItem(state, { kind: "delegate", modality: String(ev.modality ?? ""), model: name, runtime, provider: String(ev.provider ?? ""), output: "", done: false });
      // A paid model isn't a "local model": it's counted under paid models (from the run's stats).
      if (runtime === "api" || runtime === "cli") return started;
      const entry = started.usage.localModels[name] || { runs: 0, tokens: 0, active: false };
      return {
        ...started,
        usage: {
          ...started.usage,
          localModels: {
            ...started.usage.localModels,
            [name]: { ...entry, active: true, runs: entry.runs + 1 }
          }
        }
      };
    }
    case "delegate_token":
      return updateLastItem(state, it => it.kind === "delegate", it => (it.kind === "delegate" ? { ...it, output: it.output + String(ev.text ?? "") } : it));
    case "delegate_finished": {
      const tokens = Number(ev.tokens ?? 0);
      const finished = updateLastItem(
        state,
        it => it.kind === "delegate" && !it.done,
        it => (it.kind === "delegate" ? { ...it, done: true, tokens, seconds: Number(ev.seconds ?? 0) } : it)
      );
      if (ev.runtime === "api" || ev.runtime === "cli") return finished;  // paid tokens come with the run's stats, not as local output
      const localModels: Usage["localModels"] = {};
      for (const [name, entry] of Object.entries(finished.usage.localModels)) {
        localModels[name] = entry.active ? { ...entry, active: false, tokens: entry.tokens + tokens } : entry;
      }
      return {
        ...finished,
        usage: { ...finished.usage, localModels, localTokensGenerated: finished.usage.localTokensGenerated + tokens }
      };
    }
    case "model_pull":
      return { ...state, status: `Pulling ${ev.model}…` };
    case "todos_updated":
      return { ...state, todos: Array.isArray(ev.todos) ? ev.todos : [] };
    case "approval_request":
      return { ...state, approvals: [...state.approvals, { id: String(ev.id), kind: String(ev.kind ?? ""), title: String(ev.title ?? ""), detail: String(ev.detail ?? "") }] };
    case "approval_auto":
      return addItem(state, { kind: "note", text: `Auto-approved: ${ev.title}` });
    case "run_finished":
      return withStats(updateAssistant({ ...state, running: false, status: "", planPending: !!ev.plan_pending }, m => ({ ...(m.text === "" ? { ...m, text: String(ev.answer ?? "") } : m), plan: !!ev.plan_pending })), ev.stats);
    case "run_cancelled":
      return addItem({ ...state, running: false, status: "", approvals: [] }, { kind: "note", text: "Stopped." });
    case "error":
      return withStats(addError({ ...state, running: false, approvals: [] }, String(ev.message ?? "error"), ev.detail ? String(ev.detail) : undefined), ev.stats);
    case "settings":
      return { ...state, autoApprove: !!ev.auto_approve, model: ev.model ?? state.model, streamOutput: ev.stream_output ?? state.streamOutput };
    case "system_stats":
      return { ...state, systemStats: { hardware: ev.hardware ?? {}, cpuPercent: Number(ev.cpu_percent ?? 0), ramUsedGb: Number(ev.ram_used_gb ?? 0), ramTotalGb: Number(ev.ram_total_gb ?? 0) } };
    case "memory":
      return { ...state, memory: { facts: Array.isArray(ev.facts) ? ev.facts : [], narrative: String(ev.narrative ?? ""), goal: String(ev.goal ?? ""), progress: Array.isArray(ev.progress) ? ev.progress.map(String) : [] } };
    case "scratch":
      return { ...state, scratchFiles: Array.isArray(ev.files) ? ev.files : [] };
    case "session_reset":
      // usageHistory and configuredModels are project-level, not session-level
      // (they don't reset when the conversation does -- the folder is unchanged).
      return { ...initialState, connected: state.connected, backendStartedAt: state.backendStartedAt, model: state.model, autoApprove: state.autoApprove, streamOutput: state.streamOutput, usageHistory: state.usageHistory, configuredModels: state.configuredModels, orchestratorOptions: state.orchestratorOptions, setup: state.setup, pulls: state.pulls, wizard: state.wizard, budgets: state.budgets, localModelTargets: state.localModelTargets, modelsSource: state.modelsSource };
    case "queue":  // Handle the queue event
      return { ...state, queue: ev.items ?? [] };
    case "compacted":
      return addSystemMessage(state, String(ev.message ?? `Compacted ${ev.before_messages} to ${ev.after_messages} (${ev.changed})`));
    case "summary": {
      const summaryText = ev.memory ? `Memory: ${ev.memory}\nMessage count: ${ev.message_count ?? 0}\nChars: ${ev.chars ?? 0}\nModel: ${ev.model ?? ""}` : "Nothing summarised yet.";
      return addSystemMessage(state, summaryText);
    }
    case "note_added":
      return addSystemMessage(state, `Note queued for the running task: ${String(ev.note ?? "")}`);
    case "why": {
      const lines = Array.isArray(ev.lines) ? ev.lines.map(String) : [String(ev.lines ?? "Nothing pending.")];
      return addSystemMessage(state, lines.length ? lines.join("\n") : "Nothing pending.");
    }
    case "command_help": {
      const commands = Array.isArray(ev.commands) ? ev.commands : [];
      const text = commands.map((c: any) => `${c.name} — ${c.description}`).join("\n") || "No commands available.";
      return addSystemMessage(state, text);
    }
    case "model":
      return { ...state, model: ev.model ?? state.model };
    case "models": {
      const models = Array.isArray(ev.models) ? ev.models : [];
      const text = models.length
        ? models.map((m: any) => `${m.modality}: ${m.model?.name ?? "?"} (${m.model?.runtime ?? "?"}, tier ${m.model?.quality_tier ?? "?"})`).join("\n")
        : "No fitting model found for this machine.";
      return addSystemMessage(state, `Best-fit local models:\n${text}`);
    }
    case "installed": {
      const models = Array.isArray(ev.models) ? ev.models : [];
      const text = models.length ? models.map((m: any) => `- ${m.model}`).join("\n") : "No models installed.";
      return addSystemMessage(state, `Installed models:\n${text}`);
    }
    case "catalog": {
      const items = Array.isArray(ev.items) ? ev.items : [];
      const text = items
        .map((c: any) => `${c.name} (${c.modality}, ${c.runtime}) — tier ${c.quality_tier}, ${c.disk_gb} GB`)
        .join("\n");
      return addSystemMessage(state, `Full model catalog:\n${text || "(empty)"}`);
    }
    case "doctor": {
      const checks = Array.isArray(ev.checks) ? ev.checks : [];
      const text = checks.map((c: any) => `${c.ok ? "✓" : "✗"} ${c.name}: ${c.detail}`).join("\n");
      return addSystemMessage(state, `Doctor:\n${text || "(nothing checked)"}`);
    }
    // Requested silently on connect (usage_request/configured_models_request) to
    // populate the sidebar panels, as opposed to the /usage, /models, etc. text
    // commands above, which post their reply into the chat as a system message.
    case "usage":
      return { ...state, usageHistory: { previousSession: toUsageTotals(ev.previous_session), allTime: toUsageTotals(ev.all_time) } };
    case "configured_models": {
      const models = Array.isArray(ev.models) ? ev.models : [];
      return {
        ...state,
        configuredModels: models.map((m: any) => ({
          modality: String(m.modality ?? ""),
          name: String(m.model?.name ?? ""),
          runtime: String(m.model?.runtime ?? ""),
          qualityTier: Number(m.model?.quality_tier ?? 0),
        })),
      };
    }
    case "task_summary": {
      const r = ev.summary ?? {};
      const summary: TaskSummary = {
        outcome: r.outcome === "failed" || r.outcome === "stopped" ? r.outcome : "completed",
        error: r.error ? String(r.error) : null,
        durationS: Number(r.duration_s ?? 0),
        steps: Number(r.steps ?? 0),
        files: Array.isArray(r.files) ? r.files : [],
        delegations: Array.isArray(r.delegations) ? r.delegations : [],
        commands: Array.isArray(r.commands) ? r.commands : [],
        reads: Number(r.reads ?? 0),
        web: Number(r.web ?? 0),
        declined: Array.isArray(r.declined) ? r.declined : [],
        problems: Array.isArray(r.problems) ? r.problems : [],
        plainError: r.plain_error ? String(r.plain_error) : null,
      };
      return { ...state, messages: [...state.messages, { role: "summary", text: "", items: [], summary }] };
    }
    case "model_waiting":  // another window's local model call is running; ours is queued behind it
      return { ...state, status: ev.waiting ? "Waiting for another window's local model to finish…" : (state.status.startsWith("Waiting for another window") ? "" : state.status) };
    case "project_busy":
      return { ...state, projectBusy: String(ev.message ?? "This project is already open in another window."), backendExited: true, connected: false };
    case "system_text":
      return addSystemMessage(state, String(ev.text ?? ""));
    case "budget": {
      const rows = Array.isArray(ev.providers) ? ev.providers : [];
      return {
        ...state,
        budgets: rows.map((r: any) => ({ provider: String(r.provider ?? ""), spent: Number(r.spent ?? 0), limit: typeof r.limit === "number" ? r.limit : null, images: Number(r.images ?? 0) })),
      };
    }
    case "setup_status": {
      const providers = Array.isArray(ev.providers) ? ev.providers : [];
      const oll = ev.ollama ?? {};
      const setup: SetupStatus = {
        needsSetup: !!ev.needs_setup,
        orchestrator: { model: String(ev.orchestrator?.model ?? ""), ready: !!ev.orchestrator?.ready, reason: String(ev.orchestrator?.reason ?? "") },
        providers: providers.map((p: any) => ({
          id: String(p.id ?? ""), label: String(p.label ?? p.id ?? ""), keySet: !!p.key_set, keyUrl: p.key_url ? String(p.key_url) : null,
          cli: p.cli ? { command: String(p.cli.command ?? ""), installed: !!p.cli.installed, installHint: String(p.cli.install_hint ?? ""), loginHint: String(p.cli.login_hint ?? "") } : null,
        })),
        ollama: {
          installed: !!oll.installed, running: !!oll.running,
          models: (Array.isArray(oll.models) ? oll.models : []).map((m: any) => ({ name: String(m.name ?? ""), problem: m.problem ? String(m.problem) : null })),
          suggested: (Array.isArray(oll.suggested) ? oll.suggested : []).map((m: any) => ({
            name: String(m.name ?? ""), diskGb: Number(m.disk_gb ?? 0), qualityTier: Number(m.quality_tier ?? 0),
            modality: String(m.modality ?? ""), problem: m.problem ? String(m.problem) : null,
          })),
        },
      };
      return { ...state, setup };
    }
    case "setup_checks": {
      const checks: SetupCheck[] = (Array.isArray(ev.checks) ? ev.checks : []).map((c: any) => ({
        id: String(c.id ?? ""), label: String(c.label ?? ""), state: c.state === "fail" ? "fail" : c.state === "warn" ? "warn" : "ok",
        detail: String(c.detail ?? ""), fix: c.fix === "planner" || c.fix === "writers" ? c.fix : null,
      }));
      return { ...state, setupChecks: { checks, ok: !!ev.ok } };
    }
    case "pull_progress": {
      const name = String(ev.model ?? "");
      if (!name) return state;
      const num = (v: any) => (typeof v === "number" ? v : null);
      return { ...state, pulls: { ...state.pulls, [name]: { ctx: String(ev.ctx ?? ""), status: String(ev.status ?? ""), completed: num(ev.completed), total: num(ev.total), done: false } } };
    }
    case "pull_result": {
      const name = String(ev.model ?? "");
      if (!name) return state;
      const prev = state.pulls[name];
      return { ...state, pulls: { ...state.pulls, [name]: { ctx: String(ev.ctx ?? ""), status: "", completed: prev?.completed ?? null, total: prev?.total ?? null, done: true, ok: !!ev.ok, message: String(ev.message ?? ""), blocked: !!ev.blocked } } };
    }
    case "setup_result": {
      const result: SetupResult = { ok: !!ev.ok, message: String(ev.message ?? ""), n: (state.setupResult?.n ?? 0) + 1 };
      // A success is worth remembering once the setup screen has gone away.
      const next = { ...state, setupResult: result };
      return result.ok ? addSystemMessage(next, `✓ ${result.message}`) : next;
    }
    case "orchestrator_options": {
      const options = Array.isArray(ev.options) ? ev.options : [];
      return {
        ...state,
        orchestratorOptions: options.map((o: any) => ({ id: String(o.id ?? ""), label: String(o.label ?? o.id ?? ""), group: String(o.group ?? "Other"), provider: String(o.provider ?? ""), via: String(o.via ?? ""), problem: o.problem ? String(o.problem) : null })).filter((o: OrchestratorOption) => o.id),
        model: ev.current ? String(ev.current) : state.model,
      };
    }
    case "advanced_model": {
      const targets = ev.targets && typeof ev.targets === "object" ? ev.targets : {};
      const localModelTargets: ChatState["localModelTargets"] = {};
      for (const [modality, t] of Object.entries<any>(targets)) {
        localModelTargets[modality] = {
          target: String(t?.target ?? "auto"), description: String(t?.description ?? "auto"),
          ...(t?.auto_model ? { autoModel: String(t.auto_model), installed: !!t.installed, diskGb: Number(t.disk_gb ?? 0) } : {}),
        };
      }
      const source = ev.source === "project" || ev.source === "defaults" ? ev.source : state.modelsSource;
      // The first time the project's models are known decides whether it is new (no file yet).
      const wizard: WizardState = state.wizard === "unknown" ? (source === "defaults" ? "active" : "done") : state.wizard;
      return { ...state, localModelTargets, modelsSource: source, wizard };
    }
    case "delegate_options": {
      const modality = String(ev.modality ?? "");
      if (!modality) return state;
      const local = Array.isArray(ev.local) ? ev.local : [];
      const cloud = Array.isArray(ev.cloud) ? ev.cloud : [];
      return {
        ...state,
        delegateOptions: {
          ...state.delegateOptions,
          [modality]: {
            local: local.map((m: any) => ({
              name: String(m.name ?? ""), quality_tier: Number(m.quality_tier ?? 0),
              disk_gb: Number(m.disk_gb ?? 0), installed: Boolean(m.installed),
              problem: m.problem ? String(m.problem) : null,
            })),
            cloud: cloud.map((c: any) => ({ kind: c.kind === "cli" ? "cli" : "api", provider: String(c.provider ?? ""), model: String(c.model ?? ""), priceUsd: typeof c.price_usd === "number" ? c.price_usd : null })),
            current: String(ev.current ?? "auto"),
          },
        },
      };
    }
    default:
      return state;
  }
}

export function addUserMessage(state: ChatState, text: string, imageDataUrl?: string): ChatState {
  return { ...state, messages: [...state.messages, { role: "user", text, items: [], imageDataUrl }] };
}

export function addError(state: ChatState, text: string, detail?: string): ChatState {
  return { ...state, messages: [...state.messages, { role: "error", text, items: [], ...(detail && detail !== text ? { detail } : {}) }] };
}

// Slash-command replies (help, /why, /summary, /model, /models, /installed,
// /catalog, /doctor, ...) are their own message rather than being folded
// into whatever the last assistant message happens to be -- that message
// might belong to an unrelated, already-finished task, or there might not
// be one yet at all (a command typed before the first run), in which case
// the old addItem()-onto-last-assistant path silently dropped the reply.
export function addSystemMessage(state: ChatState, text: string): ChatState {
  return { ...state, messages: [...state.messages, { role: "system", text, items: [] }] };
}

export function setWizard(state: ChatState, wizard: WizardState): ChatState {
  return { ...state, wizard };
}

export function removeApproval(state: ChatState, id: string): ChatState {
  return { ...state, approvals: state.approvals.filter(a => a.id !== id) };
}

// What one paid (cloud) model has used this session: the orchestrator, and any
// task type sent to a paid model (image generation always is).
export type PaidUsage = {
  roles: string[];
  promptTokens: number;
  completionTokens: number;
  costUsd: number;
  notionalCostUsd: number;  // what a subscription's usage would have cost; never shown as billed money
  viaSubscription: boolean;
  runs: number;
};
export type Usage = {
  frontierPromptTokens: number;
  frontierCompletionTokens: number;
  frontierCostUsd: number;
  localTokensGenerated: number;
  frontierViaSubscription: boolean;
  localModels: { [key: string]: { runs: number; tokens: number; active: boolean } };
  // An "advanced" per-modality delegate target (see /advanced-model) can send
  // coding/docs/general to a paid cloud model -- neither free local compute
  // nor orchestrator spend, so it's tracked apart from both.
  delegateTokensGenerated: number;
  delegateCostUsd: number;
  delegateNotionalCostUsd: number;
  // The same, split by paid model name (the orchestrator included), for the
  // "Paid models" list.
  paidModels: { [name: string]: PaidUsage };
};

const MAX_BACKEND_LOG = 60;

// Lines the backend printed to stderr (a startup crash, a dyld error, a traceback).
export function appendBackendLog(state: ChatState, text: string): ChatState {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length === 0) return state;
  return { ...state, backendLog: [...state.backendLog, ...lines].slice(-MAX_BACKEND_LOG) };
}

// The backend process ended. (A folder the user declined to trust also ends it, on purpose.)
export function backendExited(state: ChatState): ChatState {
  return { ...state, connected: false, running: false, backendExited: true };
}
