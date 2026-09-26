export type ToolItem = { kind: "tool"; name: string; summary: string; result?: string };
export type DelegateItem = { kind: "delegate"; modality: string; model: string; output: string; done: boolean; tokens?: number; seconds?: number };
export type NoteItem = { kind: "note"; text: string };
export type Item = ToolItem | DelegateItem | NoteItem;
export type Message = { role: "user" | "assistant" | "error" | "system"; text: string; items: Item[]; round?: number; imageDataUrl?: string };
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
  };
  cpuPercent: number;
  ramUsedGb: number;
  ramTotalGb: number;
};
export type MemoryFact = { name: string; description?: string; content?: string; type?: string };
export type MemoryState = { facts: MemoryFact[]; narrative: string; goal: string };
export type ScratchFile = { path: string; size: number };
export type UsageTotals = {
  frontierPromptTokens: number;
  frontierCompletionTokens: number;
  frontierCostUsd: number;
  localTokensGenerated: number;
  subscriptionCostUsd: number;
  tasks: number;
  models: string[];
};
export type ConfiguredModel = { modality: string; name: string; runtime: string; qualityTier: number };
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
  systemStats: SystemStats | null;
  memory: MemoryState;
  scratchFiles: ScratchFile[];
  queue: string[];  // New queue field
  streamOutput: boolean;  // New ChatState field
  trustRequired: string | null;  // folder path awaiting a trust decision, or null
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
    localModels: {}
  },
  usageHistory: { previousSession: null, allTime: null },
  configuredModels: [],
  systemStats: null,
  memory: { facts: [], narrative: "", goal: "" },
  scratchFiles: [],
  queue: [],  // Initialize queue to an empty array
  streamOutput: false,  // Initialize streamOutput to false
  trustRequired: null
};

function toUsageTotals(raw: any): UsageTotals | null {
  if (!raw || typeof raw !== "object") return null;
  return {
    frontierPromptTokens: Number(raw.frontier_prompt_tokens ?? 0),
    frontierCompletionTokens: Number(raw.frontier_completion_tokens ?? 0),
    frontierCostUsd: Number(raw.frontier_cost_usd ?? 0),
    localTokensGenerated: Number(raw.local_tokens_generated ?? 0),
    subscriptionCostUsd: Number(raw.subscription_cost_usd ?? 0),
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
  return {
    ...state,
    usage: {
      ...state.usage,
      frontierPromptTokens: state.usage.frontierPromptTokens + Number(stats.frontier_prompt_tokens ?? 0),
      frontierCompletionTokens: state.usage.frontierCompletionTokens + Number(stats.frontier_completion_tokens ?? 0),
      frontierCostUsd: state.usage.frontierCostUsd + Number(stats.frontier_cost_usd ?? 0),
      frontierViaSubscription: Boolean(stats.frontier_via_subscription)
    }
  };
}

export function applyEvent(state: ChatState, ev: any): ChatState {
  switch (ev.type) {
    case "ready":
      return { ...state, connected: true, model: ev.model ?? "", autoApprove: !!ev.auto_approve, streamOutput: ev.stream_output ?? state.streamOutput };
    case "trust_required":
      return { ...state, trustRequired: String(ev.folder ?? "") };
    case "trust_result":
      // A "no" ends the session -- the backend closes and localforge-exit
      // (App.tsx) sets connected: false, so there's nothing to clear here
      // beyond the prompt itself.
      return { ...state, trustRequired: ev.trusted ? null : state.trustRequired };
    case "run_started":
      return { ...state, running: true, status: "", messages: [...state.messages, { role: "assistant", text: "", items: [] }] };
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
      const started = addItem(state, { kind: "delegate", modality: String(ev.modality ?? ""), model: name, output: "", done: false });
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
      // This event carries no model name, so credit whichever local model is active.
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
      return withStats(updateAssistant({ ...state, running: false, status: "" }, m => (m.text === "" ? { ...m, text: String(ev.answer ?? "") } : m)), ev.stats);
    case "run_cancelled":
      return addItem({ ...state, running: false, status: "", approvals: [] }, { kind: "note", text: "Stopped." });
    case "error":
      return withStats(addError({ ...state, running: false, approvals: [] }, String(ev.message ?? "error")), ev.stats);
    case "settings":
      return { ...state, autoApprove: !!ev.auto_approve, model: ev.model ?? state.model, streamOutput: ev.stream_output ?? state.streamOutput };
    case "system_stats":
      return { ...state, systemStats: { hardware: ev.hardware ?? {}, cpuPercent: Number(ev.cpu_percent ?? 0), ramUsedGb: Number(ev.ram_used_gb ?? 0), ramTotalGb: Number(ev.ram_total_gb ?? 0) } };
    case "memory":
      return { ...state, memory: { facts: Array.isArray(ev.facts) ? ev.facts : [], narrative: String(ev.narrative ?? ""), goal: String(ev.goal ?? "") } };
    case "scratch":
      return { ...state, scratchFiles: Array.isArray(ev.files) ? ev.files : [] };
    case "session_reset":
      // usageHistory and configuredModels are project-level, not session-level
      // (they don't reset when the conversation does -- the folder is unchanged).
      return { ...initialState, model: state.model, autoApprove: state.autoApprove, streamOutput: state.streamOutput, usageHistory: state.usageHistory, configuredModels: state.configuredModels };
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
    default:
      return state;
  }
}

export function addUserMessage(state: ChatState, text: string, imageDataUrl?: string): ChatState {
  return { ...state, messages: [...state.messages, { role: "user", text, items: [], imageDataUrl }] };
}

export function addError(state: ChatState, text: string): ChatState {
  return { ...state, messages: [...state.messages, { role: "error", text, items: [] }] };
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

export function removeApproval(state: ChatState, id: string): ChatState {
  return { ...state, approvals: state.approvals.filter(a => a.id !== id) };
}

export type Usage = {
  frontierPromptTokens: number;
  frontierCompletionTokens: number;
  frontierCostUsd: number;
  localTokensGenerated: number;
  frontierViaSubscription: boolean;
  localModels: { [key: string]: { runs: number; tokens: number; active: boolean } };
};
