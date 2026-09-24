export type ToolItem = { kind: "tool"; name: string; summary: string; result?: string };
export type DelegateItem = { kind: "delegate"; modality: string; model: string; output: string; done: boolean; tokens?: number; seconds?: number };
export type NoteItem = { kind: "note"; text: string };
export type Item = ToolItem | DelegateItem | NoteItem;
export type Message = { role: "user" | "assistant" | "error"; text: string; items: Item[]; round?: number };
export type Approval = { id: string; kind: string; title: string; detail: string };
export type Todo = { content: string; status: "pending" | "in_progress" | "completed" };
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
};
export const initialState: ChatState = { messages: [], approvals: [], todos: [], status: "", connected: false, running: false, model: "", autoApprove: false, usage: { frontierPromptTokens: 0, frontierCompletionTokens: 0, frontierCostUsd: 0, localTokensGenerated: 0, frontierViaSubscription: false, localModels: {} } };

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
      return { ...state, connected: true, model: ev.model ?? "", autoApprove: !!ev.auto_approve };
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
      return { ...state, autoApprove: !!ev.auto_approve, model: ev.model ?? state.model };
    default:
      return state;
  }
}

export function addUserMessage(state: ChatState, text: string): ChatState {
  return { ...state, messages: [...state.messages, { role: "user", text, items: [] }] };
}

export function addError(state: ChatState, text: string): ChatState {
  return { ...state, messages: [...state.messages, { role: "error", text, items: [] }] };
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
