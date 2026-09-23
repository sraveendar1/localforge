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
};
export const initialState: ChatState = { messages: [], approvals: [], todos: [], status: "", connected: false, running: false, model: "", autoApprove: false };

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
        it => (it.kind === "tool" ? { ...it, result: String(ev.result ?? "") } : it),
      );
    case "delegate_started":
      return addItem(state, { kind: "delegate", modality: String(ev.modality ?? ""), model: String(ev.model ?? ""), output: "", done: false });
    case "delegate_token":
      return updateLastItem(state, it => it.kind === "delegate", it => (it.kind === "delegate" ? { ...it, output: it.output + String(ev.text ?? "") } : it));
    case "delegate_finished":
      return updateLastItem(state, it => it.kind === "delegate", it => (it.kind === "delegate" ? { ...it, done: true, tokens: ev.tokens, seconds: ev.seconds } : it));
    case "model_pull":
      return { ...state, status: `Pulling ${ev.model}…` };
    case "todos_updated":
      return { ...state, todos: Array.isArray(ev.todos) ? ev.todos : [] };
    case "approval_request":
      return { ...state, approvals: [...state.approvals, { id: String(ev.id), kind: String(ev.kind ?? ""), title: String(ev.title ?? ""), detail: String(ev.detail ?? "") }] };
    case "approval_auto":
      return addItem(state, { kind: "note", text: `Auto-approved: ${ev.title}` });
    case "run_finished":
      return updateAssistant({ ...state, running: false, status: "" }, m => (m.text === "" ? { ...m, text: String(ev.answer ?? "") } : m));
    case "run_cancelled":
      return addItem({ ...state, running: false, status: "", approvals: [] }, { kind: "note", text: "Stopped." });
    case "error":
      return addError({ ...state, running: false, approvals: [] }, String(ev.message ?? "error"));
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
