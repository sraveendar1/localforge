import type { Approval, DelegateItem, Item, Message, NoteItem, Todo, ToolItem } from "./state";
import { ActivityStream } from "./ActivityStream";

export function ToolCard({ item }: { item: ToolItem }) {
  return (
    <div className="rounded-sm border border-mx-dim bg-mx-panel2 px-3 py-2 text-sm my-1">
      <span className="font-mono font-semibold text-mx-green">{item.name}</span>
      <span className="text-mx-mid ml-2 break-all">{item.summary}</span>
      {item.result === undefined ? (
        <span className="ml-2 animate-pulse text-mx-dim">…</span>
      ) : (
        <details className="mt-1">
          <summary className="cursor-pointer text-xs text-mx-dim">Result</summary>
          <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap font-mono text-[10px] text-mx-dim">{item.result}</pre>
        </details>
      )}
    </div>
  );
}

export function DelegateCard({ item }: { item: DelegateItem }) {
  return (
    <div className="rounded-sm border border-mx-dim bg-mx-panel2 px-3 py-2 text-sm my-1">
      <span className="text-mx-mid">{`Local model `}</span>
      <span className="font-mono font-semibold text-mx-green">{item.model}</span>
      <span className={"text-xs text-mx-dim ml-2" + (item.done ? "" : " animate-pulse")}>
        {item.done ? `done · ${item.tokens ?? 0} tokens · ${(item.seconds ?? 0).toFixed(1)} s` : `writing…`}
      </span>
      <details>
        <summary className="cursor-pointer text-xs text-mx-dim">Output</summary>
        <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-mx-dim">{item.output}</pre>
      </details>
    </div>
  );
}

export function NoteLine({ item }: { item: NoteItem }) {
  return <div className="my-1 text-xs italic text-mx-dim">{item.text}</div>;
}

export function ItemView({ item }: { item: Item }) {
  switch (item.kind) {
    case "tool":
      return <ToolCard item={item} />;
    case "delegate":
      return <DelegateCard item={item} />;
    case "note":
      return <NoteLine item={item} />;
  }
}

function diffLineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) return "text-mx-dim";
  if (line.startsWith("+")) return "bg-mx-panel2 text-mx-bright";
  if (line.startsWith("-")) return "bg-mx-panel text-mx-red";
  if (line.startsWith("@@")) return "text-mx-mid";
  return "text-mx-dim";
}

export function DiffView({ text }: { text: string }) {
  return (
    <div className="max-h-80 overflow-auto rounded-sm border border-mx-dim bg-mx-panel font-mono text-xs">
      {text.split("\n").map((line, i) => (
        <div key={i} className={diffLineClass(line) + " whitespace-pre px-2"}>{line || " "}</div>
      ))}
    </div>
  );
}

export function ApprovalPanel({ approvals, onDecide }: { approvals: Approval[]; onDecide: (id: string, decision: "approve" | "decline" | "always") => void }) {
  if (approvals.length === 0) return null;
  const a = approvals[0];
  return (
    <div className="m-3 rounded-sm border border-mx-amber bg-mx-panel2 p-3">
      <div className="flex items-center">
        <span className="font-semibold text-mx-green glow">{a.title}</span>
        <span className="ml-2 rounded-sm border border-mx-amber bg-mx-bg px-1.5 py-0.5 text-xs text-mx-amber">{a.kind}</span>
        {approvals.length > 1 && <span className="text-xs text-mx-dim ml-auto">+{approvals.length - 1} more pending</span>}
      </div>
      {a.detail && <div className="mt-2"><DiffView text={a.detail} /></div>}
      <div className="mt-3 flex gap-2">
        <button onClick={() => onDecide(a.id, "approve")} className="rounded-sm border border-mx-mid bg-transparent px-3 py-1 text-sm text-mx-green hover:border-mx-bright hover:text-mx-bright">Approve</button>
        {a.kind !== "delete" && <button onClick={() => onDecide(a.id, "always")} className="rounded-sm border border-mx-dim bg-transparent px-3 py-1 text-sm text-mx-mid hover:border-mx-mid hover:text-mx-bright">Always allow</button>}
        <button onClick={() => onDecide(a.id, "decline")} className="rounded-sm border border-mx-red bg-transparent px-3 py-1 text-sm text-mx-red hover:border-mx-bright hover:text-mx-bright">Decline</button>
      </div>
    </div>
  );
}

export function TodoList({ todos }: { todos: Todo[] }) {
  return (
    <div>
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-mx-mid glow">Plan</h2>
      {todos.length === 0 ? (
        <p className="text-xs text-mx-dim">No plan yet.</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {todos.map((todo, i) => (
            <li key={i} className="flex gap-2">
              <span className={todo.status === "pending" ? "text-mx-dim" : todo.status === "in_progress" ? "text-mx-amber" : "text-mx-green"}>{todo.status === "completed" ? "●" : todo.status === "in_progress" ? "◐" : "○"}</span>
              <span className={todo.status === "completed" ? "line-through text-mx-dim" : ""}>{todo.content}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function MessageView({ message, thinking }: { message: Message; thinking: boolean }) {
  switch (message.role) {
    case "user":
      return (
        <div className="flex justify-end my-2">
          <div className="max-w-[80%] whitespace-pre-wrap rounded-sm border border-mx-dim bg-mx-panel2 px-4 py-2 text-mx-bright">{message.text}</div>
        </div>
      );
    case "error":
      return (
        <div className="my-2 whitespace-pre-wrap rounded-sm border border-mx-red bg-mx-panel2 px-3 py-2 text-sm text-mx-red">{message.text}</div>
      );
    case "assistant":
      return (
        <div className="my-2 max-w-[90%]">
          <ActivityStream items={message.items} open={thinking} round={message.round} />
          {message.text && <div className="mt-1 whitespace-pre-wrap leading-relaxed">{message.text}</div>}
          {!message.text && thinking && <div className="animate-pulse text-sm text-mx-dim">Thinking…{message.round ? ` round ${message.round}` : ""}</div>}
        </div>
      );
  }
}
