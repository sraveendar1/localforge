import type { Approval, DelegateItem, Item, Message, NoteItem, Todo, ToolItem } from "./state";

export function ToolCard({ item }: { item: ToolItem }) {
  return (
    <div className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm my-1">
      <span className="font-mono font-semibold">{item.name}</span>
      <span className="text-neutral-400 ml-2 break-all">{item.summary}</span>
      {item.result === undefined ? (
        <span className="ml-2 animate-pulse text-neutral-500">…</span>
      ) : (
        <details className="mt-1">
          <summary className="cursor-pointer text-xs text-neutral-500">Result</summary>
          <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-neutral-300">{item.result}</pre>
        </details>
      )}
    </div>
  );
}

export function DelegateCard({ item }: { item: DelegateItem }) {
  return (
    <div className="rounded-md border border-violet-800 bg-neutral-900 px-3 py-2 text-sm my-1">
      <span className="text-neutral-400">{`Local model `}</span>
      <span className="font-mono font-semibold">{item.model}</span>
      <span className="text-neutral-400 ml-2">{` · ${item.modality}`}</span>
      <span className={"text-xs text-neutral-400 ml-2" + (item.done ? "" : " animate-pulse")}>
        {item.done ? `done · ${item.tokens ?? 0} tokens · ${(item.seconds ?? 0).toFixed(1)} s` : `writing…`}
      </span>
      <details>
        <summary className="cursor-pointer text-xs text-neutral-500">Output</summary>
        <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-neutral-300">{item.output}</pre>
      </details>
    </div>
  );
}

export function NoteLine({ item }: { item: NoteItem }) {
  return <div className="my-1 text-xs italic text-neutral-500">{item.text}</div>;
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
  if (line.startsWith("+++") || line.startsWith("---")) return "text-neutral-400";
  if (line.startsWith("+")) return "bg-green-950 text-green-300";
  if (line.startsWith("-")) return "bg-red-950 text-red-300";
  if (line.startsWith("@@")) return "text-cyan-400";
  return "text-neutral-300";
}

export function DiffView({ text }: { text: string }) {
  return (
    <div className="max-h-80 overflow-auto rounded border border-neutral-800 bg-neutral-900 font-mono text-xs">
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
    <div className="m-3 rounded-lg border border-amber-500 bg-neutral-900 p-3">
      <div className="flex items-center">
        <span className="font-semibold">{a.title}</span>
        <span className="ml-2 rounded bg-amber-900 px-1.5 py-0.5 text-xs text-amber-200">{a.kind}</span>
        {approvals.length > 1 && <span className="text-xs text-neutral-400 ml-auto">+{approvals.length - 1} more pending</span>}
      </div>
      {a.detail && <div className="mt-2"><DiffView text={a.detail} /></div>}
      <div className="mt-3 flex gap-2">
        <button onClick={() => onDecide(a.id, "approve")} className="bg-green-700 hover:bg-green-600 rounded px-3 py-1 text-sm text-white">Approve</button>
        {a.kind !== "delete" && <button onClick={() => onDecide(a.id, "always")} className="bg-neutral-700 hover:bg-neutral-600 rounded px-3 py-1 text-sm text-white">Always allow</button>}
        <button onClick={() => onDecide(a.id, "decline")} className="bg-red-700 hover:bg-red-600 rounded px-3 py-1 text-sm text-white">Decline</button>
      </div>
    </div>
  );
}

export function TodoList({ todos }: { todos: Todo[] }) {
  return (
    <div>
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-400">Plan</h2>
      {todos.length === 0 ? (
        <p className="text-xs text-neutral-600">No plan yet.</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {todos.map((todo, i) => (
            <li key={i} className="flex gap-2">
              <span className={todo.status === "pending" ? "text-neutral-400" : todo.status === "in_progress" ? "text-amber-400" : "text-green-500"}>{todo.status === "completed" ? "●" : todo.status === "in_progress" ? "◐" : "○"}</span>
              <span className={todo.status === "completed" ? "line-through text-neutral-500" : ""}>{todo.content}</span>
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
          <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-blue-600 px-4 py-2 text-white">{message.text}</div>
        </div>
      );
    case "error":
      return (
        <div className="my-2 whitespace-pre-wrap rounded border border-red-700 bg-red-950 px-3 py-2 text-sm text-red-300">{message.text}</div>
      );
    case "assistant":
      return (
        <div className="my-2 max-w-[90%]">
          {message.items.map((item, i) => <ItemView key={i} item={item} />)}
          {message.text && <div className="mt-1 whitespace-pre-wrap leading-relaxed">{message.text}</div>}
          {!message.text && thinking && <div className="animate-pulse text-sm text-neutral-500">Thinking…{message.round ? ` round ${message.round}` : ""}</div>}
        </div>
      );
  }
}
