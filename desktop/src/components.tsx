import type { Approval, DelegateItem, Item, Message, NoteItem, TaskSummary, Todo, ToolItem } from "./state";
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
  const paid = item.runtime === "api" || item.runtime === "cli";
  const provider = item.provider ? item.provider[0].toUpperCase() + item.provider.slice(1) : "";
  return (
    <div className="rounded-sm border border-mx-dim bg-mx-panel2 px-3 py-2 text-sm my-1" data-testid="delegate-card">
      <span className="text-mx-mid">{paid ? "Paid model " : "Local model "}</span>
      <span className={"font-mono font-semibold " + (paid ? "text-mx-amber" : "text-mx-green")}>{item.model}</span>
      {paid && <span className="ml-2 text-xs text-mx-dim">{`${provider} ${item.runtime === "cli" ? "login" : "API key"}`.trim()}</span>}
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
    <div className="m-3 rounded-sm border border-mx-amber bg-mx-panel2 p-3" role="alertdialog" aria-label="Needs your approval" data-testid="approval">
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-mx-amber">● Needs your approval</div>
      <div className="flex items-center">
        <span className="font-semibold text-mx-amber">{a.title}</span>
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

// The plan is the current task's own steps; requests typed while it runs wait in the queue and
// are listed under it as "Up next", so nothing sent is ever out of sight.
export function TodoList({ todos, queued = [] }: { todos: Todo[]; queued?: string[] }) {
  return (
    <div className="space-y-2" data-testid="plan">
      {todos.length === 0 ? (
        <p className="text-xs text-mx-dim">No plan yet — it appears once a task has more than a couple of steps.</p>
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
      {queued.length > 0 && (
        <div data-testid="plan-up-next">
          <h3 className="text-[11px] uppercase tracking-wide text-mx-mid">Up next ({queued.length})</h3>
          <ul className="mt-1 space-y-1 text-xs">
            {queued.map((q, i) => (
              <li key={i} className="flex gap-2 text-mx-dim"><span aria-hidden>{i + 1}.</span><span className="min-w-0 break-words">{q}</span></li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function took(seconds: number): string {
  if (seconds >= 60) return ` in ${Math.floor(seconds / 60)}m ${String(Math.floor(seconds % 60)).padStart(2, "0")}s`;
  return seconds >= 1 ? ` in ${Math.round(seconds)}s` : "";
}

// Shown after every task: what it did and what went wrong -- the desktop
// counterpart of the terminal's "Summary" panel.
export function TaskSummaryCard({ summary: s }: { summary: TaskSummary }) {
  const heading = s.outcome === "failed" ? "Failed" : s.outcome === "stopped" ? "Stopped" : "Done";
  const tone = s.outcome === "failed" ? "border-mx-red text-mx-red" : s.outcome === "stopped" ? "border-mx-amber text-mx-amber" : "border-mx-dim text-mx-green";
  const byModel: { [m: string]: number } = {};
  for (const d of s.delegations) byModel[d.model || "a local model"] = (byModel[d.model || "a local model"] ?? 0) + 1;
  const looked = [s.reads ? `looked through the project ${s.reads} time${s.reads === 1 ? "" : "s"}` : "", s.web ? `searched the web ${s.web} time${s.web === 1 ? "" : "s"}` : ""].filter(Boolean);
  return (
    <div className={`my-2 max-w-[90%] rounded-sm border bg-mx-panel2 px-3 py-2 text-sm ${tone.split(" ")[0]}`} role="region" aria-label="Task summary">
      <div className={`font-semibold ${tone.split(" ")[1]}`}>
        {heading}{took(s.durationS)} · {s.steps} step{s.steps === 1 ? "" : "s"}
      </div>
      <div className="mt-1 space-y-1 text-xs text-mx-mid">
        {s.files.length > 0 && (
          <div>
            <div className="text-mx-dim">Files changed</div>
            <ul className="ml-3 list-disc font-mono">
              {s.files.map((f, i) => (
                <li key={i}>{f.action.toLowerCase()} {f.path}{f.by && <span className="text-mx-dim"> (written by {f.by})</span>}</li>
              ))}
            </ul>
          </div>
        )}
        {s.delegations.length > 0 && <div><span className="text-mx-dim">Delegated to </span>{Object.entries(byModel).map(([m, n]) => `${m} ×${n}`).join(", ")}</div>}
        {s.commands.length > 0 && (
          <div>
            <div className="text-mx-dim">Commands run</div>
            <ul className="ml-3 font-mono">
              {s.commands.map((c, i) => (
                <li key={i} className={c.ok ? "" : "text-mx-red"}>{c.ok ? "✓" : "✗"} {c.command}</li>
              ))}
            </ul>
          </div>
        )}
        {looked.length > 0 && <div className="text-mx-dim">Also {looked.join(" and ")}.</div>}
        {s.declined.length > 0 && (
          <div>
            <div className="text-mx-dim">You declined</div>
            <ul className="ml-3 list-disc font-mono">{s.declined.map((d, i) => <li key={i}>{d.what}</li>)}</ul>
          </div>
        )}
        {s.steps > 0 && s.files.length === 0 && s.commands.length === 0 && s.delegations.length === 0 && looked.length === 0 && (
          <div className="text-mx-dim">No files were changed and no commands were run.</div>
        )}
        {s.problems.length > 0 && (
          <div data-testid="summary-problems">
            <div className="text-mx-red">Problems</div>
            <ul className="ml-3 space-y-1 text-mx-red">
              {s.problems.map((p, i) => (
                <li key={i} className="whitespace-pre-wrap">
                  ✗ {p.plain ?? `${p.what}: ${p.error}`}{p.recovered && <span className="text-mx-dim"> (recovered)</span>}
                  {p.plain && (
                    <details className="text-mx-dim">
                      <summary className="cursor-pointer text-[11px]">Details</summary>
                      <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[10px]">{p.what}: {p.error}</pre>
                    </details>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
        {s.error && (
          <div className="whitespace-pre-wrap text-mx-red" data-testid="summary-error">
            {s.plainError ?? s.error}
            {s.plainError && s.plainError !== s.error && (
              <details className="text-mx-dim">
                <summary className="cursor-pointer text-[11px]">Details</summary>
                <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[10px]">{s.error}</pre>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function MessageView({ message, thinking, awaitingInput = false }: { message: Message; thinking: boolean; awaitingInput?: boolean }) {
  switch (message.role) {
    case "user":
      return (
        <div className="flex justify-end my-2">
          <div className="max-w-[80%] whitespace-pre-wrap rounded-sm border border-mx-dim bg-mx-panel2 px-4 py-2 text-mx-bright">
            {message.imageDataUrl && (
              <img src={message.imageDataUrl} alt="attached" className="mb-2 max-h-48 rounded-sm border border-mx-dim" />
            )}
            {message.text}
          </div>
        </div>
      );
    case "error":
      return (
        <div className="my-2 whitespace-pre-wrap rounded-sm border border-mx-red bg-mx-panel2 px-3 py-2 text-sm text-mx-red" role="alert">
          {message.text}
          {message.detail && (
            <details className="text-mx-dim">
              <summary className="cursor-pointer text-[11px]">Details</summary>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[10px]">{message.detail}</pre>
            </details>
          )}
        </div>
      );
    case "system":
      return (
        <div className="my-2 max-w-[90%] whitespace-pre-wrap rounded-sm border border-mx-dim bg-mx-panel2 px-3 py-2 font-mono text-xs text-mx-mid">{message.text}</div>
      );
    case "summary":
      return message.summary ? <TaskSummaryCard summary={message.summary} /> : null;
    case "assistant":
      return (
        <div className="my-2 max-w-[90%]">
          <ActivityStream items={message.items} open={thinking} round={message.round} />
          {message.text && awaitingInput ? (
            <div className="mt-1 rounded-sm border border-mx-amber bg-mx-panel2 px-3 py-2" role="status" data-testid="needs-input">
              <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-mx-amber">● Needs your input</div>
              <div className="whitespace-pre-wrap leading-relaxed text-mx-amber">{message.text}</div>
              <div className="mt-1 text-[11px] text-mx-dim">Type your answer in the box below.</div>
            </div>
          ) : (
            message.text && <div className="mt-1 whitespace-pre-wrap leading-relaxed">{message.text}</div>
          )}
          {!message.text && thinking && <div className="animate-pulse text-sm text-mx-dim">Thinking…{message.round ? ` round ${message.round}` : ""}</div>}
        </div>
      );
  }
}
