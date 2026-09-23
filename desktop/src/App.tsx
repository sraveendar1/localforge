import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { ApprovalPanel, MessageView, TodoList } from "./components";
import { addError, addUserMessage, applyEvent, initialState, removeApproval } from "./state";
import type { ChatState } from "./state";
import "./App.css";

function App() {
const [chat, setChat] = useState<ChatState>(initialState);
const [folder, setFolder] = useState<string | null>(null);
const [input, setInput] = useState("");
const [modelDraft, setModelDraft] = useState("");
const [logs, setLogs] = useState<string[]>([]);
const endRef = useRef<HTMLDivElement>(null);

useEffect(() => {
  const ps = [
    listen<string>("localforge-event", e => {
      let ev: any;
      try { ev = JSON.parse(e.payload); } catch { return; }
      setChat(s => applyEvent(s, ev));
    }),
    listen<string>("localforge-stderr", e => setLogs(l => [...l, e.payload].slice(-200))),
    listen("localforge-exit", () => setChat(s => ({ ...s, connected: false, running: false }))),
  ];
  return () => { ps.forEach(p => p.then(f => f())); };
}, []);

useEffect(() => { setModelDraft(chat.model); }, [chat.model]);

useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chat.messages]);

const send = (obj: object) => invoke("send_message", { message: JSON.stringify(obj) }).catch(err => setChat(s => addError(s, String(err))));

async function openFolder() {
  const dir = await open({ directory: true, multiple: false });
  if (typeof dir !== "string") return;
  setFolder(dir);
  setLogs([]);
  setChat(initialState);
  try { await invoke("start_session", { folder: dir, model: modelDraft.trim() || null }); } catch (err) { setChat(s => addError(s, String(err))); }
}

function submit() {
  const text = input.trim();
  if (!text || !chat.connected || chat.running) return;
  setChat(s => addUserMessage(s, text));
  setInput("");
  send({ type: "user_message", text });
}

function commitModel() {
  const m = modelDraft.trim();
  if (chat.connected && !chat.running && m && m !== chat.model) send({ type: "set_model", model: m });
}

function decide(id: string, decision: "approve" | "decline" | "always") {
  send({ type: "approval_response", id, decision });
  setChat(s => removeApproval(s, id));
}

const lastAssistant = chat.messages.map(m => m.role).lastIndexOf("assistant");

return (
  <div className="flex h-full flex-col bg-neutral-950 text-neutral-100">
    <header className="flex items-center gap-3 border-b border-neutral-800 px-4 py-2">
      <span className="font-semibold">localforge</span>
      <span className={"h-2 w-2 rounded-full " + (chat.connected ? "bg-green-500" : "bg-neutral-600")} title={chat.connected ? "Connected" : "Not connected"} />
      <span className="min-w-0 flex-1 truncate text-sm text-neutral-400" title={folder ?? ""}>{folder ?? "No folder"}</span>
      <input className="w-44 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm" placeholder="model" value={modelDraft} onChange={e => setModelDraft(e.target.value)} onBlur={commitModel} onKeyDown={e => { if (e.key === "Enter") commitModel(); }} />
      <label className="flex items-center gap-1 text-sm"><input type="checkbox" checked={chat.autoApprove} disabled={!chat.connected} onChange={e => send({ type: "set_auto", enabled: e.target.checked })} />Auto-approve</label>
      <button className="rounded bg-neutral-800 px-3 py-1 text-sm hover:bg-neutral-700" onClick={openFolder}>Open folder…</button>
    </header>
    {chat.status && <div className="border-b border-neutral-800 px-4 py-1 text-xs text-neutral-400">{chat.status}</div>}
    <div className="flex min-h-0 flex-1">
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto px-4 py-3">
          {!folder ? <div className="flex h-full items-center justify-center text-neutral-500">Open a folder to start.</div> : chat.messages.map((m, i) => <MessageView key={i} message={m} thinking={chat.running && i === lastAssistant} />)}
          <div ref={endRef} />
        </div>
        <ApprovalPanel approvals={chat.approvals} onDecide={decide} />
        <div className="flex gap-2 border-t border-neutral-800 p-3">
          <textarea rows={3} className="flex-1 resize-none rounded border border-neutral-700 bg-neutral-900 p-2 text-sm" placeholder={chat.connected ? "Ask localforge…" : "Open a folder to start"} value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
          {chat.running ? <button className="self-end rounded bg-red-700 px-4 py-2 text-sm text-white hover:bg-red-600" onClick={() => send({ type: "cancel" })}>Stop</button> : <button className="self-end rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-500 disabled:opacity-40" disabled={!chat.connected || !input.trim()} onClick={submit}>Send</button>}
        </div>
      </main>
      <aside className="w-72 overflow-y-auto border-l border-neutral-800 p-3">
        <TodoList todos={chat.todos} />
        <details className="mt-4">
          <summary className="cursor-pointer text-xs text-neutral-400">Logs ({logs.length})</summary>
          <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap font-mono text-[10px] text-neutral-500">{logs.join("\n")}</pre>
        </details>
      </aside>
    </div>
  </div>
);
}

export default App;
