import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { ApprovalPanel, MessageView, TodoList } from "./components";
import { UsagePanel } from "./UsagePanel";
import { StatusBar } from "./StatusBar";
import { ModelPicker } from "./ModelPicker";
import { addError, addUserMessage, applyEvent, initialState, removeApproval } from "./state";
import type { ChatState } from "./state";
import "./App.css";

function App() {
  const [chat, setChat] = useState<ChatState>(initialState);
  const [folder, setFolder] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [modelDraft, setModelDraft] = useState("claude-opus-5");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ps = [
      listen<string>("localforge-event", e => {
        let ev: any;
        try { ev = JSON.parse(e.payload); } catch { return; }
        setChat(s => applyEvent(s, ev));
      }),
      listen("localforge-exit", () => setChat(s => ({ ...s, connected: false, running: false }))),
    ];
    return () => { ps.forEach(p => p.then(f => f())); };
  }, []);

  useEffect(() => { if (chat.model) setModelDraft(chat.model); }, [chat.model]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chat.messages]);

  const send = (obj: object) => invoke("send_message", { message: JSON.stringify(obj) }).catch(err => setChat(s => addError(s, String(err))));

  async function openFolder() {
    const dir = await open({ directory: true, multiple: false });
    if (typeof dir !== "string") return;
    setFolder(dir);
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

  function decide(id: string, decision: "approve" | "decline" | "always") {
    send({ type: "approval_response", id, decision });
    setChat(s => removeApproval(s, id));
  }

  const lastAssistant = chat.messages.map(m => m.role).lastIndexOf("assistant");

  return (
    <div className="flex h-full flex-col bg-mx-bg text-mx-mid">
      <header className="flex items-center gap-3 border-b border-mx-dim px-4 py-2">
        <span className="font-semibold glow">localforge</span>
        <span className={"h-2 w-2 rounded-full " + (chat.connected ? "bg-mx-bright" : "bg-mx-dim")} title={chat.connected ? "Connected" : "Not connected"} />
        <span className="min-w-0 flex-1 truncate text-sm text-mx-dim" title={folder ?? ""}>{folder ?? "No folder"}</span>
        <ModelPicker value={modelDraft} disabled={chat.running} onChange={m => { setModelDraft(m); if (chat.connected && !chat.running && m && m !== chat.model) send({ type: "set_model", model: m }); }} />
        <label className="flex items-center gap-1 text-sm"><input type="checkbox" checked={chat.autoApprove} disabled={!chat.connected} onChange={e => send({ type: "set_auto", enabled: e.target.checked })} />Auto-approve</label>
        <button className="rounded-sm border border-mx-dim bg-mx-panel2 px-3 py-1 text-sm text-mx-green hover:border-mx-mid hover:text-mx-bright" onClick={openFolder}>Open folder…</button>
      </header>
      {chat.status && <div className="border-b border-mx-dim px-4 py-1 text-xs text-mx-dim">{chat.status}</div>}
      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-3">
            {!folder ? <div className="flex h-full items-center justify-center text-mx-dim">Open a folder to start.</div> : chat.messages.map((m, i) => <MessageView key={i} message={m} thinking={chat.running && i === lastAssistant} />)}
            <div ref={endRef} />
          </div>
          <ApprovalPanel approvals={chat.approvals} onDecide={decide} />
          <div className="flex gap-2 border-t border-mx-dim p-3">
            <textarea rows={3} className="flex-1 resize-none rounded border border-mx-dim bg-mx-panel2 p-2 text-sm" placeholder={chat.connected ? "Ask localforge…" : "Open a folder to start"} value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
            {chat.running ? <button className="self-end rounded-sm border border-mx-red bg-transparent px-4 py-2 text-sm text-mx-red hover:border-mx-bright hover:text-mx-bright" onClick={() => send({ type: "cancel" })}>Stop</button> : <button className="self-end rounded-sm border border-mx-mid bg-transparent px-4 py-2 text-sm text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim" disabled={!chat.connected || !input.trim()} onClick={submit}>Send</button>}
          </div>
        </main>
        <aside className="w-72 shrink-0 space-y-4 overflow-y-auto border-l border-mx-dim bg-mx-panel p-3">
          <UsagePanel usage={chat.usage} model={chat.model} />
          <TodoList todos={chat.todos} />
        </aside>
      </div>
      <StatusBar folder={folder} connected={chat.connected} running={chat.running} model={chat.model} autoApprove={chat.autoApprove} todoCount={chat.todos.length} doneCount={chat.todos.filter(t => t.status === "completed").length} onCancel={() => send({ type: "cancel" })} />
    </div>
  );
}

export default App;
