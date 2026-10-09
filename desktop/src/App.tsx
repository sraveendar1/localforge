import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { configuredPaidModels } from "./paidModels";
import { BackendStatus } from "./BackendStatus";
import { SetupScreen } from "./SetupScreen";
import { asksUser } from "./needsInput";
import { ModelWizard } from "./ModelWizard";
import { RightPanel } from "./RightPanel";
import { defaultRightOpen, loadPanelOpen, loadPanelWidth, savePanelOpen, savePanelWidth } from "./panelPrefs";
import type { PanelLayout } from "./PanelResize";
import { ApprovalPanel, MessageView, TodoList } from "./components";
import { UsagePanel } from "./UsagePanel";
import { StatusBar } from "./StatusBar";
import { ForgingIndicator } from "./ForgingIndicator";
import { GoalPanel } from "./GoalPanel";
import { ActiveModelsPanel } from "./ActiveModelsPanel";
import { addError, addUserMessage, applyEvent, initialState, removeApproval, appendBackendLog, backendExited, setWizard } from "./state";
import { SystemPanel } from "./SystemPanel";
import { SlashMenu, matchSlashCommands } from "./SlashMenu";
import { Curtain } from "./Curtain";
import { LeftNav } from "./LeftNav";
import { GettingStarted } from "./GettingStarted";
import { addRecentFolder, loadRecentFolders } from "./recentFolders";
import type { ChatState } from "./state";
import "./App.css";

function App() {
  const [chat, setChat] = useState<ChatState>(initialState);
  const [folder, setFolder] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [slashActive, setSlashActive] = useState(0);
  const [leftNavOpen, setLeftNavOpen] = useState(() => loadPanelOpen("left", false));
  const [rightOpen, setRightOpen] = useState(() => loadPanelOpen("right", defaultRightOpen()));
  const [leftWidth, setLeftWidth] = useState(() => loadPanelWidth("left"));
  const [rightWidth, setRightWidth] = useState(() => loadPanelWidth("right"));
  const [expanded, setExpanded] = useState<"left" | "right" | null>(null);  // a side panel filling the whole window
  const [setupSkipped, setSetupSkipped] = useState(false);  // "Skip for now" on the first-run setup screen
  const [manageOpen, setManageOpen] = useState(false);  // Models > Accounts & keys: the same screen, opened on purpose
  const [wizardStart, setWizardStart] = useState<{ key: number; step: number; row: string | null }>({ key: 0, step: 0, row: null });  // where the model screen opens (a fresh key remounts it)
  const [wizardEdit, setWizardEdit] = useState(false);  // the model screen was opened on purpose ("Change models"), not shown to a new project
  const [recentFolders, setRecentFolders] = useState<string[]>(() => loadRecentFolders());
  const [attachedImage, setAttachedImage] = useState<{ dataUrl: string; mimeType: string; data: string } | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const slashMatches = matchSlashCommands(input.trim());

  useEffect(() => {
    const ps = [
      listen<string>("localforge-event", e => {
        let ev: any;
        try { ev = JSON.parse(e.payload); } catch { return; }
        setChat(s => applyEvent(s, ev));
      }),
      listen("localforge-exit", () => setChat(s => backendExited(s))),
      // Whatever the backend prints to stderr: normally nothing, but a start that
      // fails says why here (a crash, a dyld error) and is shown by BackendStatus.
      listen<string>("localforge-stderr", e => setChat(s => appendBackendLog(s, String(e.payload ?? "")))),
    ];
    return () => { ps.forEach(p => p.then(f => f())); };
  }, []);


  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chat.messages]);

  // `useCallback` with no deps keeps this reference stable across renders --
  // without it, this was a new function every render, and since it's a
  // dependency of the two effects below, every state update (even a single
  // streamed token) re-fired them: get_state/memory_list/scratch_list/
  // queue_list sent again on every render, and the system_stats interval
  // torn down and recreated on every render too. That's an unbounded
  // feedback loop -- each response triggers a re-render, which re-sends the
  // requests, which triggers more responses -- and a very plausible reason
  // the app felt slow regardless of the machine it ran on.
  const send = useCallback(
    (obj: object) => invoke("send_message", { message: JSON.stringify(obj) }).catch(err => setChat(s => addError(s, String(err)))),
    []
  );

  async function startSessionForFolder(dir: string) {
    setFolder(dir);
    setWizardEdit(false);
    setChat({ ...initialState, backendStartedAt: Date.now() });
    setRecentFolders(r => addRecentFolder(dir, r));
    try { // No model is passed: the backend uses this project's saved models
      // (.localforge/models.json), else the user's defaults. Carrying the
      // last folder's model over would override the next folder's own.
      await invoke("start_session", { folder: dir, model: null }); } catch (err) { setChat(s => addError(s, String(err))); }
  }

  // Set when the app is launched with a folder to open directly (e.g.
  // `localforge desktop`'s hand-off from a terminal session -- see cli.py's
  // desktop_command and lib.rs's get_initial_folder). Runs once on mount,
  // after startSessionForFolder exists (function declarations hoist within
  // the component body, so this is safe regardless of source order).
  useEffect(() => {
    invoke<string | null>("get_initial_folder").then(initial => {
      if (initial) startSessionForFolder(initial);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function openFolder() {
    const dir = await open({ directory: true, multiple: false });
    if (typeof dir !== "string") return;
    await startSessionForFolder(dir);
  }

  function pickImage() {
    fileInputRef.current?.click();
  }

  // One image can ride along with a message. It comes from the 📎 button, from pasting
  // (Cmd/Ctrl+V with an image on the clipboard, e.g. a screenshot) or from dropping a file
  // onto the message box.
  function attachImageFile(file: File | null | undefined): boolean {
    if (!file || !file.type.startsWith("image/")) return false;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result ?? "");
      // "data:image/png;base64,AAAA..." -> mime type + bare base64.
      const match = dataUrl.match(/^data:([^;]+);base64,(.*)$/s);
      if (!match) return;
      setAttachedImage({ dataUrl, mimeType: match[1], data: match[2] });
    };
    reader.readAsDataURL(file);
    return true;
  }

  function onImageSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow picking the same file again later
    attachImageFile(file);
  }

  function imageFrom(data: DataTransfer | null): File | null {
    if (!data) return null;
    for (const item of Array.from(data.items ?? [])) {
      if (item.kind === "file" && item.type.startsWith("image/")) return item.getAsFile();
    }
    return Array.from(data.files ?? []).find(f => f.type.startsWith("image/")) ?? null;
  }

  function onPaste(e: React.ClipboardEvent<HTMLTextAreaElement>) {
    const file = imageFrom(e.clipboardData);
    if (file && attachImageFile(file)) e.preventDefault();  // an image paste isn't text; ordinary text pastes untouched
  }

  function onDrop(e: React.DragEvent) {
    const file = imageFrom(e.dataTransfer);
    if (file) { e.preventDefault(); attachImageFile(file); }
  }

  function submit() {
    const text = input.trim();
    if (!text || !chat.connected || chat.trustRequired) return;
    setChat(s => addUserMessage(s, text, attachedImage?.dataUrl));
    const image = attachedImage ? { mime_type: attachedImage.mimeType, data: attachedImage.data } : undefined;
    setInput("");
    setAttachedImage(null);
    send({ type: "user_message", text, ...(image ? { image } : {}) });
  }

  function decide(id: string, decision: "approve" | "decline" | "always") {
    send({ type: "approval_response", id, decision });
    setChat(s => removeApproval(s, id));
  }

  // Every paid model in play and its role(s): the right-hand panels list these.
  // The chevron tabs toggle a side panel and remember the choice. Only a click is
  // remembered: with no choice yet the right panel just starts closed on a narrow window.
  const toggleLeft = () => { const next = !leftNavOpen; setLeftNavOpen(next); savePanelOpen("left", next); };
  const toggleRight = () => { const next = !rightOpen; setRightOpen(next); savePanelOpen("right", next); };
  const sideLayout = (name: "left" | "right"): PanelLayout => ({
    width: name === "left" ? leftWidth : rightWidth,
    expanded: expanded === name,
    onWidth: (w, commit) => { (name === "left" ? setLeftWidth : setRightWidth)(w); if (commit) savePanelWidth(name, w); },
    onToggleExpand: () => {
      if (expanded === name) { setExpanded(null); return; }
      (name === "left" ? setLeftNavOpen : setRightOpen)(true);  // expanding shows it, whatever it was
      setExpanded(name);
    },
  });
  // Esc puts the window back; so does opening the model screen, which needs the centre.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);
  const needsSetup = !!chat.setup?.needsSetup;
  // The guided model screen takes the centre of the window: automatically for a project
  // with no saved models (right after the trust prompt), and whenever "Change models" is
  // clicked. A project that already has its models but whose orchestrator can't run (a
  // key was removed) gets the plainer sign-in screen instead.
  const wizardActive = chat.wizard === "active" && !chat.trustRequired && !!folder;
  const inlineSetup = !!chat.setup && needsSetup && !setupSkipped && chat.messages.length === 0 && chat.wizard === "done";
  const overlaySetup = !!chat.setup && manageOpen;
  const openWizard = (step = 0, row: string | null = null) => {
    setWizardEdit(true);
    setWizardStart(w => ({ key: w.key + 1, step, row }));
    setChat(s => setWizard(s, "active"));
  };
  const closeWizard = () => setChat(s => setWizard(s, "done"));
  const finishWizard = () => { send({ type: "finish_project_setup" }); closeWizard(); setTimeout(() => inputRef.current?.focus(), 0); };
  // Once something works, forget having skipped setup: if it stops working later, that
  // should start from the banner.
  useEffect(() => { if (!needsSetup) setSetupSkipped(false); }, [needsSetup]);
  // (the orchestrator isn't listed as "in use" while it can't run: nothing is set up)
  const configuredPaid = configuredPaidModels(needsSetup ? "" : chat.model, chat.orchestratorOptions, chat.localModelTargets);

  const lastAssistant = chat.messages.map(m => m.role).lastIndexOf("assistant");
  // The orchestrator's latest reply is asking you something: shown in orange, and the box invites an answer.
  let lastReply = -1;  // the last message that isn't a summary or system note
  for (let i = chat.messages.length - 1; i >= 0; i--) if (chat.messages[i].role !== "summary" && chat.messages[i].role !== "system") { lastReply = i; break; }
  const awaitingInput = !chat.running && lastAssistant >= 0 && lastAssistant === lastReply && asksUser(chat.messages[lastAssistant].text);

  useEffect(() => {
    // Held back until any trust prompt is resolved -- the backend no-ops
    // everything except trust_response until then, so there's nothing yet
    // for these to usefully fetch, and firing them again once trust clears
    // (the dependency below) is what actually populates the sidebar.
    if (chat.connected && !chat.trustRequired) {
      send({ type: "get_state" });
      send({ type: "memory_list" });  // still needed: feeds GoalPanel's goal/narrative
      send({ type: "queue_list" });  // Request the queue list on connect
      send({ type: "system_stats" });  // don't wait for the first 5s interval tick
      send({ type: "usage_request" });  // historical usage (previous session, all-time)
      send({ type: "configured_models_request" });  // best-fit local model per modality
      send({ type: "orchestrator_options_request" });  // the header dropdown: installed + keyed/logged-in models
      send({ type: "setup_status_request" });
      send({ type: "budget_request" });  // this month's paid image spending and the limit  // is an orchestrator set up at all? (first-run screen)
      send({ type: "advanced_model_request" });  // per-modality delegate target (auto, or a pinned override)
    }
  }, [chat.connected, chat.trustRequired, send]);

  useEffect(() => {
    if (chat.connected) {
      const interval = setInterval(() => {
        send({ type: "system_stats" });
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [chat.connected, send]);

  return (
    <div className="flex h-full flex-col bg-mx-bg text-mx-mid">
      <header className="flex items-center gap-3 border-b border-mx-dim px-4 py-2">
        <span className="font-semibold glow">localforge</span>
        <span className={"h-2 w-2 rounded-full " + (chat.connected ? "bg-mx-bright" : "bg-mx-dim")} title={chat.connected ? "Connected" : "Not connected"} />
        <span className="min-w-0 flex-1 truncate text-sm text-mx-dim" title={folder ?? ""}>{folder ?? "No folder"}</span>
        <button
          type="button"
          disabled={!chat.connected}
          title={needsSetup ? "No model is set up yet" : "Change models"}
          aria-label="Orchestrator model (click to change models)"
          onClick={() => {
            openWizard(0);  // changing models happens in the centre of the window
          }}
          className="max-w-xs truncate rounded-sm border border-mx-dim bg-mx-panel2 px-2 py-1 text-sm text-mx-bright hover:border-mx-mid disabled:opacity-50"
        >
          {needsSetup ? "Set up models…" : chat.model || "model"}
        </button>
        <label className="flex items-center gap-1 text-sm"><input type="checkbox" checked={chat.autoApprove} disabled={!chat.connected} onChange={e => send({ type: "set_auto", enabled: e.target.checked })} />Auto-approve</label>
        <button className="rounded-sm border border-mx-dim bg-mx-panel2 px-3 py-1 text-sm text-mx-green hover:border-mx-mid hover:text-mx-bright" onClick={openFolder}>Open folder…</button>
      </header>
      {chat.status && <div className="border-b border-mx-dim px-4 py-1 text-xs text-mx-dim">{chat.status}</div>}
      {folder && !chat.connected && chat.backendExited && !chat.trustDeclined && chat.messages.length > 0 && (
        <div className="flex items-center gap-3 border-b border-mx-red bg-mx-panel px-4 py-1 text-xs text-mx-red" role="alert" data-testid="backend-lost">
          <span>The localforge backend stopped, so nothing more will run.{chat.backendLog.length ? ` Last message: ${chat.backendLog[chat.backendLog.length - 1].slice(0, 160)}` : ""}</span>
          <button type="button" className="rounded-sm border border-mx-red px-2 py-0.5 hover:text-mx-bright" onClick={() => startSessionForFolder(folder)}>Restart</button>
        </div>
      )}
      {folder && needsSetup && !inlineSetup && !overlaySetup && !wizardActive && chat.wizard === "done" && !chat.trustRequired && (
        <div className="flex items-center gap-3 border-b border-mx-amber bg-mx-panel px-4 py-1 text-xs text-mx-amber" role="alert" data-testid="setup-banner">
          <span>No model is set up yet, so tasks will fail{chat.setup?.orchestrator.reason ? `: ${chat.setup.orchestrator.reason}` : ""}.</span>
          <button type="button" className="rounded-sm border border-mx-amber px-2 py-0.5 hover:text-mx-bright" onClick={() => openWizard(0)}>Set up</button>
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        {expanded !== "right" && <LeftNav
          layout={sideLayout("left")}
          open={leftNavOpen}
          onToggle={toggleLeft}
          recentFolders={recentFolders}
          currentFolder={folder}
          onSelectFolder={startSessionForFolder}
          onOpenDialog={openFolder}
          frontier={chat.model}
          targets={chat.localModelTargets}
          connected={chat.connected}
          source={chat.modelsSource}
          onChange={which => (which && which !== "frontier" ? openWizard(1, which) : openWizard(0))}
          onManageAccounts={() => { setManageOpen(true); send({ type: "setup_status_request" }); }}
        />}
        <main className={"min-w-0 flex-1 flex-col " + (expanded ? "hidden" : "flex")}>
          <div className="flex-1 overflow-y-auto px-4 py-3">
            {!folder ? <div className="flex h-full items-center justify-center text-mx-dim">Open a folder to start.</div> : !chat.connected && !chat.trustRequired && chat.messages.length === 0 ? (
              <BackendStatus
                folder={folder}
                log={chat.backendLog}
                exited={chat.backendExited}
                startedAt={chat.backendStartedAt}
                trustDeclined={chat.trustDeclined}
                onRetry={() => startSessionForFolder(folder)}
              />
            ) : chat.trustRequired ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
                <h1 className="text-sm font-semibold uppercase tracking-widest text-mx-amber glow">Trust this folder?</h1>
                <p className="max-w-md font-mono text-xs text-mx-dim">{chat.trustRequired}</p>
                <p className="max-w-md text-xs text-mx-mid">
                  localforge hasn't run here before. Trusting it lets the orchestrator read, create, change, move
                  and delete files in this folder and run commands in it -- each change still asks for approval
                  first (or Auto-approve) unless you decline here.
                </p>
                <div className="flex gap-2 pt-2">
                  <button
                    className="rounded-sm border border-mx-mid bg-transparent px-4 py-2 text-sm text-mx-green hover:border-mx-bright hover:text-mx-bright"
                    onClick={() => send({ type: "trust_response", trust: true })}
                  >
                    Trust this folder
                  </button>
                  <button
                    className="rounded-sm border border-mx-red bg-transparent px-4 py-2 text-sm text-mx-red hover:border-mx-bright hover:text-mx-bright"
                    onClick={() => send({ type: "trust_response", trust: false })}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : wizardActive ? (
              <ModelWizard key={wizardStart.key} chat={chat} send={send} mode={wizardEdit ? "edit" : "new"} startStep={wizardStart.step} startRow={wizardStart.row} onClose={closeWizard} onFinish={finishWizard} />
            ) : inlineSetup && chat.setup ? (
              <SetupScreen status={chat.setup} result={chat.setupResult} connected={chat.connected} send={send} onSkip={() => setSetupSkipped(true)} />
            ) : chat.messages.length === 0 ? (
              <GettingStarted
                folderName={(folder ?? "").split("/").filter(Boolean).pop() ?? "this folder"}
                checks={chat.setupChecks}
                connected={chat.connected && !chat.trustRequired}
                onRequestChecks={() => send({ type: "setup_checks_request" })}
                onFix={which => openWizard(which === "planner" ? 0 : 1)}
                onExample={text => { setInput(text); inputRef.current?.focus(); }}
                onOpenWizard={() => openWizard(0)}
              />
            ) : chat.messages.map((m, i) => <MessageView key={i} message={m} thinking={chat.running && i === lastAssistant} awaitingInput={awaitingInput && i === lastAssistant} onPlan={i === lastAssistant && chat.planPending ? action => send({ type: "plan_response", action }) : undefined} />)}
            <div ref={endRef} />
          </div>
          {chat.running && (
            <ForgingIndicator activity={chat.activity} waitingFor={chat.approvals[0]?.title} queued={chat.queue.length} />
          )}
          {chat.queue.length > 0 && (
            <div className="border-b border-mx-dim px-3 py-1 bg-mx-panel2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-semibold">Queued: {chat.queue.length}</span>
                <button className="rounded-sm border border-mx-red bg-transparent px-2 py-0.5 text-xs text-mx-red hover:border-mx-bright hover:text-mx-bright" onClick={() => send({ type: "queue_clear" })}>Clear</button>
              </div>
              {chat.queue.map((q, i) => (
                <div key={i} className="flex items-center gap-2 text-mx-mid">
                  <span className="min-w-0 flex-1 truncate">{q}</span>
                  {chat.running && <button className="shrink-0 rounded-sm border border-mx-dim px-1.5 text-[11px] text-mx-mid hover:border-mx-bright hover:text-mx-bright" title="Hand this to the task that's running now" onClick={() => send({ type: "queue_to_note", index: i })}>Send to running task</button>}
                </div>
              ))}
            </div>
          )}
          <ApprovalPanel approvals={chat.approvals} onDecide={decide} />
          <SlashMenu matches={slashMatches} activeIndex={slashActive} onPick={name => { setInput(name + " "); setSlashActive(0); }} />
          {attachedImage && (
            <div className="mx-3 mt-2 flex items-center gap-2 rounded-sm border border-mx-dim bg-mx-panel2 p-2">
              <img src={attachedImage.dataUrl} alt="attached" className="h-14 w-14 rounded-sm border border-mx-dim object-cover" />
              <span className="text-xs text-mx-dim">Image attached</span>
              <button
                type="button"
                className="ml-auto rounded-sm border border-mx-red px-2 py-0.5 text-xs text-mx-red hover:border-mx-bright hover:text-mx-bright"
                onClick={() => setAttachedImage(null)}
              >
                Remove
              </button>
            </div>
          )}
          <div className="flex gap-2 border-t border-mx-dim p-3">
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={onImageSelected} />
            <button
              type="button"
              title="Attach an image (or paste or drop one into the box). Needs a vision-capable orchestrator."
              disabled={!chat.connected || !!chat.trustRequired}
              onClick={pickImage}
              className="self-end rounded-sm border border-mx-dim bg-mx-panel2 px-3 py-2 text-sm text-mx-mid hover:border-mx-mid hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim"
            >
              📎
            </button>
            <textarea
              ref={inputRef}
              rows={3}
              className={"flex-1 resize-none rounded border bg-mx-panel2 p-2 text-sm " + (awaitingInput || chat.planPending ? "border-mx-amber" : "border-mx-dim")}
              placeholder={chat.trustRequired ? "Trust this folder above to start" : chat.planPending ? "Approve the plan, or say what to change…" : awaitingInput ? "Answer localforge's question…" : chat.connected ? (chat.messages.length === 0 ? "Describe what you want to build or change…  (type / for commands; paste or drop an image)" : "Ask localforge, or type / for commands. Paste or drop an image to attach it…") : folder ? (chat.backendExited ? "localforge isn't running" : "Waiting for localforge to start…") : "Open a folder to start"}
              value={input}
              onChange={e => { setInput(e.target.value); setSlashActive(0); }}
              onPaste={onPaste}
              onDrop={onDrop}
              onDragOver={e => { if (Array.from(e.dataTransfer.items ?? []).some(i => i.kind === "file")) e.preventDefault(); }}
              onKeyDown={e => {
                if (slashMatches.length > 0 && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
                  e.preventDefault();
                  const dir = e.key === "ArrowDown" ? 1 : -1;
                  setSlashActive(i => (i + dir + slashMatches.length) % slashMatches.length);
                  return;
                }
                if (slashMatches.length > 0 && e.key === "Tab") {
                  e.preventDefault();
                  setInput(slashMatches[slashActive].name + " ");
                  return;
                }
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  const exact = slashMatches.some(m => m.name === input.trim());
                  if (slashMatches.length > 0 && !exact) {
                    setInput(slashMatches[slashActive].name + " ");
                    return;
                  }
                  submit();
                }
              }}
            />
            {chat.running && <button className="self-end rounded-sm border border-mx-red bg-transparent px-4 py-2 text-sm text-mx-red hover:border-mx-bright hover:text-mx-bright" onClick={() => send({ type: "cancel" })}>Stop</button>}
            <button className="self-end rounded-sm border border-mx-mid bg-transparent px-4 py-2 text-sm text-mx-green hover:border-mx-bright hover:text-mx-bright disabled:border-mx-dim disabled:text-mx-dim" disabled={!chat.connected || !!chat.trustRequired || !input.trim()} onClick={submit}>Send</button>
          </div>
        </main>
        {/* Requested order: Overall goal, pending task (plan), active LLMs, usage and cost. */}
        {expanded !== "left" && <RightPanel open={rightOpen} onToggle={toggleRight} layout={sideLayout("right")}>
          <Curtain title="Overall goal"><GoalPanel memory={chat.memory} onItem={(action, index) => send({ type: "memory_item", action, index })} /></Curtain>
          <Curtain title="Plan"><TodoList todos={chat.todos} queued={chat.queue} /></Curtain>
          <Curtain title={chat.queue.length ? `Queue (${chat.queue.length})` : "Queue"}>
            <div data-testid="queue-panel">
              {chat.queue.length === 0 ? (
                <p className="text-xs text-mx-dim italic">Nothing waiting. What you send while a task runs lines up here and starts when it finishes.</p>
              ) : (
                <>
                  <ol className="space-y-1 text-xs">
                    {chat.queue.map((q, i) => (
                      <li key={i} className="flex gap-2 text-mx-mid"><span className="text-mx-dim" aria-hidden>{i + 1}.</span><span className="min-w-0 flex-1 break-words">{q}</span>{chat.running && <button className="shrink-0 rounded-sm border border-mx-dim px-1.5 text-[11px] text-mx-mid hover:border-mx-bright hover:text-mx-bright" onClick={() => send({ type: "queue_to_note", index: i })}>Send now</button>}</li>
                    ))}
                  </ol>
                  <button className="mt-2 rounded-sm border border-mx-red bg-transparent px-2 py-0.5 text-xs text-mx-red hover:border-mx-bright hover:text-mx-bright" onClick={() => send({ type: "queue_clear" })}>Clear queue</button>
                </>
              )}
            </div>
          </Curtain>
          <Curtain title="Active LLMs">
            <ActiveModelsPanel
              usage={chat.usage}
              configuredModels={chat.configuredModels}
              localModelTargets={chat.localModelTargets}
              configuredPaid={configuredPaid}
              orchestrator={chat.model}
            />
          </Curtain>
          <Curtain title="Usage and cost"><UsagePanel usage={chat.usage} configuredPaid={configuredPaid} usageHistory={chat.usageHistory} /></Curtain>
          <Curtain title="System"><SystemPanel stats={chat.systemStats} /></Curtain>
        </RightPanel>}
      </div>
      {overlaySetup && chat.setup && (
        <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/80 p-6" role="dialog" aria-modal="true" aria-label="Accounts and keys" data-testid="setup-overlay">
          <div className="w-full max-w-xl rounded-sm border border-mx-mid bg-mx-bg">
            <SetupScreen status={chat.setup} result={chat.setupResult} connected={chat.connected} send={send} onSkip={() => setManageOpen(false)} skipLabel="Close" mode="manage" />
          </div>
        </div>
      )}
      <StatusBar folder={folder} connected={chat.connected} running={chat.running} model={chat.model} autoApprove={chat.autoApprove} todoCount={chat.todos.length} doneCount={chat.todos.filter(t => t.status === "completed").length} onCancel={() => send({ type: "cancel" })} />
    </div>
  );
}

export default App;
