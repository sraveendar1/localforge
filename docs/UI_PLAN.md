# localforge desktop app: UI plan

## Goal
A native desktop application, run locally, that gives localforge a Claude-like chat UI. It is a new front end over the existing Python core, not a rewrite. Decision (2026-09-22): native desktop app, not a browser web app.

## Architecture
- **Shell**: Tauri 2 (<https://tauri.app/>)
  - Small ~10 MB app bundle using the system WebView
  - Native menus, tray icon, notifications, folder picker, auto-update
- **UI**: React + Vite + Tailwind inside the Tauri window
- **Backend**:
  - App does NOT bundle Python
  - Launches `localforge` CLI as a child process with `localforge serve --stdio`
  - Exchanges JSON-lines messages over stdin/stdout
  - No network port, no auth token
- **Process management**:
  - One backend process per project window
  - Each owns its own `Conversation`, `Workspace` and scratchpad
  - Chats are saved to disk for sidebar reopening
- **Repo layout**:
  - Python `serve` command in `src/localforge/`
  - Desktop app in a new top-level `desktop/` folder

## Python core seams used
- `orchestrator.run(task, ..., hooks=, conversation=, workspace=)`
  - Runs one turn; returns `RunResult` with `RunStats`
  - Raises `OrchestrationError` on failure
- `ActivityHooks` (in `tools.py`)
  - Callbacks: `on_frontier`, `on_delegate`, `on_token`, `on_done`, `on_pull`, `on_tool`, `on_tool_result`, `on_todos`, `on_answer_text`
- `Workspace(root, approver=...)`
  - Routes file writes, makes dir, moves, deletes, shell commands through approver callback

## Bridge design (`localforge serve --stdio`)
- **Worker thread**: Each turn runs `run()`; hook callbacks write JSON events to stdout
- **Approval bridge**:
  - Approver emits an `approval_request` event (id, kind, title, detail such as a diff or command), then blocks on a `threading.Event` until the matching `approval_response` (approve/decline/always) arrives on stdin
  - "Always" is remembered per kind for the session, like the REPL
  - Deletions never auto-approve
- **Cancel**: `cancel` message sets flag checked between rounds in `orchestrator.run`
- **Output**: stdout carries protocol JSON; logs go to stderr

## Protocol
**Client to backend**:
- `user_message` {text}
- `approval_response` {id, decision}
- `cancel`
- `set_model` {model}
- `set_auto` {enabled}

**Backend to client**:
- `ready`
- `text_delta` {text}
- `frontier_round` {round}
- `tool_call_started` {name, summary}
- `tool_call_finished` {name, result}
- `delegate_started` {model}
- `delegate_token` {text}
- `delegate_finished` {tool, model, tokens, seconds}
- `model_pull` {model, progress}
- `todos_updated` {todos}
- `approval_request` {id, kind, title, detail}
- `run_finished` {answer, stats}
- `error` {message}

## UI layout
- **Left sidebar**:
  - Project picker
  - Chat list (new, rename, delete)
  - Settings link
- **Main chat**:
  - Streamed markdown with code highlighting
  - Tool cards: collapsible, warnings in amber
- **Approval card**:
  - Inline diff viewer or command text with Approve/Decline/Always buttons
- **Composer**:
  - Multiline input, `/` slash-command menu, Enter to send, Stop button while running
- **Right panel tabs**:
  - Plan (live todo checklist)
  - Files (project tree + read-only viewer)
  - Scratchpad
  - Memory
- **Top bar**:
  - Orchestrator model switcher
  - Auto-approve toggle
  - Usage pill (frontier tokens/cost vs local tokens)
  - Light/dark theme
- **Settings**:
  - API keys
  - Ollama status (doctor)
  - Installed models
  - Hardware scan

## Phases
1. **Backend**: `localforge serve --stdio` with protocol, plus pytest tests for bridge and approver
2. **Tauri MVP**: Folder picker, one chat, streaming text, tool cards, diff approval, Stop button
3. Chats and persistence; multiple windows/projects
4. Panels and top bar
5. Settings screens and themes
6. Packaging: Signed `.app`/.dmg, auto-update, first-run check for `localforge` and Ollama installations

## Prerequisites
- Tauri needs Rust (`rustup`) and Node.js on the developer machine
- Phase 1 needs only existing Python/uv setup
