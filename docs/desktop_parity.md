# Desktop Parity Plan

The localforge desktop app is native-first, with buttons, menus, and panels being the primary way to interact with the app. An embedded CLI view exists but is intended for advanced users or exceptional cases; normal users should never need it.

## Recent Changes

- **2026-09-25**: Updated the parity document to reflect the current state of the desktop app and backend. Added support for the `settings` event for both `set_auto` and `set_model` commands. Updated the `system_stats` request to return a nested `hardware` object plus live CPU, RAM usage stats. Invalid approval responses now surface as `error` events. Message queueing is implemented, with messages being queued rather than rejected while a run is in progress. The `queue` event carries the current items.

## CLI Subcommands -> Desktop

| CLI Command | What it does | Proposed Native UI Control/Panel | Protocol Support Today |
|---|---|---|---|
| scan | Detects hardware | System Info Panel | ✅ |
| doctor | Diagnoses problems | Doctor Panel (embedded webview) | ✅ |
| setup | Initial setup walkthrough | Setup Wizard | ❌ (Not done) |
| models | Lists installed models | Model Selection Panel | ✅ |
| catalog | Lists available models | Catalog Browser | ✅ |
| run | Runs a session | Session Panel (embedded React app) | ✅ |
| usage | Shows usage stats | Usage Sidebar | ✅ |
| installed | Lists installed models | Model Selection Panel | ✅ |
| delete | Deletes a model | Confirmation Dialog | ❌ (Not done) |
| uninstall | Uninstalls localforge | Confirmation Dialog | ❌ (Not done) |
| theme | Changes theme | Theme Selector | ✅ |
| help | Shows help | Help Sidebar | ❌ (Not done) |
| /clear | Clears the console | Clear Button | ✅ |
| /compact | Makes the console compact | Toggle Button | ❌ (No GUI affordance yet) |
| /auto | Toggles auto-approval | Toggle Button | ✅ |
| /model | Sets the model | Model Selector | ✅ |
| /memory | Shows memory facts, forget/clear | Memory Panel | ✅ |
| /scratch | Shows scratchpad files, clear | Scratch section of Memory Panel | ✅ |
| /summary | Shows session summary | Session Summary Panel | ❌ (Not done) |
| /queue | Shows task queue | Queued-messages strip in App.tsx (queue_list / queue_clear) | ✅ |
| /stop | Stops the current task | Stop Button | ✅ |
| /tell | Streams model tokens | Model Token Stream | ❌ (Not done) |
| /stream | Toggles streaming | Toggle Button | ❌ (Not done) |
| /why | Explains why a model was chosen | Why Panel | ❌ (Not done) |
| /usage | Shows usage stats | Usage Sidebar | ✅ |

## Protocol Gaps

- **Full Slash-Command Coverage**: Not yet implemented in the GUI chat input.
- **y/n/a Approval Affordances**: Not yet implemented in the UI (keyboard shortcuts and always-allow).
- **Background Tasks and Status Line**: Not yet implemented.
- **/tell (injecting a note mid-run)**: Not yet implemented.
- **/why (explaining model choice)**: Not yet implemented.
- **Message Queueing**: Done — the backend queues messages sent during a run and emits a `queue` event; `state.ts` tracks `chat.queue` and `App.tsx` renders the queued-message strip with a Clear action.

## Done

- **2026-09-24**: Updated `state.ts` to handle `memory`, `scratch`, and `session_reset` events. Added `MemoryFact`, `MemoryState`, and `ScratchFile` types, and updated state fields `memory`, `scratchFiles`. Implemented `session_reset` to clear messages, approvals, todos, status while keeping connection, model, auto-approve, usage, and system stats.
- **2026-09-24**: Created `MemoryPanel.tsx` to render Memory facts (per-fact `forget`, plus `clear`) and Scratch files (sizes, plus `clear`). Integrated it into `App.tsx` in the right-hand aside between `UsagePanel` and `TodoList`.
- **2026-09-24**: Updated `serve.py` to emit a `system_stats` event with a nested `hardware` object plus live CPU, RAM usage stats, matching `SystemPanel.tsx`.
- **2026-09-25**: Verified 596 Python tests pass, `npm run build` (tsc + vite) green, `cargo check` green.
