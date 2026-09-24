# Desktop Parity Plan

The localforge desktop app is native-first, with buttons, menus, and panels being the primary way to interact with the app. An embedded CLI view exists but is intended for advanced users or exceptional cases; normal users should never need it.

## CLI Subcommands -> Desktop

| CLI Command | What it does | Proposed Native UI Control/Panel | Protocol Support Today |
|---|---|---|---|
| scan | Detects hardware | System Info Panel | MISSING - needs new request type `hardware_info` |
| doctor | Diagnoses problems | Doctor Panel (embedded webview) | MISSING - needs new request type `diagnose` |
| setup | Initial setup walkthrough | Setup Wizard | MISSING - needs new request types `setup_step`, `setup_complete` |
| models | Lists installed models | Model Selection Panel | MISSING - needs new request type `list_models` |
| catalog | Lists available models | Catalog Browser | MISSING - needs new request types `list_catalog`, `catalog_details` |
| run | Runs a session | Session Panel (embedded React app) | `user_message`, `run_started`, `run_finished`, `error` |
| usage | Shows usage stats | Usage Sidebar (already drafted) | `ready`, `settings`, `frontier_round` |
| installed | Lists installed models | Model Selection Panel | `list_models` |
| delete | Deletes a model | Confirmation Dialog | MISSING - needs new request type `delete_model` |
| uninstall | Uninstalls localforge | Confirmation Dialog | MISSING - needs new request type `uninstall` |
| theme | Changes theme | Theme Selector | MISSING - needs new request type `set_theme` |
| help | Shows help | Help Sidebar | MISSING - needs new request type `help` |
| /clear | Clears the console | Clear Button | `clear` |
| /compact | Makes the console compact | Toggle Button | `compact` |
| /auto | Toggles auto-approval | Toggle Button | `set_auto` |
| /model | Sets the model | Model Selector | `set_model` |
| /memory | Shows memory facts, forget/clear | Memory Panel (DONE - MemoryPanel.tsx) | `memory_list`, `memory_forget`, `memory_clear` -> `memory` event |
| /scratch | Shows scratchpad files, clear | Scratch section of Memory Panel (DONE) | `scratch_list`, `scratch_clear` -> `scratch` event |
| /summary | Shows session summary | Session Summary Panel | MISSING - needs new request type `session_summary` |
| /queue | Shows task queue | Task Queue Panel | MISSING - needs new request type `task_queue` |
| /stop | Stops the current task | Stop Button | `cancel` |
| /tell | Streams model tokens | Model Token Stream | `delegate_token` |
| /stream | Toggles streaming | Toggle Button | MISSING - needs new request type `toggle_stream` |
| /why | Explains why a model was chosen | Why Panel | MISSING - needs new request type `why_explanation` |
| /usage | Shows usage stats | Usage Sidebar | `ready`, `settings`, `frontier_round` |

## Protocol Gaps

- `hardware_info`: Fields - `os`, `arch`, `cpu_cores`, `ram_gb`, `free_disk_gb`, `gpus` (list of `name`, `vram_gb`, `backend`)
- `diagnose`: Fields - `issues` (list of `message`)
- `setup_step`: Fields - `step`, `message`
- `setup_complete`: Fields - None
- `list_models`: Fields - `models` (list of `name`, `path`, `installed`)
- `catalog_details`: Fields - `model`, `details` (JSON object)
- `delete_model`: Fields - `name`
- `uninstall`: Fields - None
- `set_theme`: Fields - `theme`
- `help`: Fields - `commands` (list of `name`, `description`)
- ~~`memory_stats`~~ / ~~`scratchpad_state`~~: done, as `memory_list`/`memory_forget`/`memory_clear` and `scratch_list`/`scratch_clear` requests with `memory` (`facts`, `narrative`) and `scratch` (`files`: `path`, `size`) events.
- `session_summary`: Fields - `text`
- `task_queue`: Fields - `tasks` (list of `id`, `type`, `status`, `progress`)
- `toggle_stream`: Fields - `enabled`
- `why_explanation`: Fields - `reason`

## Done

- `state.ts` handles the `memory`, `scratch` and `session_reset` events (types `MemoryFact`, `MemoryState`, `ScratchFile`; state fields `memory`, `scratchFiles`). `session_reset` clears messages/approvals/todos/status while keeping connection, model, auto-approve, usage and system stats.
- `MemoryPanel.tsx` renders the Memory facts (per-fact `forget`, plus `clear`) and Scratch files (sizes, plus `clear`); App.tsx renders it in the right-hand aside between UsagePanel and TodoList and sends the real request types.
- `serve.py` now emits a `system_stats` event with a nested `hardware` object (`os`, `arch`, `cpu_cores`, `ram_gb`, `free_disk_gb`, `gpus`) plus `cpu_percent`, `ram_used_gb`, `ram_total_gb`, matching SystemPanel.tsx.
- Verified: 596 python tests pass, `npm run build` (tsc + vite) green, `cargo check` green.

## Current Desktop UI State

- App.tsx (already drafted)
- UsagePanel.tsx (already drafted)
- SystemPanel.tsx, MemoryPanel.tsx (wired into App.tsx)
- Components folder (e.g., Button.tsx, Input.tsx, etc.)

## Build Order

1. **Usage Sidebar** (UsagePanel already drafted)
2. **Approvals**
   - Create ApprovalRequest event type
   - Implement approval request handling in desktop UI
   - Update orchestrator to emit approval_request events
3. **Session Controls**
   - Implement /clear, /compact, /auto, /model commands
   - Update orchestrator to accept and process these commands
4. **Setup/Doctor/Models Screens**
   - Create Setup Wizard flow (embedded CLI view)
   - Implement doctor functionality (embedded webview)
   - Create Model Selection Panel (lists installed and catalog models)
