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
| /memory | Shows memory stats | Memory Sidebar | MISSING - needs new request type `memory_stats` |
| /scratch | Shows scratchpad | Scratchpad Panel | MISSING - needs new request type `scratchpad_state` |
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
- `memory_stats`: Fields - `total`, `used`, `free`
- `scratchpad_state`: Fields - `text`
- `session_summary`: Fields - `text`
- `task_queue`: Fields - `tasks` (list of `id`, `type`, `status`, `progress`)
- `toggle_stream`: Fields - `enabled`
- `why_explanation`: Fields - `reason`

## Current Desktop UI State

- App.tsx (already drafted)
- UsagePanel.tsx (already drafted)
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
