# Desktop Parity Plan

The localforge desktop app is native-first, with buttons, menus, and panels being the primary way to interact with the app. An embedded CLI view exists but is intended for advanced users or exceptional cases; normal users should never need it.

## Recent Changes

- **2026-09-26 (delegate targets)**: Added an "advanced" per-modality override (`delegate_target.py`): coding/docs/general default to automatic (today's behavior, unchanged), but can each independently be pinned to a specific local Ollama model, or sent to a paid cloud model instead -- via an API key (a one-shot LiteLLM call, `backends/cloud.py`'s `CloudApiBackend`) or a CLI subscription login (reusing `cli_transport`'s existing subprocess/JSON-protocol machinery with no tools, `CloudCliBackend`). `Dispatcher.resolve()` checks the configured target before falling back to `catalog.best_match()`; a cloud target returns a synthetic, non-catalog `ModelEntry` (`ModelEntry.provider`) and never goes through the local-download-approval flow. Usage tracking gained a third bucket (`delegate_tokens_generated`/`delegate_cost_usd`/`delegate_notional_cost_usd`) alongside local (free) and frontier spend, shown in `/usage` and the desktop app's Usage panel only when a cloud delegate was actually used. Surfaced two ways: `localforge local-model [coding|docs|general] [value]` (also `/local-model` in the terminal REPL and, newly, in the desktop app's chat input -- serve.py dispatches an explicit command list, unlike the REPL's generic Typer passthrough, so it needed its own `elif` branch), and a "Change" action on each row of `ActiveModelsPanel.tsx` opening a picker (`delegate_options_request`/`set_delegate_target`) that lists the local catalog for that modality plus every cloud provider actually usable right now (has an API key set, or its CLI is on PATH) -- verified end to end in a headless browser against the real backend, including picking a cloud option that a live probe of this sandbox's own `claude` CLI login correctly detected as available.
- **2026-09-26 (later)**: Added `localforge desktop` (also `/desktop` in a session): hands a terminal session off to the desktop app by launching it open to the current folder and ending the terminal session, saving memory through the same path `/exit` uses. `lib.rs` reads a startup folder from `argv[1]` (`get_initial_folder`, filtered to existing directories) and `App.tsx` auto-opens it on mount instead of waiting for "Open folder…" — verified in a headless browser with a seeded initial folder. Also redesigned the sidebar into independently collapsible sections (GoalPanel/ActiveModelsPanel/UsagePanel split out of the old combined panels, `Curtain.tsx`), removed the raw Memory-facts/Scratch panel, added a collapsible `LeftNav.tsx` (command reference + recent-project quick-switch, persisted in localStorage), fixed a real `/usage` crash (`usage_store.get_all_time_totals()`/`get_previous_session_totals()` don't exist, and nothing ever called `usage_store.record()` from the desktop backend), and started a PyInstaller-based sidecar bundling path (`scripts/build_sidecar.sh`) so the packaged app won't need a separate CLI install once the Rust wiring is verified locally.
- **2026-09-26**: Fixed two real bugs found while closing out GUI/CLI parity: `/compact` from the desktop app always raised `NameError: _installed_model_names is not defined` (it referenced a cli.py-only helper; serve.py now has its own copy), and `/why` raised `KeyError` for any actually-pending approval because `approve()` populated `self._pending` but never `self._pending_info`, which `handle_why_command()` reads. Also fixed `/auto` with no argument forcing auto-approve off instead of toggling it (mismatched the CLI's own bare `/auto`). Separately, `run_command`'s timeout could hang indefinitely past the requested timeout when a delegated shell command (e.g. `git push`) spawned a grandchild process that outlived the shell `subprocess.run(shell=True, timeout=...)` itself kills on timeout — the grandchild keeps the output pipes open, so the read for EOF never returns; fixed by killing the whole process tree (via `psutil`) on timeout instead of just the immediate child. On the frontend: `state.ts` was missing event handlers for `command_help`, `model`, `models`, `installed`, `catalog`, and `doctor` — those protocol events existed on the backend but were silently dropped by the reducer, so `/help`, `/model`, `/models`, `/installed`, `/catalog`, and `/doctor` typed into the chat produced no visible output at all. `/why`, `/summary`, `/compacted`, and `/tell`'s reply were also being folded into whatever the *last* assistant message happened to be (via the same `addItem`-onto-last-assistant path used for delegation activity), which silently dropped the reply entirely if no assistant message existed yet (e.g. a command typed before the first run) and otherwise attached it to a possibly-unrelated, already-finished task; all of these now post as their own `system`-role message instead. Fixed `SystemPanel`'s GPU line rendering `[object Object]` (it `.join()`-ed a list of `{name, vram_gb, backend}` objects as if it were a list of strings). The model picker no longer disables itself while a task is running — the CLI's `/model` switches immediately even mid-task (see the main CLAUDE.md's "`/model` was fully blocked while any task was busy" note), so the GUI now matches. Added a slash-command autocomplete menu (`SlashMenu.tsx`) to the chat input, covering every command `StdioServer.handle_help_command()` knows about, with arrow-key/Tab navigation — closing the "Full Slash-Command Coverage" gap below.
- **2026-09-25**: Updated the parity document to reflect the current state of the desktop app and backend. Added support for the `settings` event for both `set_auto` and `set_model` commands. Updated the `system_stats` request to return a nested `hardware` object plus live CPU, RAM usage stats. Invalid approval responses now surface as `error` events. Message queueing is implemented, with messages being queued rather than rejected while a run is in progress. The `queue` event carries the current items.

## CLI Subcommands -> Desktop

Every row marked ✅ is reachable two ways in the app: through a dedicated
panel/control where one exists, and always through the chat input as the
literal slash command (`/model claude-opus-5`, `/doctor`, ...) — the input
autocompletes every one of these (`SlashMenu.tsx`) and the reply renders as
its own message in the conversation.

| CLI Command | What it does | Native UI Control/Panel | Status |
|---|---|---|---|
| scan | Detects hardware | System Info Panel (`SystemPanel.tsx`), auto-refreshes every 5s | ✅ |
| doctor | Diagnoses problems | Chat reply (`/doctor`) | ✅ |
| setup | Initial setup walkthrough | — | ❌ Not done — run `localforge setup` once from a terminal first |
| models | Best-fit local model per modality | Chat reply (`/models`) | ✅ |
| catalog | Lists the full model catalog | Chat reply (`/catalog`) | ✅ |
| run | Runs a task | Chat input + Send, streamed live | ✅ |
| usage | Shows usage stats | Usage Panel (`UsagePanel.tsx`), always visible | ✅ |
| installed | Lists installed models | Chat reply (`/installed`) | ✅ |
| delete | Deletes a model | — | ❌ Not done — use `localforge delete` from a terminal |
| uninstall | Uninstalls localforge | — | ❌ Not done — use `localforge uninstall` from a terminal |
| theme | Changes the CLI's color theme | — | ❌ Not done (the desktop app has its own fixed theme, unrelated to the CLI's `theme.py`) |
| help | Lists every command | Chat reply (`/help`), plus the `/` autocomplete menu | ✅ |
| desktop | Hand off a terminal session to the desktop app | N/A from inside the GUI (this is a terminal→GUI command, not the reverse) — `localforge desktop`/`/desktop` from a terminal launches the app open to the current folder | ✅ |
| /clear, /new | Starts a new session | Chat reply, resets panels via `session_reset` | ✅ |
| /compact | Compacts conversation history | Chat reply (`/compact`) | ✅ |
| /auto | Toggles auto-approval | Header checkbox, or `/auto [on\|off]` in chat (toggles when bare, matching the CLI) | ✅ |
| /model | Shows or switches the orchestrator model | Header `ModelPicker`, switches even mid-task; or `/model <id>` in chat | ✅ |
| /local-model | Shows or overrides which model does coding/docs/general work (auto, a local model, or a paid cloud one) | "Change" action per row in `ActiveModelsPanel.tsx`, or `/local-model <type> <value>` in chat | ✅ |
| /memory | Shows memory facts, forget/clear | `MemoryPanel.tsx` | ✅ |
| /scratch | Shows scratchpad files, clear | Scratch section of `MemoryPanel.tsx` | ✅ |
| /summary | Shows session summary | Chat reply (`/summary`) | ✅ |
| /queue | Shows/clears the task queue | Queued-messages strip in `App.tsx` | ✅ |
| /stop | Stops the running task | Stop/Cancel button (header input area + status bar) | ✅ |
| /tell | Appends a note for the running task | Chat reply confirms it was queued; type `/tell <note>` while a task runs | ✅ |
| /stream | Toggles streamed local-model output | `/stream [on\|off]` in chat (output already streams live regardless; this mirrors the CLI's own toggle) | ✅ |
| /why | Explains a pending approval | Chat reply (`/why`), or ask a plain question at an approval prompt | ✅ |
| y/n/a on file/command approvals | Approve / decline / always-allow a change | `ApprovalPanel.tsx` — Approve / Always allow / Decline buttons | ✅ |

## Protocol Gaps

- **Setup wizard, `/delete`, `/uninstall`, `/theme`**: none of these have a
  server message type in `serve.py` yet (`handle()` has no case for
  `/setup`, `/delete`, `/uninstall`, or `/theme` — typing them just gets
  "Unknown command"). These need both new protocol messages and dedicated
  UI (a setup flow, a delete/uninstall confirmation dialog), which is a
  materially bigger feature than the rest of this table; run them from a
  terminal in the meantime.
- Everything else in the table above is done — this doc previously listed
  `/compact`, `/summary`, `/tell`, `/why`, `/help`, `/models`, `/installed`,
  `/catalog`, `/doctor`, `/model`, and y/n/a approvals as not done or not
  implemented, but the frontend/backend code already supported all of them
  by the time this was last accurate; see **Recent Changes** above for what
  was actually still broken (mostly missing event handlers, not missing
  features) and has now been fixed.

## Done

- **2026-09-24**: Updated `state.ts` to handle `memory`, `scratch`, and `session_reset` events. Added `MemoryFact`, `MemoryState`, and `ScratchFile` types, and updated state fields `memory`, `scratchFiles`. Implemented `session_reset` to clear messages, approvals, todos, status while keeping connection, model, auto-approve, usage, and system stats.
- **2026-09-24**: Created `MemoryPanel.tsx` to render Memory facts (per-fact `forget`, plus `clear`) and Scratch files (sizes, plus `clear`). Integrated it into `App.tsx` in the right-hand aside between `UsagePanel` and `TodoList`.
- **2026-09-24**: Updated `serve.py` to emit a `system_stats` event with a nested `hardware` object plus live CPU, RAM usage stats, matching `SystemPanel.tsx`.
- **2026-09-25**: Verified 596 Python tests pass, `npm run build` (tsc + vite) green, `cargo check` green.
- **2026-09-26**: Verified 646 Python tests pass (5 new, covering the bugfixes above; the 2 pre-existing failures are environment-specific — a missing IANA timezone database and a console-width difference in the sandbox this was run in — and were already failing before this pass, unrelated to the desktop app). `npm run build` (tsc + vite) green. `cargo check` for `src-tauri` could not be run in this environment (missing WebKitGTK/GTK system libraries, and the sandbox has no route to the OS package mirrors to install them) — no Rust code was changed in this pass, so this is a pre-existing environment limitation, not a regression; run it locally to confirm before shipping a build.
