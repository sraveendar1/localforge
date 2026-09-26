# localforge desktop

A native desktop front end for `localforge` ([React](https://react.dev) +
[Tauri](https://tauri.app)). It launches `localforge serve --stdio` as a
child process and talks to it over a JSON-lines protocol on its
stdin/stdout — see `src/localforge/serve.py` (`StdioServer`) for the
protocol itself, and the repo root [README](../README.md#desktop-app-gui)
for what the app can do. There's no separate orchestration logic here: the
UI is a client of the same engine the CLI's interactive session uses.

## Prerequisites

- The `localforge` CLI installed and on `PATH` (`./install.sh` from the repo
  root, or `uv tool install .`) — the app spawns it, it doesn't reimplement it.
- Node.js + npm.
- Rust + Cargo.
- Tauri's own OS-level dependencies (WebKitGTK + related packages on Linux,
  Xcode command line tools on macOS, the WebView2 runtime on Windows — most
  Windows installs already have it). See [Tauri's prerequisites
  guide](https://v2.tauri.app/start/prerequisites/) if a build fails partway
  through with a missing system library.

## Developing

```bash
npm install
npm run tauri dev
```

This starts Vite (hot-reloading the React UI) and a debug build of the
Tauri shell together. `LOCALFORGE_BIN=/path/to/localforge npm run tauri dev`
points the spawned backend at a specific checkout instead of whatever
`localforge` resolves to on `PATH` — handy when developing the Python side
alongside the UI.

Other scripts:

```bash
npm run dev      # Vite dev server only (no Tauri window; useful for pure UI work)
npm run build    # tsc typecheck + a production Vite build (dist/)
npm run tauri build   # a real installable app (.dmg / .AppImage / .msi / ...)
```

## Layout

- `src/` — the React UI. `App.tsx` is the shell (header, chat, side panels,
  status bar); `state.ts` is the single reducer (`applyEvent`) that turns
  each protocol event from the backend into UI state — if you add a new
  event type on the Python side, this is where the frontend learns about it.
- `src-tauri/` — the thin Rust shell (`src/lib.rs`): three commands
  (`start_session`, `send_message`, `stop_session`) that spawn/feed/kill the
  `localforge serve --stdio` child process and forward its stdout lines to
  the frontend as `localforge-event`. It has no protocol knowledge of its
  own — every message is passed through as an opaque JSON string.

See [`docs/desktop_parity.md`](../docs/desktop_parity.md) in the repo root
for the CLI-command-to-GUI coverage table.
