mod registry;

use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

use registry::{canonical, folder_from_args, next_label, window_with, MAIN_LABEL};

struct Session {
    child: CommandChild,
}

/// Every window has its own backend process and its own project folder; nothing is shared between
/// windows except the folder list below, which is what keeps one folder from being opened twice.
#[derive(Default)]
struct AppState {
    /// window label -> its running backend
    sessions: Mutex<HashMap<String, Session>>,
    /// window label -> the project folder open in it
    folders: Mutex<HashMap<String, String>>,
    /// window label -> a folder a freshly created window should open as soon as it asks
    pending: Mutex<HashMap<String, String>>,
}

/// Set from argv when the app is launched with a folder to open directly -- e.g. `localforge
/// desktop`'s hand-off (see cli.py), which runs the app with the folder as its argument. Filtered to
/// existing directories so an unrelated OS-injected argument can't be mistaken for one. It opens in
/// the first window; folders from later launches arrive through the single-instance callback.
struct InitialFolder(Option<String>);

#[tauri::command]
fn get_initial_folder(
    window: tauri::WebviewWindow,
    state: State<'_, AppState>,
    initial: State<'_, InitialFolder>,
) -> Option<String> {
    let label = window.label().to_string();
    if let Ok(mut pending) = state.pending.lock() {
        if let Some(folder) = pending.remove(&label) {
            return Some(folder);
        }
    }
    if label == MAIN_LABEL {
        return initial.0.clone();
    }
    None
}

/// How long a killed session's backend gets to run its own graceful
/// shutdown (StdioServer.close(): cancel the in-flight task, then save
/// session memory/facts -- see serve.py) before being force-killed as a
/// safety net. Writing "shutdown" and calling kill() right after it (the
/// previous code) raced the write against the kill: write() only queues
/// bytes on the pipe and returns immediately, so the kill essentially
/// always won, meaning the memory-save-on-close fix never actually got to
/// run when switching folders or quitting -- the exact case it exists for.
const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

/// Stop one window's backend and forget its folder.
fn kill_session(state: &AppState, label: &str) {
    if let Ok(mut folders) = state.folders.lock() {
        folders.remove(label);
    }
    let session = state.sessions.lock().ok().and_then(|mut sessions| sessions.remove(label));
    if let Some(session) = session {
        let mut child = session.child;
        let _ = child.write(b"{\"type\":\"shutdown\"}\n");
        // Give it the grace period to exit on its own; only force-kill
        // if it hasn't (a hang, e.g. memory extraction stuck on a
        // stalled local model). Detached so callers don't block on the
        // *old* session while this plays out -- starting a new one
        // doesn't depend on it.
        std::thread::spawn(move || {
            std::thread::sleep(SHUTDOWN_GRACE);
            let _ = child.kill();
        });
    }
}

fn kill_all(state: &AppState) {
    let labels: Vec<String> = state.sessions.lock().map(|s| s.keys().cloned().collect()).unwrap_or_default();
    for label in labels {
        kill_session(state, &label);
    }
}

fn focus(app: &AppHandle, label: &str) {
    if let Some(window) = app.get_webview_window(label) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Folders open in windows -- including ones a just-created window is about to open, so asking twice in
/// quick succession doesn't make two windows for one project.
fn open_folders(state: &AppState) -> HashMap<String, String> {
    let mut all = state.pending.lock().map(|p| p.clone()).unwrap_or_default();
    if let Ok(folders) = state.folders.lock() {
        all.extend(folders.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    all
}

/// A new, empty window (it asks for its folder, if it has one pending, when its page loads).
fn create_window(app: &AppHandle, state: &AppState, folder: Option<String>) -> Result<String, String> {
    let label = {
        let existing: Vec<String> = app.webview_windows().keys().cloned().collect();
        next_label(existing.iter())
    };
    if let Some(folder) = folder {
        if let Ok(mut pending) = state.pending.lock() {
            pending.insert(label.clone(), folder);
        }
    }
    WebviewWindowBuilder::new(app, &label, WebviewUrl::App("index.html".into()))
        .title("LocalForge Desktop")
        .inner_size(1100.0, 760.0)
        .min_inner_size(640.0, 480.0)
        // same as the first window: dropping an image onto the message box must reach the page
        .disable_drag_drop_handler()
        .build()
        .map_err(|e| format!("could not open a new window: {e}"))?;
    Ok(label)
}

/// Open `folder` in its own window -- or, if it is already open somewhere, bring that window forward and
/// tell it someone tried to open it again. Returns "focused" or "opened".
fn open_in_window(app: &AppHandle, state: &AppState, folder: &str) -> Result<String, String> {
    if let Some(label) = window_with(&open_folders(state), folder, None) {
        let _ = app.emit_to(label.as_str(), "localforge-duplicate-open", folder.to_string());
        focus(app, &label);
        return Ok("focused".to_string());
    }
    create_window(app, state, Some(folder.to_string()))?;
    Ok("opened".to_string())
}

/// Resolves the backend the same way every launch: `LOCALFORGE_BIN` first
/// (a dev convenience -- point at a checkout's own venv without rebuilding
/// the sidecar), then the bundled sidecar (the standalone binary
/// `scripts/build_sidecar.sh` produces, embedded in the packaged app so an
/// end user never needs Python or a separate CLI install), and finally a
/// plain `localforge` on PATH as a last resort for a dev running `npm run
/// tauri dev` without having built a sidecar at all. Sidecar resolution
/// only fails when none was bundled for this platform/build, which is
/// exactly the case the PATH fallback exists for.
fn spawn_backend(
    app: &AppHandle,
    args: &[String],
    folder: &str,
) -> Result<(tauri::async_runtime::Receiver<CommandEvent>, CommandChild), String> {
    let shell = app.shell();

    if let Ok(bin) = std::env::var("LOCALFORGE_BIN") {
        let bin = bin.trim();
        if !bin.is_empty() {
            return shell
                .command(bin)
                .args(args)
                .current_dir(folder)
                .spawn()
                .map_err(|e| format!("could not start {bin}: {e}"));
        }
    }

    match shell.sidecar("localforge") {
        Ok(cmd) => cmd
            .args(args)
            .current_dir(folder)
            .spawn()
            .map_err(|e| format!("could not start the bundled localforge sidecar: {e}")),
        Err(_) => shell
            .command("localforge")
            .args(args)
            .current_dir(folder)
            .spawn()
            .map_err(|e| {
                format!(
                    "could not start localforge (no bundled sidecar for this build, and none found on PATH): {e}"
                )
            }),
    }
}

#[tauri::command]
async fn start_session(
    window: tauri::WebviewWindow,
    app: AppHandle,
    state: State<'_, AppState>,
    folder: String,
    model: Option<String>,
) -> Result<String, String> {
    let label = window.label().to_string();

    // One session per project: if another window has this folder, go there and say so, and leave
    // this window as it was.
    if let Some(other) = window_with(&open_folders(&state), &folder, Some(&label)) {
        let _ = app.emit_to(other.as_str(), "localforge-duplicate-open", folder.clone());
        focus(&app, &other);
        return Ok("already_open".to_string());
    }

    kill_session(&state, &label);

    let mut args = vec!["serve".to_string(), "--stdio".to_string()];
    if let Some(m) = model.filter(|m| !m.trim().is_empty()) {
        args.push("--model".to_string());
        args.push(m);
    }

    let (mut rx, child) = spawn_backend(&app, &args, &folder)?;

    if let Ok(mut sessions) = state.sessions.lock() {
        sessions.insert(label.clone(), Session { child });
    }
    if let Ok(mut folders) = state.folders.lock() {
        folders.insert(label.clone(), canonical(&folder));
    }

    // This window's backend talks only to this window.
    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    for line in String::from_utf8_lossy(&bytes).lines() {
                        if !line.trim().is_empty() {
                            let _ = app_handle.emit_to(label.as_str(), "localforge-event", line.to_string());
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let _ = app_handle.emit_to(label.as_str(), "localforge-stderr", text);
                }
                CommandEvent::Error(err) => {
                    let _ = app_handle.emit_to(label.as_str(), "localforge-stderr", err);
                }
                CommandEvent::Terminated(_) => {
                    let _ = app_handle.emit_to(label.as_str(), "localforge-exit", ());
                    break;
                }
                _ => {}
            }
        }
    });

    Ok("started".to_string())
}

#[tauri::command]
fn send_message(window: tauri::WebviewWindow, state: State<'_, AppState>, message: String) -> Result<(), String> {
    let mut sessions = state.sessions.lock().map_err(|e| e.to_string())?;
    let session = sessions.get_mut(window.label()).ok_or_else(|| "no session running".to_string())?;
    // Keep each message on one JSON line.
    let line = format!("{}\n", message.replace('\n', " "));
    session.child.write(line.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn stop_session(window: tauri::WebviewWindow, state: State<'_, AppState>) -> Result<(), String> {
    kill_session(&state, window.label());
    Ok(())
}

/// Asked before a window changes anything: is `folder` already open in a *different* window? If so that
/// window is brought forward and told it was asked again, and this returns true (so the caller leaves
/// its own window as it is).
#[tauri::command]
fn folder_open_elsewhere(window: tauri::WebviewWindow, app: AppHandle, state: State<'_, AppState>, folder: String) -> bool {
    match window_with(&open_folders(&state), &folder, Some(window.label())) {
        Some(other) => {
            let _ = app.emit_to(other.as_str(), "localforge-duplicate-open", folder);
            focus(&app, &other);
            true
        }
        None => false,
    }
}

/// A new empty window. (Async: creating a window from a synchronous command can deadlock on Windows.)
#[tauri::command]
async fn new_window(app: AppHandle, state: State<'_, AppState>) -> Result<String, String> {
    create_window(&app, &state, None)
}

/// Open a project in its own window, or bring forward the window that already has it.
#[tauri::command]
async fn open_project_window(app: AppHandle, state: State<'_, AppState>, folder: String) -> Result<String, String> {
    open_in_window(&app, &state, &folder)
}

#[derive(serde::Serialize)]
struct OpenWindow {
    label: String,
    folder: String,
}

/// Which projects are open in which windows, for marking them in the recent-projects list.
#[tauri::command]
fn open_windows(state: State<'_, AppState>) -> Vec<OpenWindow> {
    open_folders(&state).into_iter().map(|(label, folder)| OpenWindow { label, folder }).collect()
}

/// A second launch of the app (another double-click, or `localforge desktop <folder>`) arrives here in
/// the running instance: it opens a new window -- on the folder, if one was given.
fn handle_second_launch(app: &AppHandle, argv: Vec<String>) {
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let state = app.state::<AppState>();
        match folder_from_args(&argv) {
            Some(folder) => {
                let _ = open_in_window(&app, &state, &folder);
            }
            None => {
                let _ = create_window(&app, &state, None);
            }
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let initial_folder = folder_from_args(&std::env::args().collect::<Vec<_>>());

    let mut builder = tauri::Builder::default();
    #[cfg(desktop)]
    {
        // Must be registered first. A second launch is handed to the running app, which opens a new window.
        builder = builder.plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            handle_second_launch(app, argv);
        }));
    }
    builder
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::default())
        .manage(InitialFolder(initial_folder))
        .invoke_handler(tauri::generate_handler![
            start_session,
            send_message,
            stop_session,
            get_initial_folder,
            new_window,
            open_project_window,
            open_windows,
            folder_open_elsewhere
        ])
        .on_window_event(|window, event| {
            // A closed window takes only its own backend with it.
            if let tauri::WindowEvent::Destroyed = event {
                kill_session(&window.state::<AppState>(), window.label());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                kill_all(&app.state::<AppState>());
            }
        });
}
