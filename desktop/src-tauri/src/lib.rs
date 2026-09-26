use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct Session {
    child: CommandChild,
}

#[derive(Default)]
struct AppState {
    session: Mutex<Option<Session>>,
}

fn kill_session(state: &AppState) {
    if let Ok(mut guard) = state.session.lock() {
        if let Some(session) = guard.take() {
            let mut child = session.child;
            let _ = child.write(b"{\"type\":\"shutdown\"}\n");
            let _ = child.kill();
        }
    }
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
    app: AppHandle,
    state: State<'_, AppState>,
    folder: String,
    model: Option<String>,
) -> Result<(), String> {
    kill_session(&state);

    let mut args = vec!["serve".to_string(), "--stdio".to_string()];
    if let Some(m) = model.filter(|m| !m.trim().is_empty()) {
        args.push("--model".to_string());
        args.push(m);
    }

    let (mut rx, child) = spawn_backend(&app, &args, &folder)?;

    if let Ok(mut guard) = state.session.lock() {
        *guard = Some(Session { child });
    }

    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    for line in String::from_utf8_lossy(&bytes).lines() {
                        if !line.trim().is_empty() {
                            let _ = app_handle.emit("localforge-event", line.to_string());
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    let _ = app_handle.emit("localforge-stderr", text);
                }
                CommandEvent::Error(err) => {
                    let _ = app_handle.emit("localforge-stderr", err);
                }
                CommandEvent::Terminated(_) => {
                    let _ = app_handle.emit("localforge-exit", ());
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn send_message(state: State<'_, AppState>, message: String) -> Result<(), String> {
    let mut guard = state.session.lock().map_err(|e| e.to_string())?;
    let session = guard.as_mut().ok_or_else(|| "no session running".to_string())?;
    // Keep each message on one JSON line.
    let line = format!("{}\n", message.replace('\n', " "));
    session
        .child
        .write(line.as_bytes())
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn stop_session(state: State<'_, AppState>) -> Result<(), String> {
    kill_session(&state);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![start_session, send_message, stop_session])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                kill_session(&app.state::<AppState>());
            }
        });
}
