use std::process::{Command, Child, ChildStdin, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::thread;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

struct Session {
    child: Child,
    stdin: ChildStdin,
}

#[derive(Default)]
struct AppState {
    session: Mutex<Option<Session>>,
}

fn kill_session(state: &AppState) {
    if let Ok(mut guard) = state.session.lock() {
        if let Some(mut session) = guard.take() {
            let _ = session.stdin.write_all(b"{\"type\":\"shutdown\"}\n");
            let _ = session.stdin.flush();
            drop(session.stdin);
            let _ = session.child.kill();
            let _ = session.child.wait();
        }
    }
}

#[tauri::command]
fn start_session(app: AppHandle, state: State<'_, AppState>, folder: String, model: Option<String>) -> Result<(), String> {
    kill_session(&state);
    // LOCALFORGE_BIN overrides the executable, e.g. for a dev checkout.
    let program = std::env::var("LOCALFORGE_BIN")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| "localforge".to_string());
    let mut args = vec!["serve".to_string(), "--stdio".to_string()];
    if let Some(m) = model.filter(|m| !m.trim().is_empty()) {
        args.push("--model".to_string());
        args.push(m);
    }
    let mut child = Command::new(&program)
        .args(&args)
        .current_dir(&folder)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {program}: {e}"))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdin = child.stdin.take().unwrap();

    let app_handle = app.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line) = line {
                if !line.trim().is_empty() {
                    let _ = app_handle.emit("localforge-event", line);
                }
            }
        }
        let _ = app_handle.emit("localforge-exit", ());
    });

    let app_handle = app.clone();
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                let _ = app_handle.emit("localforge-stderr", line);
            }
        }
    });

    if let Ok(mut guard) = state.session.lock() {
        *guard = Some(Session { child, stdin });
    }

    Ok(())
}

#[tauri::command]
fn send_message(state: State<'_, AppState>, message: String) -> Result<(), String> {
    let mut guard = state.session.lock().map_err(|e| e.to_string())?;
    let session = guard.as_mut().ok_or_else(|| "no session running".to_string())?;
    // Keep each message on one JSON line.
    let line = format!("{}\n", message.replace('\n', " "));
    session.stdin.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
    session.stdin.flush().map_err(|e| e.to_string())?;
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
