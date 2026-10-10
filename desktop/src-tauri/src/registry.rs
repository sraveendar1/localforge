//! Which window has which project open. Plain data and functions (no Tauri types) so the rules can be
//! unit-tested on their own: one project folder per window, never the same folder in two windows.

use std::collections::HashMap;

/// The label of the first window.
pub const MAIN_LABEL: &str = "main";

/// A folder compared the way the file system compares it: resolved to its real path when it exists, and
/// case-folded on Windows and macOS, whose default file systems ignore case.
pub fn canonical(folder: &str) -> String {
    let resolved = std::fs::canonicalize(folder)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| folder.to_string());
    // canonicalize on Windows returns a \\?\ prefixed path; the prefix is noise for comparisons.
    let resolved = resolved.strip_prefix(r"\\?\").unwrap_or(&resolved).to_string();
    if cfg!(any(target_os = "windows", target_os = "macos")) {
        resolved.to_lowercase()
    } else {
        resolved
    }
}

/// The window (other than `except`) that already has `folder` open, if any.
pub fn window_with(open: &HashMap<String, String>, folder: &str, except: Option<&str>) -> Option<String> {
    let wanted = canonical(folder);
    open.iter()
        .filter(|(label, _)| Some(label.as_str()) != except)
        .find(|(_, f)| canonical(f) == wanted)
        .map(|(label, _)| label.clone())
}

/// A window label not in use: `main` first, then `w2`, `w3`, ...
pub fn next_label<'a>(in_use: impl Iterator<Item = &'a String> + Clone) -> String {
    if !in_use.clone().any(|l| l == MAIN_LABEL) {
        return MAIN_LABEL.to_string();
    }
    let mut n = 2;
    loop {
        let candidate = format!("w{n}");
        if !in_use.clone().any(|l| *l == candidate) {
            return candidate;
        }
        n += 1;
    }
}

/// The folder a launch asks for: the first argument after the program name that is an existing
/// directory. Anything else the OS injects is ignored.
pub fn folder_from_args(args: &[String]) -> Option<String> {
    args.iter().skip(1).find(|a| std::path::Path::new(a.as_str()).is_dir()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn a_folder_already_open_is_found_in_its_window_only() {
        let open = map(&[("main", "/tmp"), ("w2", "/")]);
        assert_eq!(window_with(&open, "/tmp", None), Some("main".to_string()));
        assert_eq!(window_with(&open, "/tmp", Some("main")), None); // the window itself doesn't count
        assert_eq!(window_with(&open, "/nonexistent-dir-xyz", None), None);
    }

    #[test]
    fn the_same_folder_spelled_differently_is_the_same_folder() {
        let open = map(&[("main", "/tmp")]);
        assert_eq!(window_with(&open, "/tmp/", None), Some("main".to_string()));
        assert_eq!(window_with(&open, "/tmp/.", None), Some("main".to_string()));
    }

    #[test]
    fn labels_are_main_first_then_numbered_and_reused_when_free() {
        let none: Vec<String> = vec![];
        assert_eq!(next_label(none.iter()), "main");
        let one = vec!["main".to_string()];
        assert_eq!(next_label(one.iter()), "w2");
        let some = vec!["main".to_string(), "w2".to_string(), "w4".to_string()];
        assert_eq!(next_label(some.iter()), "w3");
        let gone_main = vec!["w2".to_string()];
        assert_eq!(next_label(gone_main.iter()), "main");
    }

    #[test]
    fn only_a_real_directory_argument_is_a_folder() {
        let args = vec!["app".to_string(), "--flag".to_string(), "/tmp".to_string()];
        assert_eq!(folder_from_args(&args), Some("/tmp".to_string()));
        assert_eq!(folder_from_args(&["app".to_string(), "-psn_0_1".to_string()]), None);
        assert_eq!(folder_from_args(&["/tmp".to_string()]), None); // argv[0] is the program
    }
}
