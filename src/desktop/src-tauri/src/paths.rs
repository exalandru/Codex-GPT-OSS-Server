//! Where the server keeps its state.
//!
//! These paths mirror `quantum_codex/config.py`. They are duplicated because a
//! Rust process cannot import Python, and that duplication is deliberately kept
//! to *locations* only — never to the shape of what the files contain. The
//! desktop app reads `runtime.json` to find the daemon and otherwise treats
//! server responses as opaque JSON, so a schema change on the Python side never
//! requires a matching change here.
//!
//! `QUANTUM_CODEX_HOME` is honoured for the same reason the CLI honours it: it
//! is what makes the whole surface testable without touching real user state.

use std::path::PathBuf;

pub const BUNDLE_ID: &str = "com.exalandru.qcs";

pub fn app_support_dir() -> PathBuf {
    if let Ok(override_path) = std::env::var("QUANTUM_CODEX_HOME") {
        if !override_path.is_empty() {
            return PathBuf::from(override_path);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join("Library")
        .join("Application Support")
        .join(BUNDLE_ID)
}

pub fn runtime_file() -> PathBuf {
    app_support_dir().join("runtime.json")
}

pub fn logs_dir() -> PathBuf {
    app_support_dir().join("logs")
}

pub fn server_log_file() -> PathBuf {
    logs_dir().join("server.log")
}
