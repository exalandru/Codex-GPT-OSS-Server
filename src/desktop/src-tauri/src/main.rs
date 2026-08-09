// Prevents an extra console window on Windows in release. Harmless on macOS,
// which is the only platform this targets.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // A diagnostic that runs the *shipped* resolution code without opening a
    // window. The failure it exists for — a GUI inheriting launchd's PATH
    // rather than a shell's — cannot be reproduced from a terminal, so being
    // able to ask the real binary what it would run is the only way to check it
    // outside a manual click-through.
    if std::env::args().any(|arg| arg == "--print-server-environment") {
        println!("{}", quantum_codex_desktop_lib::server_environment_report());
        return;
    }

    // The Models view reaches the library through the CLI, not the daemon, so
    // this must work with no server running — the regression that made both
    // library buttons look dead.
    if std::env::args().any(|arg| arg == "--print-library") {
        let report = tauri::async_runtime::block_on(quantum_codex_desktop_lib::library_report());
        print!("{report}");
        return;
    }

    quantum_codex_desktop_lib::run()
}
