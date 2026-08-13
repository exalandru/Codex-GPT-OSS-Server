//! The command surface consumed by the interface.
//!
//! The Rust side does only what a browser cannot: read a file outside the
//! sandbox, spawn a process, send a signal, and reach a loopback endpoint that
//! needs a token from disk. Everything else — what a status means, which fields
//! exist, what a profile contains — belongs to the Python server, and reaches
//! the interface as opaque JSON.
//!
//! That boundary is the reason no schema is repeated here. A capability added
//! on the server appears in the interface without a Rust change.

mod bootstrap;
mod daemon;
mod library;
mod paths;
mod server_cli;

use serde_json::Value;

#[tauri::command]
async fn daemon_discover() -> daemon::Discovery {
    daemon::discover().await
}

#[tauri::command]
async fn daemon_status() -> Result<Value, String> {
    daemon::status().await
}

#[tauri::command]
async fn daemon_start(profile: Option<String>) -> Result<(), String> {
    daemon::start(profile).await
}

/// The managed runtime's state: UNINITIALIZED, READY, UPDATE_REQUIRED, …
///
/// Independent of whether a model is configured. A healthy runtime with no
/// GPT-OSS weights anywhere is a normal, reportable situation.
#[tauri::command]
fn runtime_status(app: tauri::AppHandle) -> bootstrap::RuntimeStatus {
    bootstrap::status(&app)
}

/// Install, update or repair the managed runtime.
///
/// Never called implicitly: fetching an interpreter and a dependency tree takes
/// minutes, and a Start button must not silently become that. Progress arrives
/// on the `bootstrap` event, because a silent wait of that length reads as a
/// hang.
#[tauri::command]
async fn runtime_initialize(app: tauri::AppHandle) -> Result<(), String> {
    bootstrap::initialize(app).await
}

/// The description the configuration form is generated from.
#[tauri::command]
async fn profile_schema() -> Result<Value, String> {
    library::profile_schema().await
}

#[tauri::command]
async fn profiles() -> Result<Value, String> {
    library::profiles().await
}

#[tauri::command]
async fn set_profile(name: String, assignments: Vec<String>) -> Result<Value, String> {
    library::set_profile(&name, &assignments).await
}

#[tauri::command]
async fn import_model_for(path: String, expect: Option<String>) -> Result<Value, String> {
    library::import_expecting(&path, expect).await
}

#[tauri::command]
async fn model_config_schema() -> Result<Value, String> {
    library::model_config_schema().await
}

#[tauri::command]
async fn model_config(slug: String) -> Result<Value, String> {
    library::model_config(&slug).await
}

#[tauri::command]
async fn set_model_config(slug: String, assignments: Vec<String>) -> Result<Value, String> {
    library::set_model_config(&slug, &assignments).await
}

#[tauri::command]
async fn model_storage() -> Result<Value, String> {
    library::storage().await
}

#[tauri::command]
async fn set_model_storage(path: String) -> Result<Value, String> {
    library::set_storage(&path).await
}

#[tauri::command]
async fn model_catalog() -> Result<Value, String> {
    library::catalog().await
}

#[tauri::command]
async fn new_profile(name: String) -> Result<Value, String> {
    library::new_profile(&name).await
}

#[tauri::command]
async fn duplicate_profile(source: String, name: String) -> Result<Value, String> {
    library::duplicate_profile(&source, &name).await
}

#[tauri::command]
async fn rename_profile(name: String, new_name: String) -> Result<Value, String> {
    library::rename_profile(&name, &new_name).await
}

#[tauri::command]
async fn remove_profile(name: String, force: bool) -> Result<Value, String> {
    library::remove_profile(&name, force).await
}

#[tauri::command]
async fn set_default_profile(name: String) -> Result<Value, String> {
    library::set_default_profile(&name).await
}

/// The model library, through the CLI: disk state, not server state.
///
/// Deliberately not the management plane. Browsing and importing models must
/// work with no server running — a user imports a model in order to configure
/// and start one.
#[tauri::command]
async fn list_models() -> Result<Value, String> {
    library::list().await
}

#[tauri::command]
async fn scan_models() -> Result<Value, String> {
    library::scan().await
}

#[tauri::command]
async fn import_model(path: String) -> Result<Value, String> {
    library::import(&path).await
}

#[tauri::command]
async fn forget_model(path: String) -> Result<Value, String> {
    library::forget(&path).await
}

#[tauri::command]
async fn request_diagnostics(limit: Option<u32>) -> Result<Value, String> {
    daemon::request_diagnostics(limit.unwrap_or(50)).await
}

#[tauri::command]
async fn download_status() -> Result<Value, String> {
    daemon::download_status().await
}

#[tauri::command]
async fn start_download(repo: String, destination: Option<String>) -> Result<Value, String> {
    daemon::start_download(&repo, destination).await
}

#[tauri::command]
async fn cancel_download() -> Result<Value, String> {
    daemon::cancel_download().await
}

/// Ask the user to choose a directory, under a title that says what for.
///
/// A native picker is one of the few things the interface genuinely cannot do
/// itself. What comes back is a path and nothing more: whether it is what the
/// title asked for is the server's judgement, made when the path is used.
async fn pick_folder(app: tauri::AppHandle, title: &str) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;

    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title(title)
        .pick_folder(move |chosen| {
            let _ = tx.send(chosen.map(|path| path.to_string()));
        });
    rx.await.ok().flatten()
}

#[tauri::command]
async fn choose_model_directory(app: tauri::AppHandle) -> Option<String> {
    pick_folder(app, "Choose a GPT-OSS model directory").await
}

/// Ask for a LoRA adapter directory.
///
/// Separate from the model picker for the title alone, which is the only thing
/// telling the user which of the two kinds of directory this dialog wants.
#[tauri::command]
async fn choose_adapter_directory(app: tauri::AppHandle) -> Option<String> {
    pick_folder(app, "Choose a LoRA adapter directory").await
}

/// Show a path in the Finder.
#[tauri::command]
fn reveal_in_finder(path: String) -> Result<(), String> {
    // `-R` selects the item rather than opening it, which for a model directory
    // is what someone asking to "reveal" actually wants.
    std::process::Command::new("open")
        .args(["-R", &path])
        .spawn()
        .map_err(|e| format!("cannot reveal {path}: {e}"))?;
    Ok(())
}

/// Whether a model is configured, asked of the server rather than guessed.
///
/// Returns the CLI's own JSON untouched, so the desktop app never acquires an
/// opinion about what a profile contains.
#[tauri::command]
async fn model_status() -> Result<Value, String> {
    let cli = server_cli::resolve()?;
    let output = tokio::process::Command::new(&cli.program)
        .args(cli.argv(&["profiles", "list", "--json"]))
        .output()
        .await
        .map_err(|e| format!("cannot run {}: {e}", cli.program.display()))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    serde_json::from_slice(&output.stdout).map_err(|e| format!("unreadable profile list: {e}"))
}

/// Where the server executable was found, and everywhere that was tried.
///
/// Shares `server_cli::resolve` with start and restart, so what this reports is
/// what those would actually run.
#[tauri::command]
fn server_environment() -> server_cli::Diagnosis {
    server_cli::diagnose()
}

#[tauri::command]
async fn daemon_stop() -> Result<(), String> {
    daemon::stop().await
}

#[tauri::command]
async fn daemon_restart(profile: Option<String>) -> Result<(), String> {
    // Stop first and wait for it: starting while the old process still holds the
    // port produces a bind failure that reads as "restart is broken".
    let _ = daemon::stop().await;
    daemon::start(profile).await
}

#[tauri::command]
async fn cache_clear() -> Result<Value, String> {
    daemon::clear_cache().await
}

/// Release the resident model without stopping the server.
///
/// The same supervisor operation the idle timer performs, so a model released
/// from the dashboard and one released by the timeout leave identical state.
#[tauri::command]
async fn model_unload() -> Result<Value, String> {
    daemon::unload_model().await
}

#[tauri::command]
fn logs_tail(lines: Option<usize>) -> Result<Vec<String>, String> {
    daemon::tail_log(lines.unwrap_or(300))
}

/// The Codex command, produced by the CLI rather than rebuilt here.
///
/// Reimplementing it in Rust would create a second definition of the provider
/// wiring, and the two would drift the first time Codex changes anything.
#[tauri::command]
async fn codex_launch_command(model: Option<String>) -> Result<String, String> {
    codex_launch_for(&["codex", "launch"], model).await
}

/// The models this server can be pointed at, and their effective reasoning
/// effort, as the server resolves them.
///
/// Fetched rather than derived: which models exist, and what effort each one
/// actually runs at, are the library's and the per-model configuration's
/// answers. A selector that built its own list would be a second opinion about
/// both.
#[tauri::command]
async fn codex_launch_models() -> Result<Value, String> {
    let raw = codex_launch(&["codex", "launch", "--models-json"]).await?;
    serde_json::from_str(&raw).map_err(|e| format!("cannot read the model list: {e}"))
}

/// The persistent `~/.codex/config.toml` fragment.
///
/// Same generator, a different presentation of it: the Codex CLI's global
/// configuration and the VS Code extension take no `-c` overrides.
#[tauri::command]
async fn codex_launch_config(model: Option<String>) -> Result<String, String> {
    codex_launch_for(&["codex", "launch", "--config"], model).await
}

/// One subcommand, optionally pinned to a model.
///
/// The model is passed to the generator rather than substituted into its
/// output: the effort that travels with it is resolved server-side, and a
/// caller splicing a name into a finished command would lose that.
async fn codex_launch_for(subcommand: &[&str], model: Option<String>) -> Result<String, String> {
    match model {
        Some(model) => {
            let mut args: Vec<&str> = subcommand.to_vec();
            args.push("--model");
            args.push(&model);
            codex_launch(&args).await
        }
        None => codex_launch(subcommand).await,
    }
}

async fn codex_launch(subcommand: &[&str]) -> Result<String, String> {
    let cli = server_cli::resolve()?;

    let output = tokio::process::Command::new(&cli.program)
        .args(cli.argv(subcommand))
        .output()
        .await
        .map_err(|e| format!("cannot run {}: {e}", cli.program.display()))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The diagnostic `--print-server-environment` prints.
///
/// Lives here rather than in `main` so it goes through exactly the resolution
/// the running application uses.
pub fn server_environment_report() -> String {
    let diagnosis = server_cli::diagnose();
    let mut report = String::new();
    report.push_str("Searched:\n");
    for location in &diagnosis.searched {
        let mark = if location.found { "found  " } else { "missing" };
        report.push_str(&format!("  [{mark}] {}\n", location.description));
    }
    match &diagnosis.resolved {
        Some(cli) => report.push_str(&format!(
            "\nResolved: {} {:?}\nSource:   {}\n",
            cli.program.display(),
            cli.args,
            cli.source
        )),
        None => report.push_str("\nResolved: nothing — the server cannot be started.\n"),
    }
    report.push_str(&format!("\nInherited PATH: {}\n", diagnosis.inherited_path));
    report
}

/// What `--print-library` prints.
///
/// Runs the shipped `library` code — resolve the CLI, run it, parse its JSON —
/// so the path behind the Models view can be checked without a window. It does
/// not cover the React click or the native picker, which need a person.
pub async fn library_report() -> String {
    match library::list().await {
        Ok(value) => {
            let models = value.get("models").and_then(|m| m.as_array());
            let roots = value.get("roots").and_then(|r| r.as_array());
            let mut report = format!(
                "library: {} model(s), {} root(s)\n",
                models.map_or(0, |m| m.len()),
                roots.map_or(0, |r| r.len())
            );
            for model in models.into_iter().flatten() {
                report.push_str(&format!(
                    "  {:<16} {}\n",
                    model.get("state").and_then(|s| s.as_str()).unwrap_or("?"),
                    model.get("path").and_then(|p| p.as_str()).unwrap_or("?")
                ));
            }
            report
        }
        Err(message) => format!("library unavailable: {message}\n"),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // Decided once, before anything can resolve the server: whether this
        // build carries its own Python project, and therefore whether it owns
        // its runtime or may borrow a development checkout's.
        .setup(|app| {
            server_cli::set_packaged(bootstrap::is_packaged(app.handle()));
            Ok(())
        })
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            daemon_discover,
            server_environment,
            runtime_status,
            runtime_initialize,
            model_status,
            profile_schema,
            profiles,
            set_profile,
            model_catalog,
            model_storage,
            set_model_storage,
            model_config_schema,
            model_config,
            set_model_config,
            import_model_for,
            new_profile,
            duplicate_profile,
            rename_profile,
            remove_profile,
            set_default_profile,
            list_models,
            scan_models,
            import_model,
            forget_model,
            request_diagnostics,
            download_status,
            start_download,
            cancel_download,
            choose_model_directory,
            choose_adapter_directory,
            reveal_in_finder,
            daemon_status,
            daemon_start,
            daemon_stop,
            daemon_restart,
            cache_clear,
            model_unload,
            logs_tail,
            codex_launch_command,
            codex_launch_config,
            codex_launch_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the desktop application");
}
