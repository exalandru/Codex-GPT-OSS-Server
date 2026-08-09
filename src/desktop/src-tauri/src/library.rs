//! The model library, reached through the CLI rather than the daemon.
//!
//! ## Why not the management plane
//!
//! It was, and that was wrong. The library is *disk* state: a `models.json`
//! under Application Support, plus whatever is on the drives. It has nothing to
//! do with whether an inference server happens to be running.
//!
//! Routing it through the daemon made browsing and importing models require a
//! started server — backwards, since a user imports a model precisely so they
//! can then configure and start one. It also produced the failure that surfaced
//! this: every library action returned "No server is running", which read as the
//! buttons doing nothing at all.
//!
//! Downloads are the exception and stay on the daemon deliberately: one machine
//! should have one downloader, and two processes fetching the same repository
//! into the same directory is worse than requiring a running server.
//!
//! ## Still no duplicated logic
//!
//! Every judgement — whether a directory is a usable GPT-OSS model, what state
//! it is in, what a scan found — comes from the CLI's JSON. This file runs a
//! process and forwards bytes.

use serde_json::Value;

use crate::server_cli;

/// Run a CLI subcommand and parse its JSON.
///
/// Failures carry the CLI's own stderr: it explains *why* a directory was
/// refused, and that sentence is the whole value of validating at import time.
async fn run_json(subcommand: &[&str]) -> Result<Value, String> {
    let cli = server_cli::resolve()?;
    let output = tokio::process::Command::new(&cli.program)
        .args(cli.argv(subcommand))
        .output()
        .await
        .map_err(|e| format!("cannot run {}: {e}", cli.program.display()))?;

    if !output.status.success() {
        let message = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if message.is_empty() {
            format!("`{} {}` failed", server_cli::CLI_NAME, subcommand.join(" "))
        } else {
            // The CLI prefixes user-facing failures with "error: "; stripping it
            // avoids "Error: error: …" once the interface adds its own framing.
            message
                .strip_prefix("error: ")
                .unwrap_or(&message)
                .to_string()
        });
    }

    serde_json::from_slice(&output.stdout).map_err(|e| {
        format!(
            "unreadable output from `{} {}`: {e}",
            server_cli::CLI_NAME,
            subcommand.join(" ")
        )
    })
}

/// Run a subcommand whose success output is a human sentence, not JSON.
///
/// Returned as `{"message": "..."}` so the interface can show what the server
/// actually said instead of inventing its own confirmation wording.
async fn run_text(subcommand: &[&str]) -> Result<Value, String> {
    let cli = server_cli::resolve()?;
    let output = tokio::process::Command::new(&cli.program)
        .args(cli.argv(subcommand))
        .output()
        .await
        .map_err(|e| format!("cannot run {}: {e}", cli.program.display()))?;

    if !output.status.success() {
        let message = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if message.is_empty() {
            format!("`{} {}` failed", server_cli::CLI_NAME, subcommand.join(" "))
        } else {
            message
                .strip_prefix("error: ")
                .unwrap_or(&message)
                .to_string()
        });
    }

    Ok(serde_json::json!({
        "message": String::from_utf8_lossy(&output.stdout).trim().to_string(),
    }))
}

// -- profiles ----------------------------------------------------------------
//
// Also serverless, and for the same reason: profiles are configuration files,
// not server state. A user edits a profile in order to start a server with it.

/// The description the configuration form is generated from.
pub async fn profile_schema() -> Result<Value, String> {
    run_json(&["profiles", "schema"]).await
}

pub async fn profiles() -> Result<Value, String> {
    run_json(&["profiles", "list", "--json"]).await
}

/// Apply `field=value` assignments to one profile.
///
/// Validation stays on the server: bounds, choices and the profile's own
/// invariants are its to enforce, and a form that decided for itself would
/// disagree the first time a bound changed.
pub async fn set_profile(name: &str, assignments: &[String]) -> Result<Value, String> {
    let mut argv = vec!["profiles", "set", "--json", name];
    let owned: Vec<&str> = assignments.iter().map(String::as_str).collect();
    argv.extend(owned);
    run_json(&argv).await
}

/// Create a profile from the server's own defaults.
///
/// Every one of these forwards to the CLI rather than assembling a profile
/// here. Defaults, name rules and collision checks live on the server, and a
/// second copy in Rust would drift the first time one of them changed.
pub async fn new_profile(name: &str) -> Result<Value, String> {
    run_json(&["profiles", "new", "--json", name]).await
}

pub async fn duplicate_profile(source: &str, name: &str) -> Result<Value, String> {
    run_json(&["profiles", "duplicate", "--json", source, name]).await
}

pub async fn rename_profile(name: &str, new_name: &str) -> Result<Value, String> {
    run_json(&["profiles", "rename", "--json", name, new_name]).await
}

/// Delete a profile.
///
/// `force` is passed through rather than interpreted: the server decides
/// whether a running daemon makes this unsafe, and says so in its own words.
pub async fn remove_profile(name: &str, force: bool) -> Result<Value, String> {
    let mut argv = vec!["profiles", "remove", name];
    if force {
        argv.push("--force");
    }
    run_text(&argv).await
}

pub async fn set_default_profile(name: &str) -> Result<Value, String> {
    run_text(&["profiles", "default", name]).await
}

pub async fn list() -> Result<Value, String> {
    run_json(&["models", "list", "--json"]).await
}

/// The supported GPT-OSS models, already joined with what is installed.
///
/// One call rather than list-plus-reconcile here: deciding which installed
/// directory corresponds to which supported model is the server's judgement,
/// and repeating it in the interface would eventually disagree.
/// Import a directory, optionally requiring it to be a specific catalog model.
///
/// `expect` is forwarded, not checked: whether a directory *is* the 120B is the
/// server's judgement, made on the same rule the catalogue reconciles with.
pub async fn import_expecting(path: &str, expect: Option<String>) -> Result<Value, String> {
    let mut argv = vec!["models", "import", "--json"];
    if let Some(slug) = expect.as_deref() {
        argv.push("--expect");
        argv.push(slug);
    }
    argv.push(path);
    run_json(&argv).await
}

/// The per-model settings form description. Server-owned, like the profile's.
pub async fn model_config_schema() -> Result<Value, String> {
    run_json(&["models", "config-schema"]).await
}

/// One model's overrides, by stable id.
pub async fn model_config(slug: &str) -> Result<Value, String> {
    run_json(&["models", "config", "--json", slug]).await
}

/// Apply `field=value` assignments to one model. Coercion and validation stay
/// on the server, exactly as for a profile.
pub async fn set_model_config(slug: &str, assignments: &[String]) -> Result<Value, String> {
    let mut argv = vec!["models", "config", "--json", slug];
    let owned: Vec<&str> = assignments.iter().map(String::as_str).collect();
    argv.extend(owned);
    run_json(&argv).await
}

/// Where downloads are written, and where they should be written.
///
/// Serverless like the rest of the library: a user chooses a disk for weights
/// before there is any reason for a daemon to be running.
pub async fn storage() -> Result<Value, String> {
    run_json(&["models", "storage", "--json"]).await
}

pub async fn set_storage(path: &str) -> Result<Value, String> {
    run_json(&["models", "storage", "--json", path]).await
}

pub async fn catalog() -> Result<Value, String> {
    run_json(&["models", "catalog"]).await
}

/// Discover models under the configured roots.
///
/// Returns counts — found, added, already known — because "scanned" alone
/// cannot be told apart from "scanned and found nothing".
pub async fn scan() -> Result<Value, String> {
    run_json(&["models", "scan", "--json"]).await
}

pub async fn import(path: &str) -> Result<Value, String> {
    run_json(&["models", "import", "--json", path]).await
}

pub async fn forget(path: &str) -> Result<Value, String> {
    // No `--json`: this one's output is a sentence, and the exit code carries
    // the outcome. What matters is that the files are left alone, which the CLI
    // says and the interface repeats.
    let cli = server_cli::resolve()?;
    let output = tokio::process::Command::new(&cli.program)
        .args(cli.argv(&["models", "forget", path]))
        .output()
        .await
        .map_err(|e| format!("cannot run {}: {e}", cli.program.display()))?;

    if !output.status.success() {
        let message = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(message
            .strip_prefix("error: ")
            .unwrap_or(&message)
            .to_string());
    }
    Ok(serde_json::json!({
        "forgotten": path,
        "files_removed": false,
    }))
}
