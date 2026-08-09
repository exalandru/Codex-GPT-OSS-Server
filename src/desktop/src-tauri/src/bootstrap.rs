//! The managed Python runtime: first-run installation, updates and repair.
//!
//! The application bundle stays light. It carries `uv` and the locked project —
//! manifest, lock, interpreter pin, package — and nothing else. Python itself,
//! the dependency tree, MLX and the model weights are all absent: uv fetches an
//! interpreter and rebuilds exactly the locked set on the user's machine, into
//! the Application Support directory.
//!
//! Nothing here depends on system Python, Homebrew, the shell `PATH`, or a
//! development checkout. A packaged build must be able to reconstruct its
//! runtime from bundle resources alone.
//!
//! ## The runtime is a state, not a boolean
//!
//! A long install must never happen as a side effect of pressing Start, so the
//! state is explicit and the interface asks before doing anything expensive:
//!
//! ```text
//! UNINITIALIZED   nothing installed yet; offer to initialise
//! INITIALIZING    a sync is running
//! READY           usable
//! UPDATE_REQUIRED installed, but built from a different project
//! BROKEN          present and unusable, or nothing to install from
//! ```
//!
//! This is independent of whether a *model* is configured. A runtime can be
//! perfectly healthy with no GPT-OSS weights anywhere.
//!
//! ## Detecting an obsolete environment
//!
//! By fingerprint, not by version number. Quantum Diffusion Server learned this
//! the hard way: changing the Python code without bumping the app version left
//! the environment looking current, so the copy never happened, the sync never
//! re-ran, and the app kept serving whatever the *first* install captured.
//!
//! Their fix compares modification times. This hashes content instead —
//! the interpreter pin, the manifest, the lock and every source file — because
//! an mtime changes when a file is merely touched, and does not change when a
//! file is restored from an archive. A content hash answers the actual question:
//! would a sync produce something different from what is installed?
//!
//! ## Repair is not destruction
//!
//! Rebuilding replaces `server/`, `env/` and the uv caches. It never touches
//! `profiles.json`, `settings.json` or anything under a model directory, so
//! repairing a broken runtime costs a download and nothing else.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

use crate::paths;
use crate::server_cli;

/// Directories the runtime owns. Everything else under Application Support —
/// profiles, settings, logs — belongs to the user and is never rebuilt.
const PROJECT_DIR: &str = "server";

/// Where the staged Python project lands inside the bundle.
///
/// Must match the destination in `tauri.conf.json`'s `bundle.resources`, which
/// maps the staging directory to `server/`. `BaseDirectory::Resource` already
/// *is* `Contents/Resources`, so this is resolved relative to that -- asking
/// for `resources/server` looked for `Contents/Resources/resources/server`,
/// which never existed. The app then believed it carried nothing to install
/// from, and a factory-clean packaged launch could not have bootstrapped even
/// if it had tried. `bundled_resource_matches_the_bundle_configuration` keeps
/// the two files in agreement.
const BUNDLED_PROJECT: &str = "server";
const ENV_DIR: &str = "env";
const PYTHON_DIR: &str = "python";
const UV_CACHE_DIR: &str = "uv-cache";
const STAMP_FILE: &str = ".runtime-stamp.json";

/// One sync at a time. Two concurrent `uv sync` runs against the same
/// environment would interleave writes into the same site-packages.
static INITIALIZING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RuntimeState {
    Uninitialized,
    Initializing,
    Ready,
    UpdateRequired,
    Broken,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatus {
    pub state: RuntimeState,
    /// Which executable would actually be used: the managed environment, a
    /// development checkout, or an explicit override.
    pub source: Option<String>,
    pub env_path: String,
    pub app_version: String,
    pub installed_version: Option<String>,
    /// What a sync would produce from the current bundle.
    pub expected_fingerprint: Option<String>,
    /// What the installed environment was built from.
    pub installed_fingerprint: Option<String>,
    /// Whether this build carries the resources needed to initialise.
    pub installable: bool,
    /// Why the state is what it is, when that is not obvious.
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum BootstrapEvent {
    Step { message: String },
    Output { line: String },
    Done,
    Failed { message: String },
}

#[derive(Debug, serde::Deserialize, Serialize)]
struct Stamp {
    fingerprint: String,
    app_version: String,
}

// -- locations ---------------------------------------------------------------

fn project_dir() -> PathBuf {
    paths::app_support_dir().join(PROJECT_DIR)
}

pub fn env_dir() -> PathBuf {
    paths::app_support_dir().join(ENV_DIR)
}

fn stamp_file() -> PathBuf {
    paths::app_support_dir().join(STAMP_FILE)
}

fn managed_executable() -> PathBuf {
    env_dir().join("bin").join(crate::server_cli::CLI_NAME)
}

/// The project staged inside the bundle, when this build carries one.
/// Whether this build carries its own Python project.
///
/// The discriminator between "a packaged application, which owns its runtime"
/// and "a bare development binary, which may borrow a checkout's".
pub fn is_packaged(app: &AppHandle) -> bool {
    bundled_project(app).is_some()
}

fn bundled_project(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .resolve(BUNDLED_PROJECT, tauri::path::BaseDirectory::Resource)
        .ok()
        .filter(|path| path.join("pyproject.toml").is_file())
}

// -- fingerprinting -----------------------------------------------------------

/// A content hash of everything that decides what a sync produces.
///
/// Deterministic: the same project always yields the same value, on any machine
/// and in any order, because the tree is walked sorted.
fn fingerprint(project: &Path, app_version: &str) -> Result<String, String> {
    let mut hasher = Sha256::new();
    hasher.update(app_version.as_bytes());
    hash_tree(project, project, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_tree(root: &Path, path: &Path, hasher: &mut Sha256) -> Result<(), String> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|entry| {
            // The stamp lives beside the project in some layouts and would make
            // the fingerprint depend on itself.
            entry.file_name().and_then(|n| n.to_str()) != Some(STAMP_FILE)
        })
        .collect();
    // Directory order is filesystem-dependent; sorting is what makes the hash
    // reproducible rather than merely stable on one machine.
    entries.sort();

    for entry in entries {
        let relative = entry.strip_prefix(root).unwrap_or(&entry);
        hasher.update(relative.to_string_lossy().as_bytes());
        if entry.is_dir() {
            hash_tree(root, &entry, hasher)?;
        } else {
            let bytes = std::fs::read(&entry)
                .map_err(|e| format!("cannot read {}: {e}", entry.display()))?;
            hasher.update(&bytes);
        }
    }
    Ok(())
}

fn read_stamp() -> Option<Stamp> {
    serde_json::from_str(&std::fs::read_to_string(stamp_file()).ok()?).ok()
}

// -- status -------------------------------------------------------------------

pub fn status(app: &AppHandle) -> RuntimeStatus {
    let app_version = app.package_info().version.to_string();
    let bundle = bundled_project(app);
    let expected = bundle
        .as_ref()
        .and_then(|project| fingerprint(project, &app_version).ok());
    let stamp = read_stamp();

    let mut status = RuntimeStatus {
        state: RuntimeState::Uninitialized,
        source: None,
        env_path: env_dir().display().to_string(),
        app_version,
        installed_version: stamp.as_ref().map(|s| s.app_version.clone()),
        expected_fingerprint: expected.clone(),
        installed_fingerprint: stamp.as_ref().map(|s| s.fingerprint.clone()),
        installable: bundle.is_some(),
        detail: None,
    };

    if INITIALIZING.load(Ordering::SeqCst) {
        status.state = RuntimeState::Initializing;
        return status;
    }

    // An explicit override short-circuits everything: someone set
    // QUANTUM_CODEX_COMMAND, which is an assertion that they know what it
    // resolves to.
    //
    // A development *checkout* used to short-circuit too, and that was the bug.
    // On the machine the app was built on, a packaged build found the
    // checkout's virtualenv, reported READY, and never created the managed
    // runtime -- so deleting Application Support did nothing observable and the
    // first-install path was never exercised by the person best placed to test
    // it. `server_cli` no longer offers the checkout to a packaged build, so
    // reaching here with a checkout means this is a development binary.
    if let Ok(cli) = server_cli::resolve() {
        status.source = Some(cli.source.clone());
        if cli.source != "managed environment" {
            status.state = RuntimeState::Ready;
            status.detail = Some(format!(
                "Using the {} at {}. This build does not need its managed runtime.",
                cli.source,
                cli.program.display()
            ));
            return status;
        }
    }

    if !managed_executable().is_file() {
        status.state = if status.installable {
            RuntimeState::Uninitialized
        } else {
            RuntimeState::Broken
        };
        if !status.installable {
            status.detail = Some(
                "No runtime is installed and this build carries no resources to install from."
                    .to_string(),
            );
        }
        return status;
    }

    // The environment exists. Whether it is current is a separate question.
    match (&expected, &status.installed_fingerprint) {
        (Some(expected), Some(installed)) if expected == installed => {
            status.state = RuntimeState::Ready;
        }
        (Some(_), Some(_)) => {
            status.state = RuntimeState::UpdateRequired;
            status.detail = Some(
                "The installed runtime was built from a different version of the server."
                    .to_string(),
            );
        }
        (Some(_), None) => {
            // Present but unaccounted for: we cannot tell what code it holds.
            status.state = RuntimeState::Broken;
            status.detail = Some(
                "A runtime is installed but its provenance is unknown, so it cannot be trusted \
                 to match this build. Rebuilding is safe: it touches neither models nor settings."
                    .to_string(),
            );
        }
        (None, _) => {
            // No bundle to compare against: trust what is installed.
            status.state = RuntimeState::Ready;
        }
    }

    status
}

// -- installation -------------------------------------------------------------

fn emit(app: &AppHandle, event: BootstrapEvent) {
    // A closed window must not make the installation fail.
    let _ = app.emit("bootstrap", event);
}

fn step(app: &AppHandle, message: &str) {
    emit(
        app,
        BootstrapEvent::Step {
            message: message.to_string(),
        },
    );
}

/// Install or rebuild the managed runtime.
pub async fn initialize(app: AppHandle) -> Result<(), String> {
    if INITIALIZING.swap(true, Ordering::SeqCst) {
        return Err("An initialisation is already running.".to_string());
    }

    let result = install(&app).await;
    INITIALIZING.store(false, Ordering::SeqCst);

    match &result {
        Ok(()) => emit(&app, BootstrapEvent::Done),
        Err(message) => emit(
            &app,
            BootstrapEvent::Failed {
                message: message.clone(),
            },
        ),
    }
    result
}

async fn install(app: &AppHandle) -> Result<(), String> {
    let Some(bundle) = bundled_project(app) else {
        return Err(
            "This build carries no bundled server project, so there is nothing to install. \
             In a development checkout, run `make install` from the repository root."
                .to_string(),
        );
    };
    let app_version = app.package_info().version.to_string();

    std::fs::create_dir_all(paths::app_support_dir())
        .map_err(|e| format!("cannot create the Application Support directory: {e}"))?;

    step(app, "Copying the server project…");
    copy_project(&bundle)?;

    let python = pinned_python()?;
    step(
        app,
        &format!("Installing Python {python} and the locked dependencies…"),
    );
    sync(app, &python).await?;

    if !managed_executable().is_file() {
        return Err(format!(
            "uv reported success but {} is missing.",
            managed_executable().display()
        ));
    }

    // Stamped only after the environment is proven to exist, so an interrupted
    // install leaves the runtime detectably incomplete rather than falsely
    // current.
    let stamp = Stamp {
        fingerprint: fingerprint(&bundle, &app_version)?,
        app_version,
    };
    std::fs::write(
        stamp_file(),
        serde_json::to_string_pretty(&stamp).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("cannot write {}: {e}", stamp_file().display()))?;

    step(app, "Reclaiming the download cache…");
    prune_cache(app).await;
    Ok(())
}

/// Replace the installed copy wholesale.
///
/// Merging would leave modules from an earlier version behind, and a stale
/// module is indistinguishable from a current one once imported.
fn copy_project(source: &Path) -> Result<(), String> {
    let destination = project_dir();
    if destination.exists() {
        std::fs::remove_dir_all(&destination)
            .map_err(|e| format!("cannot clear {}: {e}", destination.display()))?;
    }
    copy_tree(source, &destination)
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    std::fs::create_dir_all(destination)
        .map_err(|e| format!("cannot create {}: {e}", destination.display()))?;
    for entry in
        std::fs::read_dir(source).map_err(|e| format!("cannot read {}: {e}", source.display()))?
    {
        let entry = entry.map_err(|e| format!("directory read interrupted: {e}"))?;
        let target = destination.join(entry.file_name());
        if entry
            .file_type()
            .map_err(|e| format!("unknown file type: {e}"))?
            .is_dir()
        {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)
                .map_err(|e| format!("copying {} failed: {e}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// The interpreter version the project pins.
///
/// Handed to uv explicitly rather than left to its discovery, which depends on
/// the current directory — unpredictable for a sidecar launched from a `.app`.
/// Without it uv takes the newest interpreter satisfying `requires-python`, and
/// `uv.lock`, whose markers tell versions apart, then resolves a different
/// package set from the one that was tested.
fn pinned_python() -> Result<String, String> {
    let file = project_dir().join(".python-version");
    let raw = std::fs::read_to_string(&file)
        .map_err(|e| format!("{} is unreadable: {e}", file.display()))?;
    let version = raw.trim().to_string();
    if version.is_empty() {
        return Err(format!("{} is empty", file.display()));
    }
    Ok(version)
}

async fn sync(app: &AppHandle, python: &str) -> Result<(), String> {
    let command = app
        .shell()
        .sidecar("uv")
        .map_err(|e| format!("the uv sidecar is missing from this build: {e}"))?
        .args([
            "sync",
            // Exactly the locked set: re-resolving on a user's machine would
            // install a combination nobody tested.
            "--frozen",
            // pytest and ruff serve development only.
            "--no-dev",
            // Without this the installed package points back at the copied
            // project, which the next update replaces.
            "--no-editable",
            // uv brings its own CPython. The system interpreter is not ours to
            // depend on, and /usr/bin/python3 must never be used implicitly.
            "--managed-python",
            "--python",
        ])
        .arg(python)
        .arg("--project")
        .arg(project_dir().as_os_str())
        // Every location explicit: nothing lands in the user's caches or in the
        // project directory.
        .env("UV_PROJECT_ENVIRONMENT", env_dir())
        .env(
            "UV_PYTHON_INSTALL_DIR",
            paths::app_support_dir().join(PYTHON_DIR),
        )
        .env("UV_CACHE_DIR", paths::app_support_dir().join(UV_CACHE_DIR))
        // Ignore any ~/.config/uv that would change the resolution.
        .env("UV_NO_CONFIG", "1");

    let (mut events, _child) = command.spawn().map_err(|e| format!("cannot run uv: {e}"))?;

    let mut failure: Option<String> = None;
    while let Some(event) = events.recv().await {
        match event {
            // uv writes progress to stderr; both streams are relayed.
            CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                for line in String::from_utf8_lossy(&bytes)
                    .split(['\r', '\n'])
                    .filter(|line| !line.trim().is_empty())
                {
                    emit(
                        app,
                        BootstrapEvent::Output {
                            line: line.trim_end().to_string(),
                        },
                    );
                }
            }
            CommandEvent::Terminated(payload) => {
                if payload.code != Some(0) {
                    failure = Some(format!(
                        "uv sync failed (code {:?}, signal {:?})",
                        payload.code, payload.signal
                    ));
                }
            }
            CommandEvent::Error(message) => failure = Some(message),
            _ => {}
        }
    }

    match failure {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// Reclaim the download cache.
///
/// Not fatal on failure: it costs only disk, and the environment already works.
/// uv hard-links the environment to the cache, so dropping a cache entry removes
/// one link and leaves the installed files reachable.
async fn prune_cache(app: &AppHandle) {
    let Ok(command) = app.shell().sidecar("uv") else {
        return;
    };
    let command = command
        .args(["cache", "prune", "--ci"])
        .env("UV_CACHE_DIR", paths::app_support_dir().join(UV_CACHE_DIR))
        .env("UV_NO_CONFIG", "1");
    if let Ok((mut events, _child)) = command.spawn() {
        while events.recv().await.is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, name: &str, contents: &str) {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn project(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("qc-fingerprint-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write(&root, "pyproject.toml", "[project]\nname='quantum-codex'\n");
        write(&root, "uv.lock", "version = 1\n");
        write(&root, ".python-version", "3.12\n");
        write(&root, "quantum_codex/app.py", "print('a')\n");
        root
    }

    #[test]
    fn the_same_project_always_hashes_the_same() {
        let root = project("stable");
        assert_eq!(
            fingerprint(&root, "0.1.0").unwrap(),
            fingerprint(&root, "0.1.0").unwrap()
        );
    }

    #[test]
    fn changed_source_changes_the_fingerprint() {
        // The failure this exists for: code edited without a version bump left
        // the environment looking current, so the app kept serving old code.
        let root = project("source");
        let before = fingerprint(&root, "0.1.0").unwrap();
        write(&root, "quantum_codex/app.py", "print('b')\n");

        assert_ne!(before, fingerprint(&root, "0.1.0").unwrap());
    }

    #[test]
    fn a_changed_lock_changes_the_fingerprint() {
        let root = project("lock");
        let before = fingerprint(&root, "0.1.0").unwrap();
        write(&root, "uv.lock", "version = 2\n");

        assert_ne!(before, fingerprint(&root, "0.1.0").unwrap());
    }

    #[test]
    fn a_changed_app_version_changes_the_fingerprint() {
        let root = project("version");

        assert_ne!(
            fingerprint(&root, "0.1.0").unwrap(),
            fingerprint(&root, "0.2.0").unwrap()
        );
    }

    #[test]
    fn touching_a_file_without_changing_it_does_not() {
        // Where an mtime comparison would report a spurious update, and force a
        // download for nothing.
        let root = project("touch");
        let before = fingerprint(&root, "0.1.0").unwrap();
        let path = root.join("uv.lock");
        let contents = std::fs::read(&path).unwrap();
        std::fs::write(&path, contents).unwrap();

        assert_eq!(before, fingerprint(&root, "0.1.0").unwrap());
    }

    #[test]
    fn the_runtime_owns_only_its_own_directories() {
        // Repair must cost a download, never a user's profiles or settings.
        for owned in [PROJECT_DIR, ENV_DIR, PYTHON_DIR, UV_CACHE_DIR] {
            assert!(!["profiles.json", "settings.json", "runtime.json", "logs"].contains(&owned));
        }
    }
}

#[cfg(test)]
mod bundle_layout_tests {
    use super::BUNDLED_PROJECT;

    /// The staged project must be looked for where the bundle actually puts it.
    ///
    /// Two files have to agree and neither imports the other, so nothing but a
    /// test can hold them together. They did not agree: the code asked for
    /// `resources/server` while `tauri.conf.json` stages `server/`, so a
    /// packaged build reported that it carried no resources to install from.
    #[test]
    fn bundled_resource_matches_the_bundle_configuration() {
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tauri.conf.json"))
                .expect("tauri.conf.json must be readable"),
        )
        .expect("tauri.conf.json must be valid JSON");

        let resources = config["bundle"]["resources"]
            .as_object()
            .expect("bundle.resources must be an object of source -> destination");

        let destinations: Vec<&str> = resources
            .values()
            .filter_map(serde_json::Value::as_str)
            .collect();

        assert!(
            destinations
                .iter()
                .any(
                    |destination| destination.trim_end_matches('/') == BUNDLED_PROJECT
                        || destination.starts_with(&format!("{BUNDLED_PROJECT}/"))
                ),
            "no bundled resource lands in {BUNDLED_PROJECT:?}; destinations are {destinations:?}"
        );
    }
}
