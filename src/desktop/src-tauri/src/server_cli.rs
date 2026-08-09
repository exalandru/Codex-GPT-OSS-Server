//! Locating the headless server CLI.
//!
//! A macOS application launched from Finder or the Dock inherits launchd's
//! environment, not a shell's. `PATH` there is roughly `/usr/bin:/bin` — it
//! contains neither `~/.local/bin` nor any project virtualenv. So looking up
//! `quantum-codex-server` by name works when the app is started from a terminal
//! and fails with `ENOENT` the moment a user double-clicks it, which is exactly
//! the wrong way round for a desktop application.
//!
//! Everything here therefore resolves to an **absolute path that is verified to
//! exist** before anything is spawned, and every consumer — start, restart and
//! the diagnostic — goes through the same function, so they cannot disagree
//! about which server they are talking about.
//!
//! A failure names every location that was tried. `No such file or directory`
//! tells a user nothing they can act on; a list of paths tells them whether
//! they forgot `make install` or are running an app whose environment was never
//! bootstrapped.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Serialize;

use crate::paths;

/// The executable name inside a Python virtualenv's `bin`.
///
/// Must match the canonical `[project.scripts]` entry in `src/server/pyproject.toml`.
/// The alias `qcs` installs beside it and runs the same code, but resolution
/// looks for exactly one name so a half-installed environment fails loudly
/// rather than silently picking whichever name happens to exist.
pub(crate) const CLI_NAME: &str = "quantum-codex-server";

/// Where a packaged application keeps its own managed Python environment.
///
/// Under application support rather than inside the bundle: the bundle is
/// read-only once signed, and the environment has to be created on first run.
const MANAGED_ENV_DIR: &str = "env";

/// The repository this binary was compiled in.
///
/// Baked in at build time so a development build can find the checkout's
/// virtualenv without being told where it is. It is always checked for
/// existence before use, so a distributed binary — where this path means
/// nothing — simply falls through to the next candidate.
/// Debug builds only. A release binary is what ships, and `CARGO_MANIFEST_DIR`
/// is an absolute path on the machine that built it — baking it into a public
/// artifact publishes the author's directory layout to every user for no
/// benefit, since a packaged build already refuses the checkout at runtime.
///
/// The consequence is deliberate: `cargo run --release` from a checkout no
/// longer finds the checkout's virtualenv. Set `QUANTUM_CODEX_COMMAND`, which
/// is what the Makefile does anyway.
#[cfg(debug_assertions)]
const CRATE_DIR: Option<&str> = Some(env!("CARGO_MANIFEST_DIR"));
#[cfg(not(debug_assertions))]
const CRATE_DIR: Option<&str> = None;

/// Whether this build carries its own Python project and therefore owns its
/// runtime.
///
/// Set once at startup. It decides whether the development checkout is a
/// legitimate place to find the server, and the answer is no for a packaged
/// application: `CRATE_DIR` still points at the machine the `.app` was built
/// on, so on a developer's own machine a packaged build would silently run the
/// checkout's virtualenv. It would then appear to work perfectly while the
/// managed runtime was never built — and fail on every machine that has no
/// checkout, which is every user's.
///
/// Defaults to allowing the checkout, so a bare `cargo run` with no staged
/// resources still finds a server.
static PACKAGED: OnceLock<bool> = OnceLock::new();

/// Declare whether this build carries bundled server resources.
pub fn set_packaged(packaged: bool) {
    let _ = PACKAGED.set(packaged);
}

fn packaged() -> bool {
    *PACKAGED.get().unwrap_or(&false)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerCli {
    /// Absolute, and verified to exist.
    pub program: PathBuf,
    /// Arguments that must precede the subcommand.
    pub args: Vec<String>,
    /// Which candidate matched, for the diagnostic.
    pub source: String,
}

impl ServerCli {
    /// The full argument list for one subcommand invocation.
    pub fn argv(&self, subcommand: &[&str]) -> Vec<String> {
        let mut argv = self.args.clone();
        argv.extend(subcommand.iter().map(|s| s.to_string()));
        argv
    }
}

/// The development checkout's root, if this binary was built inside one.
fn checkout_root() -> Option<PathBuf> {
    // <repo>/src/desktop/src-tauri -> <repo>
    Path::new(CRATE_DIR?)
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .filter(|root| root.join("src").join("server").is_dir())
}

/// An explicit override, for the Makefile and for anyone wiring this by hand.
///
/// Split on whitespace so a wrapper like `uv run --project … quantum-codex-server`
/// can be expressed. The program is resolved to an absolute path only when it
/// already is one; a bare name is trusted, because someone who set this
/// variable is asserting they know it resolves.
fn from_override() -> Option<ServerCli> {
    let raw = std::env::var("QUANTUM_CODEX_COMMAND").ok()?;
    let mut parts = raw.split_whitespace();
    let program = PathBuf::from(parts.next()?);
    if program.is_absolute() && !program.is_file() {
        return None;
    }
    Some(ServerCli {
        program,
        args: parts.map(str::to_string).collect(),
        source: "QUANTUM_CODEX_COMMAND".to_string(),
    })
}

fn venv_cli(venv: &Path, source: &str) -> Option<ServerCli> {
    let program = venv.join("bin").join(CLI_NAME);
    program.is_file().then(|| ServerCli {
        program,
        args: Vec::new(),
        source: source.to_string(),
    })
}

/// Every place a server CLI could legitimately be, in priority order.
fn candidates() -> Vec<(String, Option<ServerCli>)> {
    candidates_for(packaged())
}

/// The policy, with the packaged/development decision passed in.
///
/// Separated so the rule can be tested without a process-wide `OnceLock`, which
/// a test can only set once.
fn candidates_for(packaged: bool) -> Vec<(String, Option<ServerCli>)> {
    let managed = paths::app_support_dir().join(MANAGED_ENV_DIR);
    let mut found = vec![
        ("QUANTUM_CODEX_COMMAND".to_string(), from_override()),
        (
            format!(
                "{} (managed environment)",
                managed.join("bin").join(CLI_NAME).display()
            ),
            venv_cli(&managed, "managed environment"),
        ),
    ];

    // A packaged application never falls back to a checkout: doing so is what
    // let a build with no managed runtime look healthy on the one machine where
    // it could not possibly be representative.
    if !packaged {
        if let Some(root) = checkout_root() {
            for candidate in [
                root.join(".venv"),
                root.join("src").join("server").join(".venv"),
            ] {
                found.push((
                    format!(
                        "{} (development checkout)",
                        candidate.join("bin").join(CLI_NAME).display()
                    ),
                    venv_cli(&candidate, "development checkout"),
                ));
            }
        }
    }

    found
}

/// The one resolution path, shared by start, restart and the diagnostic.
pub fn resolve() -> Result<ServerCli, String> {
    let candidates = candidates();
    for (_, found) in &candidates {
        if let Some(cli) = found {
            return Ok(cli.clone());
        }
    }

    let tried: Vec<String> = candidates
        .iter()
        .map(|(description, _)| format!("  - {description}"))
        .collect();

    Err(format!(
        "The Quantum Codex server executable was not found.\n\nLooked in:\n{}\n\n\
         In a development checkout, run `make install` from the repository root. \
         In an installed application, its Python environment has not been set up yet.",
        tried.join("\n")
    ))
}

/// What `resolve` would pick, and everything it considered.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnosis {
    pub resolved: Option<ServerCli>,
    pub searched: Vec<SearchedLocation>,
    /// The `PATH` this process actually inherited, which is the thing people
    /// assume and should be able to check.
    pub inherited_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchedLocation {
    pub description: String,
    pub found: bool,
}

pub fn diagnose() -> Diagnosis {
    let candidates = candidates();
    Diagnosis {
        resolved: candidates.iter().find_map(|(_, found)| found.clone()),
        searched: candidates
            .iter()
            .map(|(description, found)| SearchedLocation {
                description: description.clone(),
                found: found.is_some(),
            })
            .collect(),
        inherited_path: std::env::var("PATH").unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_development_checkout_is_recognised() {
        // This test only runs from the checkout, which is precisely the case
        // the development candidate exists for.
        let root = checkout_root().expect("tests run inside the checkout");
        assert!(root.join("src").join("server").is_dir());
    }

    #[test]
    fn a_release_binary_carries_no_path_from_the_machine_that_built_it() {
        // `cargo test` is a debug build, so this asserts the compile-time rule
        // rather than the value: whatever ships must not embed the author's
        // checkout path, and the only thing keeping that true is the cfg.
        assert_eq!(CRATE_DIR.is_some(), cfg!(debug_assertions));
    }

    #[test]
    fn an_absolute_override_that_does_not_exist_is_rejected() {
        // Trusting it would reproduce the ENOENT this module exists to prevent,
        // just with a longer path in the message.
        temp_env("/definitely/not/here/quantum-codex-server", || {
            assert!(from_override().is_none());
        });
    }

    #[test]
    fn an_override_may_carry_arguments() {
        temp_env("uv run --project /srv quantum-codex-server", || {
            let cli = from_override().expect("a bare program name is trusted");
            assert_eq!(cli.program, PathBuf::from("uv"));
            assert_eq!(
                cli.args,
                vec!["run", "--project", "/srv", "quantum-codex-server"]
            );
            assert_eq!(cli.argv(&["serve"]).last().unwrap(), "serve");
        });
    }

    #[test]
    fn a_failure_names_every_location_it_tried() {
        // The whole point: ENOENT is unactionable, a list of paths is not.
        temp_env("/definitely/not/here/quantum-codex-server", || {
            let message = match resolve() {
                Ok(_) => return, // a real checkout resolves; nothing to assert
                Err(message) => message,
            };
            assert!(message.contains("Looked in:"));
            assert!(message.contains("make install"));
        });
    }

    fn temp_env(value: &str, body: impl FnOnce()) {
        let previous = std::env::var("QUANTUM_CODEX_COMMAND").ok();
        unsafe { std::env::set_var("QUANTUM_CODEX_COMMAND", value) };
        body();
        match previous {
            Some(value) => unsafe { std::env::set_var("QUANTUM_CODEX_COMMAND", value) },
            None => unsafe { std::env::remove_var("QUANTUM_CODEX_COMMAND") },
        }
    }
}

#[cfg(test)]
mod packaging_policy_tests {
    use super::*;

    fn described(candidates: &[(String, Option<ServerCli>)]) -> String {
        candidates
            .iter()
            .map(|(description, _)| description.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The bug this rule exists for.
    ///
    /// `CRATE_DIR` is baked in at compile time, so on the machine an `.app` was
    /// built on it still points at a real checkout with a real virtualenv. A
    /// packaged build that accepted it would run the developer's environment,
    /// report READY, and never create the managed runtime -- looking perfect on
    /// the one machine where the packaged path most needed testing, and failing
    /// on every machine without a checkout.
    #[test]
    fn a_packaged_build_never_offers_the_development_checkout() {
        let text = described(&candidates_for(true));

        assert!(
            !text.contains("development checkout"),
            "a packaged build must not look in a checkout:\n{text}"
        );
    }

    #[test]
    fn a_packaged_build_looks_in_the_managed_environment() {
        let text = described(&candidates_for(true));

        assert!(text.contains("managed environment"), "{text}");
    }

    #[test]
    fn a_development_build_may_still_use_a_checkout() {
        // Only when this binary was actually compiled inside one; a distributed
        // binary has nothing to find.
        if checkout_root().is_none() {
            return;
        }
        let text = described(&candidates_for(false));

        assert!(text.contains("development checkout"), "{text}");
    }

    #[test]
    fn the_managed_environment_outranks_a_checkout() {
        // Order is the policy: when both exist, the runtime this build owns is
        // the one it uses.
        let candidates = candidates_for(false);
        let managed = candidates
            .iter()
            .position(|(d, _)| d.contains("managed environment"));
        let checkout = candidates
            .iter()
            .position(|(d, _)| d.contains("development checkout"));

        if let (Some(managed), Some(checkout)) = (managed, checkout) {
            assert!(managed < checkout, "{}", described(&candidates));
        }
    }

    #[test]
    fn an_explicit_override_outranks_everything() {
        assert!(candidates_for(true)[0].0.contains("QUANTUM_CODEX_COMMAND"));
        assert!(candidates_for(false)[0].0.contains("QUANTUM_CODEX_COMMAND"));
    }

    #[test]
    fn the_managed_environment_is_under_application_support() {
        // Deterministic location: deleting Application Support must genuinely
        // remove the runtime, which is only true if it lives there.
        let text = described(&candidates_for(true));
        let expected = paths::app_support_dir().join(MANAGED_ENV_DIR);

        assert!(text.contains(&expected.display().to_string()), "{text}");
    }

    #[test]
    fn a_missing_managed_environment_yields_no_executable() {
        // What a factory-clean packaged install looks like: the candidate is
        // listed, and nothing is found, so bootstrap has to run.
        let managed = paths::app_support_dir().join(MANAGED_ENV_DIR);
        if managed.join("bin").join(CLI_NAME).is_file() {
            return; // a runtime is installed on this machine; nothing to assert
        }
        let found = candidates_for(true)
            .into_iter()
            .filter(|(_, cli)| cli.is_some())
            .any(|(_, cli)| cli.unwrap().source == "managed environment");

        assert!(!found);
    }
}
