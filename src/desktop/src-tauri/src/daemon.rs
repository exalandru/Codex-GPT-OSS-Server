//! Finding, querying and controlling the server daemon.
//!
//! The daemon owns itself (D1). The desktop app never holds a child handle: it
//! spawns the server in its own session and then talks to whatever is described
//! by `runtime.json`. Closing the window therefore cannot take the server with
//! it, which is the whole point — a Codex session runs for hours and must not
//! depend on a window staying open.
//!
//! Everything the interface displays comes back as opaque JSON. This file knows
//! how to *reach* the management plane, never what its responses mean, so the
//! server stays the single source of truth for capabilities, profiles and
//! configuration.

use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::paths;
use crate::server_cli::{self, ServerCli};

/// The default endpoint, used when `runtime.json` cannot tell us where to look.
///
/// Matches the server's own default. It is a *candidate to probe*, never an
/// assumption that something is there.
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8123;

/// What a running server publishes about itself.
///
/// Only the fields needed to reach it and to report liveness are named. The
/// rest of `runtime.json` is ignored rather than mirrored.
///
/// Read field by field rather than through a derived `Deserialize`, because a
/// single unusable field must not cost us the others. The file exists to say
/// *how to reach the daemon*; losing the endpoint and the token because some
/// unrelated field changed shape is how a healthy daemon came to be reported as
/// stopped.
#[derive(Debug, Clone)]
pub struct RuntimeState {
    pub pid: i32,
    pub host: String,
    pub port: u16,
    /// Which model was resident *when the file was written*. Advisory only: the
    /// daemon loads on demand, so this goes stale the moment a request names a
    /// different one. `/internal/status` is the authority, which is why nothing
    /// reads this field -- it is kept so the shape stays documented.
    #[allow(dead_code)]
    pub model: Option<String>,
    pub management_token: String,
}

impl RuntimeState {
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Whether the recorded process still exists.
    ///
    /// Existence only: a pid can be reused, so this is a cheap negative test.
    /// Confirmation comes from actually reaching the management plane.
    pub fn process_alive(&self) -> bool {
        // Signal 0 performs the permission and existence checks without
        // delivering anything.
        unsafe { libc::kill(self.pid, 0) == 0 || *libc::__error() == libc::EPERM }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Discovery {
    /// `true` only when the daemon actually answered a request.
    pub connected: bool,
    pub endpoint: Option<String>,
    pub model: Option<String>,
    /// Why the daemon could not be reached, in words the interface can show.
    pub detail: Option<String>,
    /// A problem with `runtime.json` itself. Deliberately separate from
    /// `connected`: a metadata file that cannot be read says nothing about
    /// whether the daemon is healthy, and conflating the two is what showed a
    /// live server as stopped.
    pub metadata_error: Option<String>,
    /// The daemon was found by probing rather than through `runtime.json`.
    pub adopted: bool,
    /// Whether management actions are available. They need the token, which
    /// only `runtime.json` carries; an adopted daemon can be watched in full
    /// but not commanded.
    pub manageable: bool,
}

/// Is the thing listening on `base` our daemon?
///
/// `/health` needs no token, and its body is distinctive: a `lifecycle` object
/// and a `prompt_cache` object together are not something another service on
/// this port would answer with. Positive identification matters -- adopting
/// whatever happens to hold the port would point the interface at a stranger.
async fn identify(base: &str) -> Option<Value> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .ok()?;
    let body: Value = client
        .get(format!("{base}/health"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    let ours = body.get("lifecycle").is_some_and(Value::is_object)
        && body.get("prompt_cache").is_some_and(Value::is_object);
    ours.then_some(body)
}

fn model_of(health: &Value) -> Option<String> {
    health
        .get("lifecycle")
        .and_then(|l| l.get("model"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Parse a runtime file tolerantly.
///
/// Every field is read individually. A file missing or mangling one of them
/// still yields whatever else it contains, and only the genuinely
/// indispensable ones -- host, port and token, the three needed to *reach and
/// authenticate against* the daemon -- can make this fail.
pub fn parse_runtime_state(text: &str) -> Result<RuntimeState, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("not valid JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "not a JSON object".to_string())?;

    let host = object
        .get("host")
        .and_then(Value::as_str)
        .ok_or_else(|| "no usable `host`".to_string())?
        .to_string();
    let port = object
        .get("port")
        .and_then(Value::as_u64)
        .and_then(|p| u16::try_from(p).ok())
        .ok_or_else(|| "no usable `port`".to_string())?;
    let management_token = object
        .get("management_token")
        .and_then(Value::as_str)
        .ok_or_else(|| "no usable `management_token`".to_string())?
        .to_string();

    Ok(RuntimeState {
        // A missing pid costs only the cheap liveness pre-check; probing still
        // settles it. 0 never matches a real process, so it reads as "unknown".
        pid: object.get("pid").and_then(Value::as_i64).unwrap_or(0) as i32,
        host,
        port,
        // Null is normal: a daemon may hold no model at all.
        model: object
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        management_token,
    })
}

pub fn read_runtime_state() -> Result<Option<RuntimeState>, String> {
    let path = paths::runtime_file();
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse_runtime_state(&text)
        .map(Some)
        .map_err(|detail| format!("{} is unusable: {detail}", path.display()))
}

async fn management_request(method: reqwest::Method, path: &str) -> Result<Value, String> {
    let Some(state) = read_runtime_state()? else {
        return Err("No server is running.".to_string());
    };
    if !state.process_alive() {
        return Err(format!(
            "The runtime file names pid {}, which is gone. The server exited without cleaning up.",
            state.pid
        ));
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .request(method, format!("{}{}", state.base_url(), path))
        .bearer_auth(&state.management_token)
        .send()
        .await
        .map_err(|e| format!("cannot reach {}: {e}", state.base_url()))?;

    if !response.status().is_success() {
        return Err(format!(
            "management request failed: HTTP {}",
            response.status()
        ));
    }
    response.json::<Value>().await.map_err(|e| e.to_string())
}

async fn management_request_with_body(
    method: reqwest::Method,
    path: &str,
    body: Value,
) -> Result<Value, String> {
    let Some(state) = read_runtime_state()? else {
        return Err("No server is running.".to_string());
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .request(method, format!("{}{}", state.base_url(), path))
        .bearer_auth(&state.management_token)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("cannot reach {}: {e}", state.base_url()))?;

    let status = response.status();
    let payload: Value = response.json().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        // The server explains *why* a directory was refused, and that reason is
        // the entire value of validating at import time.
        let message = payload
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("the request was refused");
        return Err(message.to_string());
    }
    Ok(payload)
}

/// Is a daemon there, and can we actually talk to it?
/// Find the daemon, whether or not `runtime.json` is usable.
///
/// Two independent routes, in order of authority:
///
/// 1. `runtime.json` names an endpoint and a token -> reach the management
///    plane. This is the full-fidelity path: status and control.
/// 2. Otherwise probe the default endpoint and *positively identify* the
///    listener as ours. A daemon found this way is watched, not commanded,
///    because the token lives in the file we could not read.
///
/// A runtime file that cannot be parsed is reported as a metadata problem and
/// never, on its own, as "stopped": the daemon owns itself and outlives this
/// app, so its health is established by asking it, not by reading a file.
pub async fn discover() -> Discovery {
    let (state, metadata_error) = match read_runtime_state() {
        Ok(state) => (state, None),
        Err(detail) => (None, Some(detail)),
    };

    if let Some(state) = &state {
        let endpoint = state.base_url();
        match management_request(reqwest::Method::GET, "/internal/status").await {
            Ok(status) => {
                return Discovery {
                    connected: true,
                    endpoint: Some(endpoint),
                    model: status
                        .get("lifecycle")
                        .and_then(|l| l.get("model"))
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    manageable: true,
                    metadata_error,
                    ..Default::default()
                };
            }
            Err(detail) => {
                // The file pointed somewhere that did not answer. Fall through
                // to probing: the daemon may have been restarted on the same
                // port with a file we raced.
                if let Some(health) = identify(&endpoint).await {
                    return Discovery {
                        connected: true,
                        endpoint: Some(endpoint),
                        model: model_of(&health),
                        adopted: true,
                        detail: Some(format!(
                            "Reconnected by probing; management is unavailable because the \
                             runtime file no longer matches this server ({detail})."
                        )),
                        metadata_error,
                        ..Default::default()
                    };
                }
            }
        }
    }

    // No usable file, or it pointed at nothing. Probe the default endpoint.
    let fallback = format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}");
    if let Some(health) = identify(&fallback).await {
        return Discovery {
            connected: true,
            endpoint: Some(fallback),
            model: model_of(&health),
            adopted: true,
            detail: Some(
                "Reconnected to a running Quantum Codex server by probing. Management \
                 actions need its runtime file, which is missing or unreadable; restart \
                 the server to restore them."
                    .to_string(),
            ),
            metadata_error,
            ..Default::default()
        };
    }

    Discovery {
        connected: false,
        detail: Some(match &metadata_error {
            // Said precisely: the file is broken *and* nothing is answering.
            // Either half alone would be the wrong conclusion.
            Some(_) => "No server answered, and the runtime file is unusable.".to_string(),
            None => "No server is running.".to_string(),
        }),
        metadata_error,
        ..Default::default()
    }
}

pub async fn status() -> Result<Value, String> {
    management_request(reqwest::Method::GET, "/internal/status").await
}

pub async fn clear_cache() -> Result<Value, String> {
    management_request(reqwest::Method::DELETE, "/internal/cache").await
}

/// Release the resident model, leaving the daemon running.
///
/// Sent with a body so the server's own refusal reaches the user: "Model is
/// currently in use" is the actionable half of a 409, and replacing it with
/// "HTTP 409" would throw away the only part worth reading.
pub async fn unload_model() -> Result<Value, String> {
    management_request_with_body(
        reqwest::Method::POST,
        "/internal/model/unload",
        serde_json::json!({}),
    )
    .await
}

/// Per-request diagnostics and their aggregates.
///
/// Interpretation — medians, ratios, throughput — is computed by the server, so
/// no client has to and none can disagree.
pub async fn request_diagnostics(limit: u32) -> Result<Value, String> {
    management_request(
        reqwest::Method::GET,
        &format!("/internal/requests?limit={limit}"),
    )
    .await
}

pub async fn download_status() -> Result<Value, String> {
    management_request(reqwest::Method::GET, "/internal/downloads").await
}

/// Make sure a daemon is answering, starting one if necessary.
///
/// Downloads are server-owned by design: one machine, one downloader, one set
/// of preflight and resume rules. That is an architecture decision, not
/// something a user should have to know -- being told to visit another view and
/// press Start before a download is the implementation showing through.
///
/// Starting the daemon does **not** load a model. The daemon serves with
/// nothing resident and loads on demand, so this costs a socket, not sixty
/// gigabytes of weights.
pub async fn ensure_running() -> Result<(), String> {
    if discover().await.connected {
        return Ok(());
    }

    // `start` refuses if something is already there, and reports a foreign
    // listener by name. Its errors are the actionable ones, so they are passed
    // through rather than replaced with "could not start".
    start(None).await?;

    // Poll rather than sleep a fixed time: the socket appears in well under a
    // second, and waiting longer than necessary is the delay this removes.
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        if discover().await.connected {
            return Ok(());
        }
    }
    Err("The server was started but did not answer in time. Check the Logs view.".to_string())
}

pub async fn start_download(repo: &str, destination: Option<String>) -> Result<Value, String> {
    // Started on demand, so Download works from a cold app.
    ensure_running().await?;
    let mut body = serde_json::json!({ "repo": repo });
    if let Some(destination) = destination {
        body["destination"] = Value::String(destination);
    }
    management_request_with_body(reqwest::Method::POST, "/internal/downloads", body).await
}

pub async fn cancel_download() -> Result<Value, String> {
    management_request(reqwest::Method::DELETE, "/internal/downloads").await
}

/// Start the server, detached, and let it publish its own runtime file.
///
/// `process_group(0)` puts it in a new process group, so a signal sent to this
/// application's group — or this application exiting — does not take it with
/// it (D1). Output goes to a log file because a detached process has nowhere
/// else to write, and the interface needs something to show.
pub async fn start(profile: Option<String>) -> Result<(), String> {
    // Resolved and verified before anything is spawned, so a missing
    // environment reports where it looked instead of ENOENT.
    let cli: ServerCli = server_cli::resolve()?;

    // Never spawn alongside a daemon that is already answering. The check is a
    // *probe*, not a reading of the runtime file: an unusable file is exactly
    // the case where a second daemon would be started, collide on the port, and
    // die with "address already in use" while the first one carried on serving.
    let found = discover().await;
    if found.connected {
        return Err(format!(
            "A Quantum Codex server is already running on {}.",
            found.endpoint.unwrap_or_else(|| "this machine".to_string())
        ));
    }

    // Nothing of ours answered. If the port is nevertheless occupied, say so
    // rather than letting uvicorn fail with a message only the log carries.
    if tokio::net::TcpStream::connect((DEFAULT_HOST, DEFAULT_PORT))
        .await
        .is_ok()
    {
        return Err(format!(
            "Port {DEFAULT_PORT} is in use by something that is not a Quantum Codex server. \
             Stop it, or configure a different port in the profile."
        ));
    }

    std::fs::create_dir_all(paths::logs_dir()).map_err(|e| e.to_string())?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths::server_log_file())
        .map_err(|e| format!("cannot open the log file: {e}"))?;
    let errors = log.try_clone().map_err(|e| e.to_string())?;

    // Arguments are passed as a vector, never interpolated into a shell
    // command: a model path containing a space would otherwise become an
    // argument injection (cahier 42).
    let mut argv = cli.argv(&["serve"]);
    if let Some(profile) = profile {
        argv.push("--profile".to_string());
        argv.push(profile);
    }

    tokio::process::Command::new(&cli.program)
        .args(&argv)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(errors))
        .process_group(0) // its own session: quitting this app leaves it alone
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", cli.program.display()))?;

    Ok(())
}

/// Ask the daemon to stop, escalating only if it will not.
///
/// SIGTERM lets uvicorn shut down gracefully, which is what removes the runtime
/// file and unloads the model. SIGKILL is the fallback for a process that is
/// wedged; it leaves a stale runtime file behind, which every reader already
/// has to tolerate.
pub async fn stop() -> Result<(), String> {
    let Some(state) = read_runtime_state()? else {
        return Err("No server is running.".to_string());
    };
    if !state.process_alive() {
        // Nothing to signal, but the file is lying. Removing it is the honest
        // outcome rather than reporting a failure the user cannot act on.
        let _ = std::fs::remove_file(paths::runtime_file());
        return Ok(());
    }

    unsafe { libc::kill(state.pid, libc::SIGTERM) };

    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        if !state.process_alive() {
            return Ok(());
        }
    }

    unsafe { libc::kill(state.pid, libc::SIGKILL) };
    tokio::time::sleep(Duration::from_millis(500)).await;
    let _ = std::fs::remove_file(paths::runtime_file());
    Ok(())
}

/// The last `lines` of server output.
pub fn tail_log(lines: usize) -> Result<Vec<String>, String> {
    let path = paths::server_log_file();
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let all: Vec<&str> = text.lines().collect();
    let start = all.len().saturating_sub(lines);
    Ok(all[start..].iter().map(|s| s.to_string()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The desktop app reads `runtime.json` to find the daemon, so it must
    /// tolerate the file the Python side actually writes — including fields it
    /// does not care about. Naming every field here would make an unrelated
    /// server change break the app.
    #[test]
    fn extra_fields_in_the_runtime_file_are_ignored() {
        let json = r#"{
            "version": 1,
            "pid": 4321,
            "host": "127.0.0.1",
            "port": 8123,
            "model": "gpt-oss-20b",
            "management_token": "tok",
            "started_at": 1.5,
            "something_added_later": true
        }"#;

        let state = parse_runtime_state(json).expect("should parse");

        assert_eq!(state.pid, 4321);
        assert_eq!(state.base_url(), "http://127.0.0.1:8123");
    }

    #[test]
    fn a_truncated_runtime_file_is_an_error_not_a_default() {
        // A half-written file must not read as a server on port 0.
        assert!(parse_runtime_state(r#"{"pid": 1}"#).is_err());
    }

    #[test]
    fn our_own_process_counts_as_alive() {
        let state = RuntimeState {
            pid: std::process::id() as i32,
            host: "127.0.0.1".into(),
            port: 8123,
            model: Some("m".to_string()),
            management_token: "t".into(),
        };

        assert!(state.process_alive());
    }

    #[test]
    fn a_pid_above_the_ceiling_is_not_alive() {
        // What a stale file left by a crash looks like.
        let state = RuntimeState {
            pid: 1 << 22,
            host: "127.0.0.1".into(),
            port: 8123,
            model: Some("m".to_string()),
            management_token: "t".into(),
        };

        assert!(!state.process_alive());
    }
}

#[cfg(test)]
mod runtime_file_tests {
    use super::*;

    /// The exact bytes a running daemon writes today.
    ///
    /// `model` is null because a daemon holds no weights until a request names
    /// one. An earlier build declared this field `String`, so serde rejected
    /// the whole file and a healthy server was reported as stopped.
    const OBSERVED: &str = r#"{
  "version": 1,
  "pid": 91123,
  "host": "127.0.0.1",
  "port": 8123,
  "model": null,
  "management_token": "DcFbjrk7UcTv-I5iOI9g_RQ4Z5OFsKEpbIX0WfYSS-A",
  "started_at": 1786222317.626543
}"#;

    #[test]
    fn the_runtime_file_a_daemon_actually_writes_is_readable() {
        let state = parse_runtime_state(OBSERVED).expect("the observed file must parse");

        assert_eq!(state.host, "127.0.0.1");
        assert_eq!(state.port, 8123);
        assert_eq!(state.pid, 91123);
        assert!(state.model.is_none());
        assert_eq!(state.base_url(), "http://127.0.0.1:8123");
    }

    #[test]
    fn a_null_model_is_a_normal_state_not_a_parse_failure() {
        // The regression itself: "invalid type: null, expected a string".
        assert!(parse_runtime_state(OBSERVED).is_ok());
    }

    #[test]
    fn a_named_model_is_still_read() {
        let text = OBSERVED.replace("\"model\": null", "\"model\": \"gpt-oss-120b\"");
        let state = parse_runtime_state(&text).unwrap();

        assert_eq!(state.model.as_deref(), Some("gpt-oss-120b"));
    }

    #[test]
    fn an_unknown_future_field_does_not_make_the_file_unusable() {
        let text = OBSERVED.replace("\"version\": 1,", "\"version\": 1, \"invented_later\": {},");

        assert!(parse_runtime_state(&text).is_ok());
    }

    #[test]
    fn a_missing_pid_costs_only_the_liveness_pre_check() {
        // Reaching the daemon settles liveness anyway; losing the endpoint and
        // the token over a missing pid would not.
        let text = OBSERVED.replace("\"pid\": 91123,", "");
        let state = parse_runtime_state(&text).unwrap();

        assert_eq!(state.pid, 0);
        assert_eq!(state.port, 8123);
    }

    #[test]
    fn only_the_fields_needed_to_reach_the_daemon_are_indispensable() {
        for missing in ["\"host\": \"127.0.0.1\",", "\"port\": 8123,"] {
            let text = OBSERVED.replace(missing, "");
            assert!(
                parse_runtime_state(&text).is_err(),
                "a file without {missing} cannot be acted on"
            );
        }
        let text = OBSERVED.replace(
            "\"management_token\": \"DcFbjrk7UcTv-I5iOI9g_RQ4Z5OFsKEpbIX0WfYSS-A\",",
            "",
        );
        assert!(parse_runtime_state(&text).is_err());
    }

    #[test]
    fn garbage_is_refused_with_a_reason() {
        let detail = parse_runtime_state("not json at all").unwrap_err();

        assert!(detail.contains("not valid JSON"), "{detail}");
    }

    #[test]
    fn an_empty_file_is_refused_rather_than_read_as_an_empty_daemon() {
        assert!(parse_runtime_state("").is_err());
        assert!(parse_runtime_state("[]").is_err());
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    fn health(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    /// The shape `/health` answers with. Both objects together are what
    /// distinguishes our daemon from whatever else might hold the port.
    #[test]
    fn our_own_health_body_identifies_us() {
        let body = health(
            r#"{"status":"ok","lifecycle":{"state":"idle","model":null},"prompt_cache":{"entries":0}}"#,
        );

        assert!(body.get("lifecycle").is_some_and(Value::is_object));
        assert!(body.get("prompt_cache").is_some_and(Value::is_object));
        assert_eq!(model_of(&body), None);
    }

    #[test]
    fn a_loaded_model_is_read_from_the_lifecycle_not_the_runtime_file() {
        let body =
            health(r#"{"lifecycle":{"state":"ready","model":"gpt-oss-20b"},"prompt_cache":{}}"#);

        assert_eq!(model_of(&body).as_deref(), Some("gpt-oss-20b"));
    }

    #[test]
    fn another_service_on_the_port_is_not_mistaken_for_ours() {
        // The discriminating case: adopting whatever holds the port would point
        // the interface at a stranger.
        for foreign in [
            r#"{"status":"ok"}"#,
            r#"{"lifecycle":"ready"}"#,
            r#"{"prompt_cache":{}}"#,
            r#"{"message":"welcome to some other api"}"#,
        ] {
            let body = health(foreign);
            let ours = body.get("lifecycle").is_some_and(Value::is_object)
                && body.get("prompt_cache").is_some_and(Value::is_object);
            assert!(!ours, "{foreign} must not identify as our daemon");
        }
    }
}
