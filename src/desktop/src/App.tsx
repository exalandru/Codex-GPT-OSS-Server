// The dashboard: one dense operational page (cahier 27).
//
// Everything shown comes from `/internal/status`. Nothing is computed here that
// the server already knows, and nothing is displayed that the server did not
// report — an empty field reads as "—" rather than as a plausible default,
// because a fabricated zero is worse than a visible gap.

import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import { Configuration } from "./Configuration";
import { Diagnostics } from "./Diagnostics";
import { Models } from "./Models";
import { Setup } from "./Setup";

type Busy = "start" | "stop" | "restart" | "cache" | "unload" | null;

const POLL_MS = 2000;

// A single missed poll is not an outage. The daemon is a separate process on
// loopback: a request can lose a race with a restart, or arrive while the
// process is momentarily busy, and flashing STOPPED for one second teaches a
// user to distrust the indicator.
//
// Three consecutive failures is about six seconds — long enough that a real
// death is still reported promptly, short enough that nobody stares at a lie.
// Last-known-good data is retained meanwhile, so the dashboard keeps showing
// what it last saw instead of blanking.
const FAILURES_BEFORE_OFFLINE = 3;

// `starting` exists because pressing Start is an action whose effect takes
// seconds to become observable. Without it the interface reports "stopped"
// while the daemon is in fact coming up, which is indistinguishable from the
// button having done nothing.
type Connection = "starting" | "online" | "reconnecting" | "offline";

const CONNECTION_LABEL: Record<Connection, string> = {
  starting: "starting…",
  online: "running",
  reconnecting: "reconnecting…",
  offline: "stopped",
};

// What the server calls its lifecycle, in the words a user would use. Anything
// unrecognised falls through to the raw value rather than being hidden.
const LIFECYCLE_LABEL: Record<string, string> = {
  idle: "No model loaded",
  model_loading: "Loading",
  model_warming_up: "Warming up MLX",
  ready: "Ready",
  model_unloading: "Unloading",
  stopping: "Stopping",
  error: "Error",
};

// Why the model is no longer resident. Shown beside "No model loaded" because
// that phrase alone reads, to someone who did not press anything, as the server
// having lost its model rather than deliberately released it.
// No `shutdown`: the server has no such reason to report. Releasing at shutdown
// happens outside the supervisor, and a daemon that has stopped answers no
// status anyway, so the label could never be rendered.
const UNLOAD_REASON_LABEL: Record<string, string> = {
  manual: "released on request",
  idle_timeout: "released after being idle",
  model_switch: "replaced by another model",
};

/** The configured idle timeout, in the unit that reads naturally for it.
 *
 * The server reports seconds because that is what it enforces; whole minutes
 * are what a user configured, and a short witness timeout stays legible instead
 * of rounding to "0 minutes".
 */
function idleTimeout(seconds: number | undefined): string | null {
  if (seconds === undefined || seconds <= 0) return null;
  if (seconds < 60) return `${seconds} s`;
  return `${Math.round(seconds / 60)} min`;
}

export default function App() {
  const [discovery, setDiscovery] = useState<api.Discovery | null>(null);
  const [status, setStatus] = useState<unknown>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<string | null>(null);
  const [launchCommand, setLaunchCommand] = useState<string | null>(null);
  const [launchConfig, setLaunchConfig] = useState<string | null>(null);
  const [launchForm, setLaunchForm] = useState<"command" | "config">("command");
  // The Launch Codex panel is open when this is set, whether or not a model has
  // been chosen yet — the selector has to be reachable before there is anything
  // to copy.
  const [launchOpen, setLaunchOpen] = useState(false);
  const [launchModels, setLaunchModels] = useState<api.LaunchModels | null>(null);
  const [launchModel, setLaunchModel] = useState<string>("");
  const [environment, setEnvironment] = useState<api.ServerEnvironment | null>(null);
  const [runtime, setRuntime] = useState<api.RuntimeStatus | null>(null);
  const [models, setModels] = useState<unknown>(null);
  const [catalog, setCatalog] = useState<Record<string, unknown>[]>([]);
  const [connection, setConnection] = useState<Connection>("offline");
  const failures = useRef(0);
  // How long a start is allowed to look like starting rather than stopped.
  // Generous: the daemon binds in under a second, but a cold import is slower.
  const startingUntil = useRef(0);
  const [view, setView] = useState<
    "dashboard" | "models" | "diagnostics" | "logs" | "settings"
  >("dashboard");

  const refresh = useCallback(async () => {
    // Environment and runtime are local facts and are read whatever the daemon
    // is doing; only the daemon-facing calls feed the connection state.
    try {
      setEnvironment(await api.serverEnvironment());
      const state = await api.runtimeStatus();
      setRuntime(state);
      // Two independent facts, asked separately on purpose. Which models are
      // installed is *disk state*: it does not depend on a profile existing, on
      // one being selected, on the profile loader being healthy, or on the
      // daemon running. Deriving it from the profile list is what produced
      // "No GPT-OSS model is installed yet" beside two READY models.
      setModels(state.state === "READY" ? await api.modelStatus().catch(() => null) : null);
      setCatalog(
        state.state === "READY"
          ? ((api.pick(await api.modelCatalog().catch(() => null), "models") as
              | Record<string, unknown>[]
              | undefined) ?? [])
          : [],
      );
      setLogs(await api.tailLogs(300));
    } catch (cause) {
      setError(String(cause));
    }

    try {
      const found = await api.discover();
      if (!found.connected) {
        throw new Error(found.detail || "the server is not answering");
      }
      const next = await api.status();
      setDiscovery(found);
      setStatus(next);
      failures.current = 0;
      setConnection("online");
    } catch {
      failures.current += 1;
      // While a start is in flight, an unanswered poll is expected: the daemon
      // has not bound its socket yet. Reporting "stopped" here is the bug this
      // state exists to prevent.
      if (startingUntil.current > Date.now()) {
        setConnection("starting");
        return;
      }
      if (failures.current >= FAILURES_BEFORE_OFFLINE) {
        setConnection("offline");
        // Only now is the old data misleading rather than merely stale.
        setStatus(null);
        setDiscovery(await api.discover().catch(() => null));
      } else {
        // Keep the last good status on screen: it is what the daemon most
        // recently said, and it is still the best answer available.
        setConnection("reconnecting");
      }
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);


  /** Pick the model the generated configuration is for.
   *
   * Both forms are regenerated from the server for that model, rather than the
   * name being substituted into text already produced: the reasoning effort
   * that has to travel with it is resolved backend-side, and splicing would
   * lose it. An empty choice clears both, so there is never a copyable command
   * with a missing or placeholder model in it.
   */
  const selectLaunchModel = async (slug: string) => {
    setLaunchModel(slug);
    if (!slug) {
      setLaunchCommand(null);
      setLaunchConfig(null);
      return;
    }
    try {
      setLaunchCommand(await api.codexLaunchCommand(slug));
      setLaunchConfig(await api.codexLaunchConfig(slug));
    } catch (cause) {
      setLaunchCommand(null);
      setLaunchConfig(null);
      setError(String(cause));
    }
  };

  const act = async (kind: Exclude<Busy, null>, run: () => Promise<unknown>) => {
    setBusy(kind);
    setError(null);
    if (kind === "start" || kind === "restart") {
      // Immediately, before the request is even sent: the user pressed a
      // button and must see that it registered.
      startingUntil.current = Date.now() + 30_000;
      setConnection("starting");
      failures.current = 0;
    }
    try {
      await run();
    } catch (cause) {
      // Surfaced verbatim: these are the server's own words about why it would
      // not start, and paraphrasing them loses the actionable part.
      startingUntil.current = 0;
      setError(String(cause));
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  // Actions are enabled on the last-known-good state: a transient miss must not
  // grey out Stop on a server that is plainly still running.
  const connected = connection === "online" || connection === "reconnecting";
  const lifecycle = api.pick(status, "lifecycle") as Record<string, unknown> | undefined;
  const lifecycleState = typeof lifecycle?.state === "string" ? lifecycle.state : null;
  const loadingModel =
    lifecycleState === "model_loading" || lifecycleState === "model_warming_up";
  const unloadingModel = lifecycleState === "model_unloading";
  // Offered only when there is actually something to release. A button that is
  // present whenever the server runs would be enabled for the state in which it
  // does nothing, which is how a control teaches a user to distrust it.
  const modelResident = lifecycleState === "ready";

  // With no usable runtime nothing else is actionable, so the installer is
  // shown alone rather than a dashboard whose every control would fail.
  if (runtime && runtime.state !== "READY") {
    return <Setup runtime={runtime} onDone={() => void refresh()} />;
  }

  const profiles = api.pick(models, "profiles");
  const hasProfile = Array.isArray(profiles) && profiles.length > 0;
  // Installed at all, whatever state it is in. A model on an unplugged volume
  // was still installed; saying otherwise would send the user to download
  // something they already have.
  const installed = catalog.filter((entry) => entry.installed);
  const hasModel = installed.length > 0;
  const allUnreachable =
    hasModel &&
    installed.every((entry) => api.pick(entry, "model", "state") === "MISSING_VOLUME");

  return (
    <main>
      <header>
        <h1>Quantum Codex GPT-OSS Server</h1>
        <span
          className={
            connection === "online"
              ? "pill pill-live"
              : connection === "reconnecting"
                ? "pill pill-warn"
                : "pill pill-down"
          }
        >
          {CONNECTION_LABEL[connection]}
        </span>
        <nav className="views" role="tablist">
          {(["dashboard", "models", "diagnostics", "logs", "settings"] as const).map((id) => (
            <button
              key={id}
              role="tab"
              aria-selected={view === id}
              className="view-tab"
              onClick={() => setView(id)}
            >
              {
                {
                  dashboard: "Dashboard",
                  models: "Models",
                  diagnostics: "Diagnostics",
                  logs: "Logs",
                  settings: "Configuration",
                }[id]
              }
            </button>
          ))}
        </nav>
        {discovery?.endpoint && <code className="endpoint">{discovery.endpoint}</code>}
      </header>

      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}
      {/* The prominent, indeterminate progress the load deserves. MLX reports
          no progress for a weight load, so this shows elapsed time and never a
          predicted completion. */}
      {loadingModel && (
        <div className="notice notice-busy" role="status">
          <span className="spinner" aria-hidden="true" />
          {LIFECYCLE_LABEL[lifecycleState ?? ""] ?? lifecycleState}
          {typeof lifecycle?.display_name === "string" ? ` ${lifecycle.display_name}` : ""}…{" "}
          {typeof lifecycle?.elapsed_seconds === "number"
            ? `${Math.round(lifecycle.elapsed_seconds)} s`
            : ""}
        </div>
      )}
      {/* Releasing weights is fast but not instant, and the state is real: the
          server reports it rather than the interface guessing from a button
          press. Elapsed time is deliberately absent — there is nothing honest
          to measure over an operation this short. */}
      {unloadingModel && (
        <div className="notice notice-busy" role="status">
          <span className="spinner" aria-hidden="true" />
          Unloading the model. The server keeps running.
        </div>
      )}
      {connection === "online" && lifecycleState === "idle" && (
        <div className="notice">
          The daemon is running with no model loaded. That is normal: pick a model when you
          launch Codex and it is loaded on demand.
          {typeof lifecycle?.unload_reason === "string" &&
            ` The last model was ${
              UNLOAD_REASON_LABEL[lifecycle.unload_reason] ?? lifecycle.unload_reason
            }.`}
        </div>
      )}
      {/* Distinct situations, distinct advice. Conflating them is what told a
          user with two READY models to go and download one. */}
      {runtime?.state === "READY" && !hasModel && (
        <div className="notice">
          No GPT-OSS model is installed yet. Import or download one from the Models view.
        </div>
      )}
      {runtime?.state === "READY" && allUnreachable && (
        <div className="notice">
          Your models are installed but their volume is not mounted. Reattach the drive; nothing
          needs downloading again.
        </div>
      )}
      {runtime?.state === "READY" && hasModel && !hasProfile && (
        <div className="notice">
          No profile configured yet. Models are ready; create a profile in Configuration to
          choose the port and defaults the server starts with.
        </div>
      )}
      {/* A runtime file we cannot read is a metadata problem, not a verdict on
          the daemon. Shown whether or not the daemon turned out to be alive. */}
      {discovery?.metadataError && (
        <div className="notice">
          The server's runtime file is unusable: {discovery.metadataError}
          {discovery.connected
            ? " The server itself is answering, so this only affects actions that need its token."
            : ""}
        </div>
      )}
      {discovery?.adopted && discovery.detail && (
        <div className="notice">{discovery.detail}</div>
      )}
      {connection === "offline" && discovery?.detail && !discovery.adopted && (
        <div className="notice">{discovery.detail}</div>
      )}


      {view === "models" ? (
        <Models />
      ) : view === "diagnostics" ? (
        <Diagnostics serverRunning={connected} />
      ) : view === "logs" ? (
        <Logs lines={logs} />
      ) : view === "settings" ? (
        <Configuration serverRunning={connected} />
      ) : (
      <>
      <section className="actions">
        <button disabled={connected || busy !== null} onClick={() => act("start", api.start)}>
          {busy === "start" ? "Starting…" : "Start"}
        </button>
        <button disabled={!connected || busy !== null} onClick={() => act("stop", api.stop)}>
          {busy === "stop" ? "Stopping…" : "Stop"}
        </button>
        <button disabled={busy !== null} onClick={() => act("restart", () => api.restart())}>
          {busy === "restart" ? "Restarting…" : "Restart"}
        </button>
        {/* Only when weights are actually resident. The server refuses an
            unload while inference is in flight and says so; that refusal is
            shown verbatim rather than pre-empted here, because the dashboard's
            two-second poll cannot know what arrived a moment ago. */}
        {modelResident && (
          <button
            disabled={!connected || busy !== null}
            onClick={() => act("unload", api.unloadModel)}
          >
            {busy === "unload" ? "Unloading…" : "Unload model"}
          </button>
        )}
        <button
          disabled={!connected || busy !== null}
          onClick={() => act("cache", api.clearCache)}
        >
          {busy === "cache" ? "Clearing…" : "Clear cache"}
        </button>
        <button
          aria-expanded={launchOpen}
          onClick={async () => {
            // A disclosure: pressing it again puts it away. Opening something
            // with no way to close it is the complaint this answers.
            if (launchOpen) {
              setLaunchOpen(false);
              return;
            }
            try {
              const choices = await api.codexLaunchModels();
              setLaunchModels(choices);
              setLaunchOpen(true);
              // The profile's default, or the only installed model. When the
              // server resolves neither, nothing is preselected and there is
              // nothing to copy until the user says which model they mean.
              await selectLaunchModel(choices.default ?? "");
            } catch (cause) {
              setError(String(cause));
            }
          }}
        >
          Launch Codex
        </button>
      </section>

      {launchOpen && (
        <section className="panel launch">
          <div className="catalog-head">
            <h2>Launch Codex</h2>
            <button onClick={() => setLaunchOpen(false)}>Hide</button>
          </div>

          {/* Codex must be told which model to use. Given none, it falls back to
              its own cloud model selection — against a provider that serves
              none of them — so an omitted model is not "load on demand", it is
              a launch that quietly does not use this server. Hence a required
              choice rather than a blank. The list and the preselection come
              from the server; this form decides nothing about either, and
              choosing here does not change the profile's default. */}
          <div className="setting model-setting">
            <label className="setting-label" htmlFor="launch-model">
              Model
            </label>
            <select
              id="launch-model"
              className="repo-input"
              value={launchModel}
              onChange={(event) => void selectLaunchModel(event.target.value)}
            >
              <option value="">Select a model…</option>
              {(launchModels?.models ?? []).map((model) => (
                <option key={model.slug} value={model.slug}>
                  {model.reasoning_effort
                    ? `${model.slug} — reasoning ${model.reasoning_effort}`
                    : model.slug}
                </option>
              ))}
            </select>
            <p className="setting-help">
              Used for this configuration only. The profile&rsquo;s default model is
              unchanged.
            </p>
          </div>

          {launchCommand === null ? (
            <p className="library-detail">
              Choose a model to generate a command you can run.
            </p>
          ) : (
            <>
              {/* Two presentations of one configuration, both generated by the
                  headless CLI. Rebuilding either here would be a second
                  interpretation of the provider wiring. */}
              <div className="actions" role="tablist" aria-label="Configuration form">
                {(["command", "config"] as const).map((form) => (
                  <button
                    key={form}
                    role="tab"
                    aria-selected={launchForm === form}
                    className="view-tab"
                    onClick={() => setLaunchForm(form)}
                  >
                    {form === "command" ? "Temporary command" : "Global / VS Code config"}
                  </button>
                ))}
              </div>

              <pre className="command">
                {launchForm === "command" ? launchCommand : (launchConfig ?? "")}
              </pre>
              <button
                onClick={() =>
                  void navigator.clipboard.writeText(
                    (launchForm === "command" ? launchCommand : launchConfig) ?? "",
                  )
                }
              >
                Copy
              </button>
            </>
          )}
        </section>
      )}

      <div className="grid">
        <Panel title="Server">
          {/* The daemon's own state, which no longer depends on whether any
              weights are resident. */}
          <Row label="State" value={connection === "online" ? "running" : CONNECTION_LABEL[connection]} />
          <Row label="Executable" value={environment?.resolved?.source ?? "not found"} />
          <Row label="Uptime" value={`${api.text(status, "server", "uptime_seconds")} s`} />
          <Row label="Endpoint" value={api.text(status, "server", "endpoint")} />
        </Panel>

        <Panel title="Model">
          {/* Separate from the server's state: RUNNING / NONE is a normal
              pairing, and so is RUNNING / LOADING. */}
          <Row
            label="Loaded"
            value={
              typeof lifecycle?.display_name === "string" ? lifecycle.display_name : "none"
            }
          />
          <Row
            label="State"
            value={
              lifecycleState
                ? `${LIFECYCLE_LABEL[lifecycleState] ?? lifecycleState}${
                    loadingModel && typeof lifecycle?.elapsed_seconds === "number"
                      ? ` — ${Math.round(lifecycle.elapsed_seconds)} s`
                      : ""
                  }`
                : "—"
            }
          />
          {/* Enough to understand the feature without a live countdown: what
              was configured, and whether a release is actually scheduled. The
              remaining time is derivable and would jitter with the poll. */}
          <Row
            label="Auto-unload"
            value={(() => {
              // "not reported" and "reported as 0" are different facts. Folding
              // them together said "disabled" whenever the daemon was
              // unreachable, which is the one moment the claim is unfounded.
              const seconds = api.count(lifecycle, "idle_timeout_seconds");
              if (seconds === undefined) return "—";
              const configured = idleTimeout(seconds);
              if (configured === null) return "disabled";
              if (!lifecycle?.auto_unload_armed) return `after ${configured} idle`;
              const idle = api.count(lifecycle, "idle_seconds");
              return `after ${configured} idle — idle ${api.duration(idle)}`;
            })()}
          />
          <Row label="Served as" value={api.text(status, "model", "served_name")} />
          <Row label="Quantization" value={api.text(status, "model", "quantization")} />
          <Row label="Layers" value={api.text(status, "model", "layers")} />
          <Row label="Path" value={api.text(status, "model", "path")} wrap />
        </Panel>

        <Panel title="Capabilities">
          <Row
            label="Context"
            value={`${api.tokens(api.count(status, "capabilities", "context_window"))} tokens`}
          />
          <Row
            label="Reasoning"
            value={
              (api.pick(status, "capabilities", "reasoning_efforts") as string[] | undefined)?.join(
                ", ",
              ) ?? "—"
            }
          />
          <Row label="Tools" value={yesNo(api.pick(status, "capabilities", "supports_tools"))} />
          <Row
            label="Parallel calls"
            value={yesNo(api.pick(status, "capabilities", "supports_parallel_tool_calls"))}
          />
        </Panel>

        <Panel title="Inference">
          <Row label="Active" value={api.text(status, "inference", "active_requests")} />
          <Row label="Queued" value={api.text(status, "inference", "queued_requests")} />
        </Panel>

        <Panel title="Prompt cache">
          <Row
            label="Sessions"
            value={`${api.text(status, "prompt_cache", "entries")} / ${api.text(
              status,
              "prompt_cache",
              "max_entries",
            )}`}
          />
          <Row
            label="Memory"
            value={api.bytes(api.count(status, "prompt_cache", "bytes"))}
          />
          <Row label="Hit ratio" value={ratio(api.count(status, "prompt_cache", "hit_ratio"))} />
          <Row
            label="Tokens reused"
            value={api.tokens(api.count(status, "prompt_cache", "cached_tokens_total"))}
          />
          <Row label="Evictions" value={api.text(status, "prompt_cache", "evictions")} />
        </Panel>
      </div>

      </>
      )}
    </main>
  );
}

/** The server log, in its own view.
 *
 * Moved off the dashboard: it is long, it pushed the operational summary below
 * the fold, and it is consulted deliberately rather than watched. The lines
 * come from the same poll the dashboard already runs, so it stays live.
 */
function Logs({ lines }: { lines: string[] }) {
  return (
    <section className="panel logs">
      <h2>Server log</h2>
      {lines.length === 0 ? (
        <p className="empty">Nothing logged yet.</p>
      ) : (
        <pre>{lines.join("\n")}</pre>
      )}
    </section>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      <dl>{children}</dl>
    </section>
  );
}

function Row({ label, value, wrap }: { label: string; value: string; wrap?: boolean }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className={wrap ? "wrap" : undefined}>{value}</dd>
    </>
  );
}

function yesNo(value: unknown): string {
  if (typeof value !== "boolean") return "—";
  return value ? "yes" : "no";
}

function ratio(value: number | undefined): string {
  if (value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}
