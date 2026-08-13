// The dashboard: one dense operational page (cahier 27).
//
// Everything shown comes from `/internal/status`. Nothing is computed here that
// the server already knows, and nothing is displayed that the server did not
// report — an empty field reads as "—" rather than as a plausible default,
// because a fabricated zero is worse than a visible gap.
//
// Three cards, in the order an operator asks the questions: what is this session
// (is it up, what is loaded, where do I point Codex), what is it doing (activity
// and cache reuse), what can the model do. Each card leads with one dominant
// value and puts the rest in secondary rows, because the previous five
// equal-weight key/value tables made every fact cost the same to find.
//
// The one derivation this file performs is telling a *cold* cache from a cache
// that is genuinely missing: the server reports `hit_ratio` as 0.0 when nothing
// has been looked up at all, so 0% and "no lookups yet" arrive identically. The
// discriminator is `hits + misses`, which the server also reports. Everything
// else is passed through.

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

/** How long the daemon has been up, at the coarseness a glance wants.
 *
 * The raw seconds are still the server's; only the unit is chosen here. A row
 * reading "2543.1 s" makes the reader do the arithmetic that this dashboard
 * exists to have already done.
 */
function uptime(seconds: number | undefined): string {
  if (seconds === undefined) return "—";
  const whole = Math.round(seconds);
  if (whole < 60) return `${whole}s`;
  if (whole < 3600) return `${Math.floor(whole / 60)}m`;
  return `${Math.floor(whole / 3600)}h ${Math.floor((whole % 3600) / 60)}m`;
}

/** Which of the connection states deserves which existing status colour.
 *
 * Deliberately the same three the header pill uses. A running server must be
 * obvious without a fourth colour being invented for each new nuance.
 */
const CONNECTION_TONE: Record<Connection, "live" | "warn" | "down"> = {
  starting: "warn",
  online: "live",
  reconnecting: "warn",
  offline: "down",
};

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
  const selectLaunchModel = async (modelId: string) => {
    setLaunchModel(modelId);
    if (!modelId) {
      setLaunchCommand(null);
      setLaunchConfig(null);
      return;
    }
    try {
      setLaunchCommand(await api.codexLaunchCommand(modelId));
      setLaunchConfig(await api.codexLaunchConfig(modelId));
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
        <Models serverRunning={connected} />
      ) : view === "diagnostics" ? (
        <Diagnostics serverRunning={connected} />
      ) : view === "logs" ? (
        <Logs lines={logs} />
      ) : view === "settings" ? (
        <Configuration serverRunning={connected} />
      ) : (
      <>
      {/* Three groups, separated by spacing rather than by boxes: what the
          server's lifecycle is, what maintenance can be done to it, and the one
          thing the user actually came to do. Every handler, every disabled
          condition and every label is unchanged — only the grouping is new. */}
      <section className="controlbar" aria-label="Server controls">
        <div className="control-group">
          <button disabled={connected || busy !== null} onClick={() => act("start", api.start)}>
            {busy === "start" ? "Starting…" : "Start"}
          </button>
          <button disabled={!connected || busy !== null} onClick={() => act("stop", api.stop)}>
            {busy === "stop" ? "Stopping…" : "Stop"}
          </button>
          <button disabled={busy !== null} onClick={() => act("restart", () => api.restart())}>
            {busy === "restart" ? "Restarting…" : "Restart"}
          </button>
        </div>

        <div className="control-group">
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
        </div>

        <div className="control-group control-cta">
        {/* The workflow this application exists for, so it carries the one
            accent border on the bar. Not a filled button: it is reachable at
            every moment, including while the server is down, and a permanent
            call to action shouting from a stopped dashboard is noise. */}
        <button
          className="primary"
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
        </div>
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
                <option key={model.id ?? model.slug} value={model.id ?? model.slug}>
                  {model.reasoning_effort
                    ? `${model.display_name ?? model.slug} — served as ${model.slug} — reasoning ${model.reasoning_effort}`
                    : `${model.display_name ?? model.slug} — served as ${model.slug}`}
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

      <div className="grid dashboard">
        <CurrentSession
          connection={connection}
          status={status}
          lifecycle={lifecycle}
          lifecycleState={lifecycleState}
          loadingModel={loadingModel}
          executable={environment?.resolved?.source ?? null}
        />
        <Performance status={status} />
        <ModelCapabilities status={status} />
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

/** One dashboard card: a heading row, a lead, and whatever follows.
 *
 * The status badge lives on the heading row rather than above the lead. Stacked,
 * it was an extra line that only the first card had, so that card's model name
 * started 22px below the other two cards' primary values and the row of three
 * read as misaligned. On the heading row it costs no vertical space at all.
 */
function Card({
  title,
  flag,
  children,
}: {
  title: string;
  /** An operational state, in the colours the header pill already uses. */
  flag?: { label: string; tone: "live" | "warn" | "down" };
  children: React.ReactNode;
}) {
  return (
    <section className="panel dash" aria-label={title}>
      <div className="dash-head">
        <h2>{title}</h2>
        {flag && <span className={`pill pill-${flag.tone}`}>{flag.label}</span>}
      </div>
      {children}
    </section>
  );
}

/** The dominant value of a card, with the phrase that says what it is.
 *
 * One per card, deliberately. A dashboard where six numbers are all 22px is the
 * flat table this replaces, drawn larger.
 */
function Lead({
  value,
  caption,
  children,
}: {
  value: string;
  caption: string;
  /** Anything else belonging above the rows. Inside the lead rather than beside
   *  it so one floor normalises the height of all three cards' lead blocks. */
  children?: React.ReactNode;
}) {
  return (
    <div className="dash-lead">
      <p className="dash-value">{value}</p>
      <p className="dash-caption">{caption}</p>
      {children}
    </div>
  );
}

/** The secondary rows. Labels recede, values carry the scan. */
function Rows({ children }: { children: React.ReactNode }) {
  return <dl className="metrics">{children}</dl>;
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className={mono ? "mono" : undefined}>{value}</dd>
    </>
  );
}

/** A card whose subject is genuinely absent, said once instead of six times.
 *
 * Not the same as a value the server did not report — that is still "—" in a
 * row. This is for a whole section having nothing behind it, where a column of
 * dashes claims the shape of data that does not exist.
 */
function Absent({ headline, detail }: { headline: string; detail: string }) {
  return (
    <div className="dash-lead">
      <p className="dash-value dash-value-quiet">{headline}</p>
      <p className="dash-caption">{detail}</p>
    </div>
  );
}

/** Is it up, what is loaded, and where does Codex point.
 *
 * The daemon's state and the model's residency stay separate facts: RUNNING with
 * nothing loaded is normal, and so is RUNNING while weights are still coming in.
 */
function CurrentSession({
  connection,
  status,
  lifecycle,
  lifecycleState,
  loadingModel,
  executable,
}: {
  connection: Connection;
  status: unknown;
  lifecycle: Record<string, unknown> | undefined;
  lifecycleState: string | null;
  loadingModel: boolean;
  executable: string | null;
}) {
  const loaded = typeof lifecycle?.display_name === "string" ? lifecycle.display_name : null;
  // The daemon answered at some point and the answer is still on screen. This,
  // not the connection enum, is what decides whether there is anything to put
  // in the rows: `reconnecting` keeps the last good status, `offline` discards
  // it, and both are "not online".
  const reported = status !== null && status !== undefined;

  // Only claims the server's own state establishes. "Ready for Codex" is said
  // for `ready` and nothing else: that is the one lifecycle value meaning
  // weights are resident and requests will be served without a load first.
  // `idle` gets the on-demand sentence instead, which is what the server
  // actually does, and a daemon that is not answering gets an instruction
  // rather than a diagnosis — it used to read "Connected." while reconnecting
  // with no data at all, which is the one moment the word is unfounded.
  const caption = !reported
    ? connection === "starting"
      ? "Waiting for the server to answer."
      : connection === "reconnecting"
        ? "The server stopped answering. Retrying."
        : "Start the server to accept Codex requests."
    : lifecycleState === "ready"
      ? "Ready for Codex"
      : loadingModel
        ? `${LIFECYCLE_LABEL[lifecycleState ?? ""] ?? lifecycleState}${
            typeof lifecycle?.elapsed_seconds === "number"
              ? ` — ${Math.round(lifecycle.elapsed_seconds)} s`
              : ""
          }`
        : lifecycleState === "model_unloading"
          ? "Releasing the weights; the server keeps running."
          : lifecycleState === "error"
            ? "The server reported an error."
            : lifecycleState === "idle"
              ? "A model is loaded on demand when Codex connects."
              : "The server is answering.";

  return (
    <Card
      title="Current session"
      flag={{ label: CONNECTION_LABEL[connection], tone: CONNECTION_TONE[connection] }}
    >
      <Lead value={loaded ?? "No model loaded"} caption={caption} />
      {/* With nothing reported, every one of these rows would be a dash. Six
          dashes in a column look like readings that came back empty; they are
          in fact questions that were never answered, and saying so once is both
          shorter and truer. The executable stays: it is read from this machine,
          not from the daemon, so it is known either way. */}
      {reported ? (
        <Rows>
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
          <Row label="Served as" value={api.text(status, "model", "served_name")} mono />
          {/* Present only when an adapter is resident, and reporting what was
              *applied* rather than what was configured: the count is measured
              against the loaded weights, so it is the one place that can tell
              an adapter that took effect from one that did not. The Models tab
              shows the configured path. */}
          {api.pick(status, "model", "adapter") != null && (
            <Row
              label="LoRA adapter"
              value={`${api.text(status, "model", "adapter", "fine_tune_type")} · ${api.count(
                status,
                "model",
                "adapter",
                "applied_tensors",
              )}/${api.count(status, "model", "adapter", "adapter_tensors")} tensors applied`}
            />
          )}
          <Row label="Endpoint" value={api.text(status, "server", "endpoint")} mono />
          <Row label="Uptime" value={uptime(api.count(status, "server", "uptime_seconds"))} />
          {/* Enough to understand the feature without a live countdown: what was
              configured, and whether a release is actually scheduled. The
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
          <Row label="Executable" value={executable ?? "not found"} />
        </Rows>
      ) : (
        <>
          <p className="dash-note">
            State, endpoint, uptime and auto-unload are reported by the server, and appear
            once it answers.
          </p>
          <Rows>
            <Row label="Executable" value={executable ?? "not found"} />
          </Rows>
        </>
      )}
    </Card>
  );
}

/** What the server is doing, and whether prefix reuse is paying.
 *
 * Inference activity lives here rather than in a card of its own: two integers
 * did not justify a fifth panel, and "busy" and "reusing prompts" are the same
 * question asked twice.
 */
function Performance({ status }: { status: unknown }) {
  const cache = api.pick(status, "prompt_cache") as Record<string, unknown> | undefined;
  const active = api.count(status, "inference", "active_requests");
  const queued = api.count(status, "inference", "queued_requests");

  // No cache object at all: the daemon is not answering. Nothing measured means
  // nothing to show, and a column of dashes would imply otherwise.
  if (cache === undefined || cache === null) {
    return (
      <Card title="Performance">
        <Absent
          headline="No active session"
          detail="Activity and prompt-cache statistics appear once the server is running."
        />
      </Card>
    );
  }

  // The server computes `hit_ratio` as hits / (hits + misses), and returns 0.0
  // when that denominator is zero. So "0%" and "nothing has been looked up yet"
  // arrive as the same number, and only the counters tell them apart. Printing
  // 0% for a cache nobody has consulted states a measurement that was never
  // taken.
  const hits = api.count(cache, "hits");
  const misses = api.count(cache, "misses");
  const lookups = hits !== undefined && misses !== undefined ? hits + misses : undefined;
  const measured = lookups !== undefined && lookups > 0;
  const busy = (active ?? 0) > 0 || (queued ?? 0) > 0;

  return (
    <Card title="Performance">
      <Lead
        value={measured ? ratio(api.count(cache, "hit_ratio")) : "Cache cold"}
        caption={measured ? "Prompt cache hit ratio" : "No prompt reuse measured yet"}
      >
        {/* Louder when something is in flight, quiet when nothing is. The
            difference is one colour and one weight, which is all it takes for a
            zero-activity dashboard to stop competing with the metric above. */}
        <p className={busy ? "dash-activity is-busy" : "dash-activity"}>
          {`${active ?? 0} active`}
          <span className="dash-dot">·</span>
          {`${queued ?? 0} queued`}
        </p>
      </Lead>
      <Rows>
        <Row
          label="Sessions"
          value={`${api.text(cache, "entries")} / ${api.text(cache, "max_entries")}`}
        />
        <Row label="Memory" value={api.bytes(api.count(cache, "bytes"))} />
        <Row label="Tokens reused" value={api.tokens(api.count(cache, "cached_tokens_total"))} />
        <Row label="Evictions" value={api.text(cache, "evictions")} />
      </Rows>
    </Card>
  );
}

/** What the model can do, and the copy of it on disk.
 *
 * Two groups with different owners, kept apart. The capabilities come from the
 * installed model the server describes; quantization, layers and path describe
 * the weights that are resident *now*, and are simply absent while none are —
 * said in one line rather than as three dashes pretending to be readings.
 */
function ModelCapabilities({ status }: { status: unknown }) {
  const capabilities = api.pick(status, "capabilities") as Record<string, unknown> | undefined;
  const model = api.pick(status, "model") as Record<string, unknown> | undefined;

  if (capabilities === undefined || capabilities === null) {
    return (
      <Card title="Model capabilities">
        <Absent
          headline="No model loaded"
          detail="Start the server with a model installed to see what it supports."
        />
      </Card>
    );
  }

  const path = typeof model?.path === "string" ? model.path : null;

  return (
    <Card title="Model capabilities">
      <Lead
        value={api.tokens(api.count(capabilities, "context_window"))}
        caption="context tokens"
      />
      <Rows>
        <Row
          label="Reasoning"
          value={
            (api.pick(capabilities, "reasoning_efforts") as string[] | undefined)?.join(", ") ?? "—"
          }
        />
        <Row label="Tools" value={yesNo(api.pick(capabilities, "supports_tools"))} />
        <Row
          label="Parallel calls"
          value={yesNo(api.pick(capabilities, "supports_parallel_tool_calls"))}
        />
        {model && <Row label="Quantization" value={api.text(model, "quantization")} />}
        {model && <Row label="Layers" value={api.text(model, "layers")} />}
      </Rows>
      {/* Pinned to the bottom of the card and held to one line. A path is the
          least-read thing here and the most able to set the card's height, so
          it truncates and carries the whole value in its tooltip. */}
      {path ? (
        <p className="dash-path" title={path}>
          {path}
        </p>
      ) : (
        <p className="dash-path dash-path-empty">
          Quantization, layers and path appear while a model is resident.
        </p>
      )}
    </Card>
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
