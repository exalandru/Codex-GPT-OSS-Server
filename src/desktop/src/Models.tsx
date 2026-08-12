// The model library.
//
// Every judgement shown here — whether a directory is a usable GPT-OSS model,
// what state it is in, how much space its volume has left — is the server's.
// This file renders those answers and forms no opinion of its own, so adding a
// state on the server needs no change here beyond a label.
//
// The state that matters most is MISSING_VOLUME. It looks like a broken entry
// and is not one: the weights are intact on a drive that happens to be
// unplugged, and the interface has to say so rather than invite a re-download
// of sixty gigabytes.

import { useCallback, useEffect, useState } from "react";

import * as api from "./api";
import { ModelConfiguration } from "./ModelConfiguration";

/** How each state reads, and whether it deserves alarm. */
const STATES: Record<string, { label: string; tone: "ok" | "warn" | "bad" }> = {
  READY: { label: "Ready", tone: "ok" },
  PARTIAL_DOWNLOAD: { label: "Incomplete download", tone: "warn" },
  MISSING_VOLUME: { label: "Drive not attached", tone: "warn" },
  MISSING: { label: "Missing", tone: "bad" },
  INVALID: { label: "Invalid", tone: "bad" },
  INCOMPATIBLE: { label: "Not GPT-OSS", tone: "bad" },
};

type Busy = "import" | "scan" | "forget" | "download" | "storage" | null;

export function Models() {
  const [library, setLibrary] = useState<unknown>(null);
  const [catalog, setCatalog] = useState<Record<string, unknown>[]>([]);
  const [configuring, setConfiguring] = useState<{ slug: string; name: string } | null>(null);
  // Set the moment Cancel is pressed, so the button stops offering itself
  // before the backend has answered. Cleared when the state it describes
  // actually arrives.
  const [cancelRequested, setCancelRequested] = useState(false);
  const [storage, setStorage] = useState<Record<string, unknown> | null>(null);
  const [download, setDownload] = useState<unknown>(null);
  const [repo, setRepo] = useState("");
  const [busy, setBusy] = useState<Busy>(null);
  // What the user's last action produced. Kept apart from `downloadNotice`
  // below because the two have different lifetimes: an action result is news,
  // a background failure is a condition.
  const [result, setResult] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [downloadNotice, setDownloadNotice] = useState<string | null>(null);

  /** The library. Never needs a running server. */
  const refresh = useCallback(async () => {
    try {
      // The catalogue already joins the supported models with what is
      // installed. Doing that here would mean this file deciding which
      // directory is "the 20B", which is the server's judgement.
      const [installed, supported] = await Promise.all([
        api.listModels(),
        api.modelCatalog(),
      ]);
      setLibrary(installed);
      setCatalog((api.pick(supported, "models") as Record<string, unknown>[]) ?? []);
      setStorage((await api.modelStorage().catch(() => null)) as Record<string, unknown> | null);
      setFailure(null);
    } catch (cause) {
      setFailure(String(cause));
    }
  }, []);

  /** Download state. Needs the daemon, so its absence is a note, not a failure. */
  // The optimistic "cancelling" flag is released once the backend reports a
  // state that is no longer running: either it stopped, or it says CANCELLING
  // itself and the flag is redundant.
  useEffect(() => {
    // Through `api.downloadState`, never off the envelope's top level: the
    // server answers `{active, last}`, and reading `download.state` gave
    // `undefined` on every poll. That made the condition below always true, so
    // the optimistic flag was cleared by the very next refresh and the button
    // flipped back to "Cancel" — which is also how a second click became
    // possible.
    const state = api.downloadState(download);
    if (state !== "DOWNLOADING" && state !== "PENDING") setCancelRequested(false);
  }, [download]);

  const refreshDownload = useCallback(async () => {
    try {
      setDownload(await api.downloadStatus());
      setDownloadNotice(null);
    } catch {
      setDownload(null);
      // Explains where downloads run, and says what will happen — it does not
      // send the user to another view. `start_download` starts the daemon
      // itself, so "start the server first" was an instruction the architecture
      // exists to make unnecessary, and it was printed beside a Download button
      // this notice had disabled.
      setDownloadNotice(
        "Downloads run through the server so one machine has one downloader. " +
          "It is not running; starting a download will start it.",
      );
    }
  }, []);

  useEffect(() => {
    void refresh();
    void refreshDownload();
    // Only the download poll repeats: the library changes when the user changes
    // it, and a repeating poll here used to overwrite the result of whatever
    // the user had just done.
    const timer = setInterval(() => void refreshDownload(), 2000);
    return () => clearInterval(timer);
  }, [refresh, refreshDownload]);

  // Deliberately *not* the raw library listing. The server has already decided
  // which installed directory is which catalog model; taking its answer is what
  // keeps a matched model from appearing both as a card and as a row. Matching
  // by display name or path here would be a second, disagreeing opinion.
  const presets = catalog.filter((entry) => entry.supported);
  const models = catalog.filter((entry) => !entry.supported && entry.model);
  const roots = (api.pick(library, "roots") as string[] | undefined) ?? [];

  /** Run an action with a visible busy state and a visible outcome. */
  async function act(kind: Exclude<Busy, null>, run: () => Promise<string | null>) {
    setBusy(kind);
    setResult(null);
    setFailure(null);
    try {
      const message = await run();
      await refresh();
      if (message) setResult(message);
    } catch (cause) {
      // The server's own words: it says which directory, and why.
      setFailure(String(cause));
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="panel">
      <h2>Model library</h2>

      <div className="actions">
        <button
          disabled={busy !== null}
          onClick={() =>
            void act("import", async () => {
              const chosen = await api.chooseModelDirectory();
              // Cancelling is a decision, not a failure — but it still gets an
              // acknowledgement, so the click never looks ignored.
              if (!chosen) return "Import cancelled.";
              const imported = await api.importModel(chosen);
              return `Added ${String(api.pick(imported, "name") ?? chosen)}.`;
            })
          }
        >
          {busy === "import" ? "Choosing…" : "Import existing…"}
        </button>
        <button
          disabled={busy !== null}
          onClick={() =>
            void act("scan", async () => {
              const outcome = await api.scanModels();
              const found = api.count(outcome, "found") ?? 0;
              const added = api.count(outcome, "added") ?? 0;
              const known = api.count(outcome, "already_known") ?? 0;
              return `${found} model${found === 1 ? "" : "s"} found, ${added} added, ${known} already registered.`;
            })
          }
        >
          {busy === "scan" ? "Scanning…" : "Scan roots"}
        </button>
      </div>

      {result && <div className="notice notice-ok">{result}</div>}
      {failure && (
        <div className="notice notice-error" role="alert">
          {failure}
        </div>
      )}


      {/* The two models this build is specialised for, shown whether or not
          they are installed: "not installed yet" is a state with an action
          attached, not an absence. */}
      <ul className="catalog" aria-label="Supported models">
        {presets.map((entry) => (
            <li key={String(entry.slug)} className="catalog-card">
              <div className="catalog-head">
                <strong>{String(entry.display_name)}</strong>
                <span className={`pill ${entry.installed ? "pill-live" : "pill-down"}`}>
                  {entry.installed
                    ? String(api.pick(entry, "model", "state") ?? "installed")
                    : "not installed"}
                </span>
              </div>
              {/* What was actually installed, read-only. The title is the model;
                  this is the copy on disk. Deliberately not `Served as`, which
                  is a setting and lives in Configure. */}
              {entry.installed ? (
                <div className="catalog-installed">
                  <code>{String(api.pick(entry, "model", "name") ?? "")}</code>
                  <span className="catalog-facts">
                    {[
                      api.pick(entry, "model", "quantization"),
                      api.count(entry, "model", "context_length")
                        ? `${api.tokens(api.count(entry, "model", "context_length"))} ctx`
                        : null,
                      api.count(entry, "model", "disk_bytes")
                        ? api.bytes(api.count(entry, "model", "disk_bytes"))
                        : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </div>
              ) : (
                <p className="catalog-note">{String(entry.note ?? "")}</p>
              )}
              {entry.served_conflict === true && (
                <p className="notice notice-error" role="alert">
                  Another installed model is also served as{" "}
                  <code>{String(entry.served_name ?? entry.slug)}</code>. Neither is offered
                  to Codex until one of them is configured with a different name.
                </p>
              )}
              <div className="actions">
                {entry.installed ? (
                  <>
                    <button
                      disabled={busy !== null}
                      onClick={() =>
                        setConfiguring({
                          slug: String(entry.id ?? entry.slug),
                          name: String(entry.display_name),
                        })
                      }
                    >
                      Configure…
                    </button>
                    <button
                      disabled={busy !== null}
                      onClick={() =>
                        void api.revealInFinder(
                          String(api.pick(entry, "model", "path") ?? ""),
                        )
                      }
                    >
                      Reveal in Finder
                    </button>
                    <button
                      disabled={busy !== null}
                      onClick={() =>
                        void act("forget", async () => {
                          await api.forgetModel(String(api.pick(entry, "model", "path") ?? ""));
                          return `Removed ${entry.display_name} from the library.`;
                        })
                      }
                    >
                      Remove from library
                    </button>
                  </>
                ) : (
                  <>
                    {/* Disabled while any transfer is in flight. One download at
                        a time is a server rule, and it enforces it — but a card
                        that stays enabled and answers a click with "already
                        downloading" makes the user discover the rule by being
                        refused. `busy` does not cover this: it is cleared as
                        soon as `start_download` returns, which is immediately,
                        while the transfer runs for hours. */}
                    <button
                      className="primary"
                      disabled={busy !== null || api.isDownloading(download)}
                      onClick={() =>
                        void act("download", async () => {
                          // The repository id comes from the server's catalogue,
                          // never from a copy kept here.
                          await api.startDownload(String(entry.repo));
                          await refreshDownload();
                          return `Downloading ${entry.display_name}…`;
                        })
                      }
                    >
                      {/* The card title already says which model this is. */}
                      Download
                    </button>
                    <button
                      disabled={busy !== null}
                      onClick={() =>
                        void act("import", async () => {
                          const chosen = await api.chooseModelDirectory();
                          if (!chosen) return "Locate cancelled.";
                          // Scoped to this card: the server refuses a directory
                          // that is a different model rather than attaching it
                          // to the wrong entry.
                          await api.importModelFor(chosen, String(entry.slug));
                          return `${entry.display_name} located.`;
                        })
                      }
                    >
                      Locate…
                    </button>
                  </>
                )}
              </div>
            </li>
        ))}
      </ul>

      {/* This list is the models that are *not* one of the two presets, so its
          emptiness says nothing about whether anything is installed. It used to
          announce "No models yet" directly beneath two READY catalogue cards,
          which is simply false to anyone reading the screen. The real
          no-model state is already carried by the cards themselves — NOT
          INSTALLED, with Download and Locate… on them — so when a preset is
          installed there is nothing left for this to explain and it says
          nothing at all. */}
      {models.length === 0 ? (
        presets.some((entry) => entry.installed) ? null : (
          <p className="empty">
            No models yet. Import a GPT-OSS directory, or add a root to scan:{" "}
            {roots.map((root) => (
              <code key={root}>{root}</code>
            ))}
          </p>
        )
      ) : (
        <ul className="library" aria-label="Other installed models">
          {models.map((entry) => {
            const model = entry.model as Record<string, unknown>;
            return (
            <ModelRow
              key={String(entry.id ?? model.path)}
              entry={entry}
              model={model}
              busy={busy !== null}
              onConfigure={() =>
                setConfiguring({
                  slug: String(entry.id ?? entry.slug),
                  name: String(entry.display_name ?? model.name),
                })
              }
              onForget={() =>
                void act("forget", async () => {
                  await api.forgetModel(String(model.path));
                  return `Removed ${String(model.name)} from the library. Its files were left in place.`;
                })
              }
              onReveal={() => void api.revealInFinder(String(model.path))}
            />
            );
          })}
        </ul>
      )}

      {/* Advanced, and below the presets on purpose: the two supported models
          are the product, and this is the escape hatch for anything else. */}
      {configuring && (
        <ModelConfiguration
          slug={configuring.slug}
          displayName={configuring.name}
          onClose={() => setConfiguring(null)}
        />
      )}

      <hr className="divider" />

      {/* Where downloads go. One choice for the installation -- not per profile
          and not per model -- so it belongs beside the library rather than on
          any card. */}
      <div className="storage">
        <span className="advanced-heading">Download location</span>
        <code className="library-path">{String(storage?.download_root ?? "—")}</code>
        <button
          disabled={busy !== null}
          onClick={() =>
            void act("storage", async () => {
              const chosen = await api.chooseModelDirectory();
              if (!chosen) return "Unchanged.";
              await api.setModelStorage(chosen);
              return `Downloads will go to ${chosen}. Models already installed stay where they are.`;
            })
          }
        >
          Choose…
        </button>
        {storage && storage.available === false && (
          <span className="pill pill-down">volume not mounted</span>
        )}
      </div>

      {/* A rule between the two: where downloads go is a standing setting,
          fetching one repository is an action. */}
      <hr className="divider" />
      <h3 className="advanced-heading">Download from Hugging Face</h3>
      <Downloader
        status={download}
        notice={downloadNotice}
        repo={repo}
        onRepoChange={setRepo}
        busy={busy !== null}
        onStart={() =>
          void act("download", async () => {
            await api.startDownload(repo);
            await refreshDownload();
            return `Started downloading ${repo}.`;
          })
        }
        cancelling={cancelRequested || api.downloadState(download) === "CANCELLING"}
        onCancel={() => {
          // The optimistic flag first, so the button stops offering itself
          // before the round trip. Then the request that actually stops the
          // transfer — a bare `return` used to sit between these two lines,
          // and automatic semicolon insertion made everything after it dead
          // code: the button said "Cancelling…" forever while the download ran
          // to completion, because the server was never told.
          setCancelRequested(true);
          void act("download", async () => {
            await api.cancelDownload();
            await refreshDownload();
            return "Cancelling; the partial download is kept and can be resumed.";
          });
        }}
      />
    </section>
  );
}

function ModelRow({
  entry,
  model,
  busy,
  onConfigure,
  onForget,
  onReveal,
}: {
  entry: Record<string, unknown>;
  model: Record<string, unknown>;
  busy: boolean;
  onConfigure: () => void;
  onForget: () => void;
  onReveal: () => void;
}) {
  const state = String(model.state);
  const presentation = STATES[state] ?? { label: state, tone: "bad" as const };
  const volume = model.volume as Record<string, unknown> | undefined;
  const reachable = state !== "MISSING_VOLUME" && state !== "MISSING";

  return (
    <li className="library-row">
      <div className="library-head">
        <strong>{String(entry.display_name ?? model.name)}</strong>
        <span className={`pill pill-${presentation.tone}`}>{presentation.label}</span>
        {model.usable === true && (
          <span className="library-spec">
            {String(model.quantization ?? "—")} ·{" "}
            {api.tokens(api.count(model, "context_length"))} ctx ·{" "}
            {api.bytes(api.count(model, "disk_bytes"))}
          </span>
        )}
      </div>

      <p className="library-detail">
        Served as <code>{String(entry.served_name ?? entry.slug)}</code>
      </p>

      {/* Two models answering to one name are served by neither: the server
          cannot know which weights a request meant. Said here because the
          alternative is a model that looks installed and is quietly absent
          from `/v1/models`. */}
      {entry.served_conflict === true && (
        <p className="notice notice-error" role="alert">
          Another installed model is also served as{" "}
          <code>{String(entry.served_name ?? entry.slug)}</code>. Neither is offered to Codex
          until one of them is configured with a different name.
        </p>
      )}

      <code className="library-path">{String(model.path)}</code>

      {/* The server's own words. It knows why, and paraphrasing loses the
          part that tells the user what to do about it. */}
      {model.usable !== true && <p className="library-detail">{String(model.detail ?? "")}</p>}

      {volume?.external === true && (
        <p className="library-detail">
          Volume {String(volume.name)}:{" "}
          {volume.mounted === true
            ? `attached, ${api.bytes(api.count(volume, "free_bytes"))} free`
            : "not attached"}
        </p>
      )}

      <div className="library-actions">
        <button disabled={busy} onClick={onConfigure}>
          Configure…
        </button>
        <button disabled={busy || !reachable} onClick={onReveal}>
          Reveal in Finder
        </button>
        {/* Worded as removal from the list, because that is all it does: the
            files stay, which matters most when the drive is elsewhere. */}
        <button disabled={busy} onClick={onForget}>
          Remove from library
        </button>
      </div>
    </li>
  );
}


/** Fetching weights from Hugging Face. */
function Downloader({
  status,
  notice,
  repo,
  onRepoChange,
  busy,
  onStart,
  onCancel,
  cancelling,
}: {
  status: unknown;
  notice: string | null;
  repo: string;
  onRepoChange: (value: string) => void;
  busy: boolean;
  onStart: () => void;
  onCancel: () => void;
  /** Cancellation has been requested; the transfer has not necessarily
   *  stopped. The button must not offer itself again meanwhile. */
  cancelling: boolean;
}) {
  // The same reader the rest of the view uses, so this panel and the catalogue
  // cards cannot disagree about whether a transfer is running.
  const active = api.activeDownload(status);
  const last = api.pick(status, "last") as Record<string, unknown> | null;
  const current = active ?? last;
  const running = active !== null;

  const fraction = api.count(current, "fraction");

  return (
    <div className="downloader">
      {/* A missing server is a condition to explain, not an error to alarm
          about: everything else on this view works without one. */}
      {notice && <p className="library-detail">{notice}</p>}
      <div className="actions">
        <input
          className="repo-input"
          value={repo}
          spellCheck={false}
          onChange={(event) => onRepoChange(event.target.value)}
          placeholder="Paste the HugginFace ID"
          disabled={running}
        />
        {/* Deliberately not disabled by `notice`. The notice means the daemon
            is unreachable, and starting a download is exactly what starts it
            (`daemon::ensure_running`). Disabling here made that whole path
            unreachable from the interface: the one action that would have
            fixed the condition was the one the condition switched off. */}
        <button disabled={busy || running || !repo.trim()} onClick={onStart}>
          Download
        </button>
        {running && (
          <button disabled={busy || cancelling} onClick={onCancel}>
            {/* Disabled the instant it is pressed. A second click while the
                first was still in flight used to reach a backend that answered
                400, which reads as the user's mistake rather than our latency. */}
            {cancelling ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </div>

      {current && (
        <div className="download-status">
          <div className="download-head">
            <code>{String(current.repo)}</code>
            <span className="pill pill-warn">{String(current.state)}</span>
          </div>

          {/* Only drawn when the server reported a total. A bar with an
              invented denominator is worse than no bar. */}
          {fraction !== undefined ? (
            <div className="bar">
              <div className="bar-fill" style={{ width: `${fraction * 100}%` }} />
            </div>
          ) : (
            <p className="library-detail">
              This repository does not publish file sizes, so there is no total to
              measure against.
            </p>
          )}

          <p className="library-detail">
            {api.bytes(api.count(current, "downloaded_bytes"))}
            {current.total_bytes ? ` of ${api.bytes(api.count(current, "total_bytes"))}` : ""}
            {current.bytes_per_second
              ? ` · ${api.bytes(api.count(current, "bytes_per_second"))}/s`
              : ""}
            {current.eta_seconds ? ` · ${api.duration(api.count(current, "eta_seconds"))} left` : ""}
          </p>

          {current.detail ? <p className="library-detail">{String(current.detail)}</p> : null}
        </div>
      )}
    </div>
  );
}
