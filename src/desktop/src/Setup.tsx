// First run, update and repair of the managed runtime.
//
// Shown alone, without the dashboard behind it: with no runtime, every other
// control would fail, and offering them would only produce errors the user
// cannot act on.
//
// The install is minutes of downloading, so it never happens as a side effect
// of pressing Start. The user is told what will be fetched and where, and
// starts it deliberately. uv's own output is relayed rather than replaced by an
// indeterminate spinner: a progress bar that cannot report progress is a lie
// about how long this takes.

import { useEffect, useRef, useState } from "react";

import * as api from "./api";

interface Props {
  runtime: api.RuntimeStatus;
  onDone: () => void;
}

export function Setup({ runtime, onDone }: Props) {
  const [running, setRunning] = useState(runtime.state === "INITIALIZING");
  const [step, setStep] = useState<string | null>(null);
  const [output, setOutput] = useState<string[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  const log = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const pending = api.onBootstrap((event) => {
      switch (event.kind) {
        case "step":
          setStep(event.message);
          break;
        case "output":
          setOutput((lines) => [...lines.slice(-400), event.line]);
          break;
        case "done":
          setRunning(false);
          setStep(null);
          onDone();
          break;
        case "failed":
          setRunning(false);
          setFailure(event.message);
          break;
      }
    });
    return () => {
      void pending.then((unlisten) => unlisten());
    };
  }, [onDone]);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [output]);

  async function initialize() {
    setRunning(true);
    setFailure(null);
    setOutput([]);
    try {
      await api.runtimeInitialize();
    } catch (cause) {
      setRunning(false);
      setFailure(String(cause));
    }
  }

  const { title, explanation, action } = describe(runtime);

  return (
    <main className="setup">
      <section className="panel">
        <h2>{title}</h2>
        <p className="prose">{explanation}</p>

        <dl className="setup-facts">
          <dt>Runtime</dt>
          <dd className="wrap">{runtime.envPath}</dd>
          <dt>Application</dt>
          <dd>{runtime.appVersion}</dd>
          {runtime.installedVersion && (
            <>
              <dt>Installed by</dt>
              <dd>{runtime.installedVersion}</dd>
            </>
          )}
        </dl>

        <div className="actions">
          <button onClick={initialize} disabled={running || !runtime.installable}>
            {running ? "Working…" : action}
          </button>
          {step && <span className="pill pill-live">{step}</span>}
        </div>

        {!runtime.installable && (
          <p className="empty">
            This build carries no bundled server project. In a development checkout, run{" "}
            <code>make install</code> from the repository root.
          </p>
        )}

        {failure && (
          <div className="notice notice-error" role="alert">
            {failure}
          </div>
        )}

        {output.length > 0 && (
          <pre className="setup-log" ref={log}>
            {output.join("\n")}
          </pre>
        )}
      </section>
    </main>
  );
}

/** What to say for each runtime state, and what the button does. */
function describe(runtime: api.RuntimeStatus): {
  title: string;
  explanation: string;
  action: string;
} {
  switch (runtime.state) {
    case "UPDATE_REQUIRED":
      return {
        title: "Update the runtime",
        explanation:
          "The installed runtime was built from a different version of the server, so it would " +
          "keep answering with the old code. Rebuilding reinstalls it. Most dependencies are " +
          "already downloaded, so this is usually quick — and it touches neither your models nor " +
          "your settings.",
        action: "Update",
      };
    case "BROKEN":
      return {
        title: "Repair the runtime",
        explanation:
          runtime.detail ??
          "The runtime is present but unusable. Rebuilding it is safe: models, profiles and " +
            "settings are stored separately and are left untouched.",
        action: "Repair",
      };
    case "INITIALIZING":
      return {
        title: "Setting up",
        explanation: "Installing Python and the locked dependencies.",
        action: "Working…",
      };
    default:
      return {
        title: "Set up Quantum Codex",
        explanation:
          "The application installs its own Python and dependencies, so it needs nothing from " +
          "your machine — no system Python, no Homebrew, no terminal setup. Expect a few minutes " +
          "of downloading. GPT-OSS model weights are separate and come later, on demand.",
        action: "Install the runtime",
      };
  }
}
