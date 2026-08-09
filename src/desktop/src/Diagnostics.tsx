// How requests actually executed.
//
// Every figure here is computed by the server — medians, ratios, throughput —
// so this file renders and never calculates. A second implementation of the
// arithmetic would eventually disagree with the first, and the disagreement
// would look like a measurement problem rather than a code one.
//
// Nothing shown is conversation content: no prompt, no reasoning text, no tool
// arguments. Tool *names* appear because a call sequence is often the whole
// explanation for a turn's shape.

import { useCallback, useEffect, useState } from "react";

import * as api from "./api";

export function Diagnostics({ serverRunning }: { serverRunning: boolean }) {
  const [data, setData] = useState<unknown>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setData(await api.requestDiagnostics(50));
      setFailure(null);
    } catch (cause) {
      setData(null);
      setFailure(String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 3000);
    return () => clearInterval(timer);
  }, [refresh]);

  if (!serverRunning) {
    return (
      <section className="panel">
        <h2>Diagnostics</h2>
        <p className="empty">
          Request history lives in the running server and is not written to disk, so
          there is nothing to show while it is stopped.
        </p>
      </section>
    );
  }

  if (failure) {
    return (
      <section className="panel">
        <h2>Diagnostics</h2>
        <div className="notice notice-error">{failure}</div>
      </section>
    );
  }

  const lifetime = api.pick(data, "lifetime") as Record<string, number> | undefined;
  const window = api.pick(data, "window") as Record<string, number | null> | undefined;
  const requests = (api.pick(data, "requests") as Record<string, unknown>[] | undefined) ?? [];

  return (
    <>
      <div className="grid">
        <section className="panel">
          <h2>Since start</h2>
          <dl>
            <dt>Requests</dt>
            <dd>{lifetime?.requests ?? "—"}</dd>
            <dt>Completed</dt>
            <dd>{lifetime?.completed ?? "—"}</dd>
            <dt>Incomplete</dt>
            <dd>{lifetime?.incomplete ?? "—"}</dd>
            <dt>Cancelled</dt>
            <dd>{lifetime?.cancelled ?? "—"}</dd>
            <dt>Failed</dt>
            <dd>{lifetime?.failed ?? "—"}</dd>
          </dl>
        </section>

        <section className="panel">
          <h2>Recent throughput</h2>
          {/* Labelled as a window, because it is: the server keeps a bounded
              history and these are medians over it, not over all time. */}
          <dl>
            <dt>Prefill</dt>
            <dd>{api.rate(window?.median_prefill_tokens_per_second ?? undefined, "tok/s")}</dd>
            <dt>Decode</dt>
            <dd>{api.rate(window?.median_decode_tokens_per_second ?? undefined, "tok/s")}</dd>
            <dt>First token</dt>
            <dd>{api.rate(window?.median_time_to_first_token_seconds ?? undefined, "s")}</dd>
            <dt>Queue wait</dt>
            <dd>{api.rate(window?.median_queue_wait_seconds ?? undefined, "s")}</dd>
            <dt>Window</dt>
            <dd>
              {window?.size ?? "—"} of {window?.capacity ?? "—"}
            </dd>
          </dl>
        </section>

        <section className="panel">
          <h2>Prefix reuse</h2>
          <dl>
            <dt>Requests reusing</dt>
            <dd>{api.percent(window?.cache_hit_ratio ?? undefined)}</dd>
            <dt>Tokens reused</dt>
            <dd>{api.tokens(window?.tokens_reused ?? undefined)}</dd>
            <dt>Tokens evaluated</dt>
            <dd>{api.tokens(window?.tokens_evaluated ?? undefined)}</dd>
          </dl>
        </section>
      </div>

      <section className="panel">
        <h2>Recent requests</h2>
        {requests.length === 0 ? (
          <p className="empty">Nothing served yet.</p>
        ) : (
          <div className="request-table">
            <div className="request-row request-head">
              <span>Request</span>
              <span>Outcome</span>
              <span>Input</span>
              <span>Reused</span>
              <span>Output</span>
              <span>Prefill</span>
              <span>Decode</span>
              <span>Total</span>
              <span>Tools</span>
              <span>Ended because</span>
            </div>
            {requests.map((entry) => (
              <RequestRow key={String(entry.request_id)} entry={entry} />
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function RequestRow({ entry }: { entry: Record<string, unknown> }) {
  const outcome = (entry.outcome as string | null) ?? "running";
  const tone =
    outcome === "completed"
      ? "ok"
      : outcome === "running"
        ? "warn"
        : outcome === "incomplete"
          ? "warn"
          : "bad";
  const tools = (entry.tool_calls as { name: string }[] | undefined) ?? [];

  return (
    <div className="request-row">
      <code>{String(entry.request_id)}</code>
      <span className={`pill pill-${tone}`}>{outcome}</span>
      <span>{api.tokens(api.count(entry, "input_tokens"))}</span>
      <span>{api.tokens(api.count(entry, "cached_tokens"))}</span>
      <span>{api.tokens(api.count(entry, "output_tokens"))}</span>
      <span>{api.rate(api.count(entry, "prefill_tokens_per_second"), "tok/s")}</span>
      <span>{api.rate(api.count(entry, "decode_tokens_per_second"), "tok/s")}</span>
      <span>{api.rate(api.count(entry, "duration_seconds"), "s")}</span>
      <span className="request-tools">{tools.map((call) => call.name).join(", ") || "—"}</span>
      {/* Why the turn ended, in one sentence. This is the question a session
          that stopped early leaves behind, and the answer belongs here rather
          than in a log file. Server-computed; nothing is inferred here. */}
      <span className="request-why">{(entry.termination as string | null) ?? "—"}</span>
    </div>
  );
}
