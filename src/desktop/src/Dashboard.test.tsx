/** The dashboard's information architecture.
 *
 * The complaint these answer is visual, but the fix is structural: five cards of
 * equal-weight key/value rows became three sections — what this session is, what
 * it is doing, what the model can do — each leading with one dominant value.
 *
 * So the assertions are about *where a fact lives* and *whether it is claimed at
 * all*, never about pixels. Two properties carry most of the weight:
 *
 * 1. A number the server never measured must not be printed as a measurement.
 *    `hit_ratio` is 0.0 both for a cache nobody consulted and for one that
 *    missed every time, and only `hits + misses` tells them apart.
 * 2. A fact the daemon never reported must not appear as a row of "—". Six
 *    dashes look like readings that came back empty; they are questions that
 *    were never asked.
 */

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const mocked = vi.mocked(invoke);

const CAPABILITIES = {
  reasoning_efforts: ["low", "medium", "high"],
  default_reasoning_effort: "medium",
  context_window: 131072,
  effective_context_window: 131072,
  supports_tools: true,
  supports_parallel_tool_calls: false,
};

/** A prompt cache in whatever condition the test needs, in the server's shape. */
function cache(over: Record<string, unknown> = {}) {
  return {
    entries: 1, bytes: 1_932_735_283, max_entries: 4, max_bytes: 8_589_934_592,
    hits: 41, misses: 4, hit_ratio: 0.9111, cached_tokens_total: 449_000,
    evaluated_tokens_total: 51_000, evictions: 0, by_model: {},
    ...over,
  };
}

/** The whole `/internal/status` envelope, as the daemon answers it. */
function statusPayload(over: Record<string, unknown> = {}) {
  return {
    server: {
      state: "running", lifecycle: "ready",
      uptime_seconds: 2543.1, endpoint: "http://127.0.0.1:8123",
    },
    model: {
      served_name: "gpt-oss-120b",
      path: "/Volumes/Weights/mlx/gpt-oss-120b-mxfp4-bf16",
      quantization: "mxfp4-4bit", context_length: 131072, layers: 36,
    },
    lifecycle: {
      state: "ready", display_name: "gpt-oss-120b", elapsed_seconds: null,
      idle_timeout_seconds: 600, idle_seconds: 42,
      auto_unload_armed: true, unload_reason: null,
    },
    capabilities: CAPABILITIES,
    inference: { active_requests: 0, queued_requests: 0 },
    prompt_cache: cache(),
    ...over,
  };
}

/** A daemon that is answering, or one that is not. */
function daemon({ up = true, status = statusPayload() }: { up?: boolean; status?: unknown } = {}) {
  mocked.mockImplementation((async (command: string) => {
    if (command === "runtime_status") return { state: "READY", envPath: "/env", appVersion: "1.0.1" };
    if (command === "server_environment") return { resolved: { source: "managed environment" } };
    if (command === "model_status") return { profiles: [{ name: "main", port: 8123 }] };
    if (command === "logs_tail") return [];
    if (command === "model_catalog")
      return {
        models: [
          { slug: "gpt-oss-120b", display_name: "GPT-OSS 120B", tier: "stock", installed: true,
            model: { name: "gpt-oss-120b-mxfp4-bf16", state: "READY" } },
        ],
      };
    if (command === "daemon_discover")
      return up
        ? { connected: true, endpoint: "http://127.0.0.1:8123", model: null, detail: null,
            metadataError: null, adopted: false, manageable: true }
        : { connected: false, endpoint: null, model: null, detail: null,
            metadataError: null, adopted: false, manageable: false };
    if (command === "daemon_status") {
      if (!up) throw new Error("the server is not answering");
      return status;
    }
    return {};
  }) as never);
}

const session = () => screen.findByRole("region", { name: /current session/i });
const performance = () => screen.findByRole("region", { name: /performance/i });
const capabilities = () => screen.findByRole("region", { name: /model capabilities/i });

beforeEach(() => mocked.mockReset());
afterEach(cleanup);

describe("three sections, not five tables", () => {
  it("presents exactly the three dashboard cards", async () => {
    daemon();
    render(<App />);
    await session();

    const grid = document.querySelector(".grid.dashboard")!;
    expect(grid.querySelectorAll(":scope > .panel")).toHaveLength(3);
  });

  it("no longer gives Inference and Prompt cache cards of their own", async () => {
    // They were two- and five-row panels driving a five-card grid whose second
    // row never filled. Their data did not disappear — it moved into
    // Performance — so the discriminating half is the assertion below this one.
    daemon();
    render(<App />);
    await session();

    expect(screen.queryByRole("region", { name: /^inference$/i })).toBeNull();
    expect(screen.queryByRole("region", { name: /^prompt cache$/i })).toBeNull();
    expect(screen.queryByRole("region", { name: /^server$/i })).toBeNull();
    expect(screen.queryByRole("region", { name: /^capabilities$/i })).toBeNull();
  });

  it("keeps what those cards reported, inside Performance", async () => {
    daemon();
    render(<App />);
    const panel = within(await performance());

    // Inference activity, and every prompt-cache counter the old card carried.
    expect(panel.getByText(/0 active/)).toBeTruthy();
    expect(panel.getByText(/0 queued/)).toBeTruthy();
    expect(panel.getByText("Sessions")).toBeTruthy();
    expect(panel.getByText("1 / 4")).toBeTruthy();
    expect(panel.getByText("Memory")).toBeTruthy();
    expect(panel.getByText("1.8 GiB")).toBeTruthy();
    expect(panel.getByText("Tokens reused")).toBeTruthy();
    expect(panel.getByText("449,000")).toBeTruthy();
    expect(panel.getByText("Evictions")).toBeTruthy();
  });
});

describe("the running session", () => {
  it("states the server is running, in the card and not only in the header", async () => {
    daemon();
    render(<App />);

    const panel = within(await session());
    expect(panel.getByText("running")).toBeTruthy();
    expect(panel.getByText("Ready for Codex")).toBeTruthy();
  });

  it("puts the status badge on the heading row, not above the lead", async () => {
    // Stacked, the badge was a line only this card had, so its model name
    // started below the other two cards' primary values. On the heading row it
    // costs no vertical space, which is what re-aligns the three leads.
    daemon();
    render(<App />);

    const panel = await session();
    const badge = within(panel).getByText("running");
    expect(badge.className).toContain("pill");
    expect(badge.closest(".dash-head")).toBeTruthy();
    expect(badge.closest(".dash-lead")).toBeNull();
    // The heading is still first on that row; the badge follows it.
    const head = badge.closest(".dash-head")!;
    expect(head.firstElementChild!.tagName).toBe("H2");
    expect(head.lastElementChild).toBe(badge);
  });

  it("gives every card the same lead structure, so their primaries line up", async () => {
    // No card may carry an extra block above its lead. The floor on `.dash-lead`
    // does the rest, and it can only work from a common starting line.
    daemon();
    render(<App />);
    await session();

    for (const card of document.querySelectorAll(".grid.dashboard > .panel")) {
      expect(card.children[0].className).toContain("dash-head");
      expect(card.children[1].className).toMatch(/dash-lead/);
    }
  });

  it("keeps the badge on the heading row in every server state", async () => {
    for (const [label, up] of [["running", true], ["reconnecting…", false]] as const) {
      cleanup();
      daemon({ up });
      render(<App />);
      const panel = await session();
      const badge = within(panel).getByText(label);
      expect(badge.closest(".dash-head")).toBeTruthy();
    }
  });

  it("makes the loaded model the card's dominant value", async () => {
    daemon();
    render(<App />);

    const panel = within(await session());
    const model = panel.getByText("gpt-oss-120b", { selector: "p" });
    // Not merely present: present as the lead, which is the one thing in this
    // card that is not a row.
    expect(model.className).toContain("dash-value");
  });

  it("carries the session's own facts, and the served name apart from the model", async () => {
    daemon();
    render(<App />);

    const panel = within(await session());
    expect(panel.getByText("Endpoint")).toBeTruthy();
    expect(panel.getByText("http://127.0.0.1:8123")).toBeTruthy();
    expect(panel.getByText("Served as")).toBeTruthy();
    expect(panel.getByText("Uptime")).toBeTruthy();
    expect(panel.getByText("42m")).toBeTruthy();
    expect(panel.getByText("Auto-unload")).toBeTruthy();
    expect(panel.getByText(/after 10 min idle/)).toBeTruthy();
  });

  it("never shows the opaque stable library id", async () => {
    daemon({
      status: statusPayload({
        lifecycle: { state: "ready", display_name: "My Local Model", model: "library-7f3a",
                     idle_timeout_seconds: 600, auto_unload_armed: false },
      }),
    });
    render(<App />);

    await session();
    expect(document.body.textContent).not.toContain("library-7f3a");
  });
});

describe("model capabilities", () => {
  it("leads with the context window and keeps every capability the server reports", async () => {
    daemon();
    render(<App />);

    const panel = within(await capabilities());
    expect(panel.getByText("131,072").className).toContain("dash-value");
    expect(panel.getByText("context tokens")).toBeTruthy();
    expect(panel.getByText("low, medium, high")).toBeTruthy();
    expect(panel.getByText("Tools")).toBeTruthy();
    expect(panel.getByText("Parallel calls")).toBeTruthy();
    expect(panel.getByText("Quantization")).toBeTruthy();
    expect(panel.getByText("mxfp4-4bit")).toBeTruthy();
    expect(panel.getByText("Layers")).toBeTruthy();
    expect(panel.getByText("36")).toBeTruthy();
  });

  it("shows the resident model's path with the whole value recoverable", async () => {
    daemon();
    render(<App />);

    const path = within(await capabilities()).getByText(/gpt-oss-120b-mxfp4-bf16/);
    expect(path.className).toContain("dash-path");
    // Truncation is visual; the value must not be lost with it.
    expect(path.getAttribute("title")).toBe("/Volumes/Weights/mlx/gpt-oss-120b-mxfp4-bf16");
  });

  it("says the resident-only facts are absent rather than printing dashes for them", async () => {
    // Capabilities describe an installed model and are reported with none
    // resident; quantization, layers and path describe the weights in memory.
    daemon({ status: statusPayload({ model: null, lifecycle: { state: "idle" } }) });
    render(<App />);

    const panel = within(await capabilities());
    expect(panel.getByText("131,072")).toBeTruthy();
    expect(panel.queryByText("Quantization")).toBeNull();
    expect(panel.queryByText("Layers")).toBeNull();
    expect(panel.getByText(/appear while a model is resident/i)).toBeTruthy();
  });
});

describe("the prompt cache lead tells a cold cache from a measured zero", () => {
  it("leads with the hit ratio once lookups have happened", async () => {
    daemon();
    render(<App />);

    const panel = within(await performance());
    expect(panel.getByText("91%").className).toContain("dash-value");
    expect(panel.getByText("Prompt cache hit ratio")).toBeTruthy();
  });

  it("says the cache is cold when nothing has been looked up", async () => {
    // The server computes 0.0 for an empty denominator, so this state and the
    // one below arrive with an identical `hit_ratio`.
    daemon({
      status: statusPayload({
        prompt_cache: cache({ hits: 0, misses: 0, hit_ratio: 0, entries: 0, bytes: 0,
                              cached_tokens_total: 0 }),
      }),
    });
    render(<App />);

    const panel = within(await performance());
    expect(panel.getByText("Cache cold")).toBeTruthy();
    expect(panel.getByText(/no prompt reuse measured yet/i)).toBeTruthy();
    expect(panel.queryByText("0%")).toBeNull();
  });

  it("still reports a genuine 0%, which is a measurement and not an absence", async () => {
    // The counterexample. Without it, a card that said "Cache cold" whenever
    // the ratio was zero would pass the test above.
    daemon({
      status: statusPayload({
        prompt_cache: cache({ hits: 0, misses: 7, hit_ratio: 0 }),
      }),
    });
    render(<App />);

    const panel = within(await performance());
    expect(panel.getByText("0%")).toBeTruthy();
    expect(panel.queryByText("Cache cold")).toBeNull();
  });

  it("makes work in flight louder than an idle server", async () => {
    daemon({ status: statusPayload({ inference: { active_requests: 1, queued_requests: 2 } }) });
    render(<App />);

    const activity = within(await performance()).getByText(/1 active/);
    expect(activity.className).toContain("is-busy");
    expect(activity.textContent).toContain("2 queued");
  });

  it("leaves an idle server quiet", async () => {
    daemon();
    render(<App />);

    const activity = within(await performance()).getByText(/0 active/);
    expect(activity.className).not.toContain("is-busy");
  });
});

describe("the stopped dashboard", () => {
  /** Three cards, and no operational reading among them. */
  it("keeps all three sections rather than collapsing the layout", async () => {
    daemon({ up: false });
    render(<App />);
    await session();

    expect(document.querySelectorAll(".grid.dashboard > .panel")).toHaveLength(3);
    expect(await performance()).toBeTruthy();
    expect(await capabilities()).toBeTruthy();
  });

  it("prints no wall of dashes where the daemon reported nothing", async () => {
    daemon({ up: false });
    render(<App />);

    const panel = await session();
    // Every one of State / Served as / Endpoint / Uptime / Auto-unload would
    // have been "—". They are said once, as a sentence, instead.
    expect(within(panel).queryAllByText("—")).toHaveLength(0);
    expect(within(panel).getByText(/appear once it answers/i)).toBeTruthy();
    expect(within(panel).getByText("No model loaded")).toBeTruthy();
  });

  it("keeps the fact that is genuinely known while the daemon is down", async () => {
    // The executable is read from this machine, not from the server.
    daemon({ up: false });
    render(<App />);

    const panel = within(await session());
    expect(panel.getByText("Executable")).toBeTruthy();
    expect(panel.getByText("managed environment")).toBeTruthy();
  });

  it("claims no cache or capability readings at all", async () => {
    daemon({ up: false });
    render(<App />);

    const cachePanel = within(await performance());
    expect(cachePanel.getByText(/no active session/i)).toBeTruthy();
    for (const invented of ["0%", "Sessions", "Memory", "Tokens reused", "Evictions"]) {
      expect(cachePanel.queryByText(invented)).toBeNull();
    }
    const capPanel = within(await capabilities());
    expect(capPanel.getByText("No model loaded")).toBeTruthy();
    for (const invented of ["Reasoning", "Tools", "Parallel calls", "131,072"]) {
      expect(capPanel.queryByText(invented)).toBeNull();
    }
  });

  it("does not call itself connected while the daemon is not answering", async () => {
    // The regression this pins: the caption was chosen from the connection
    // enum, and `reconnecting` counted as connected, so a dashboard with no
    // status at all said "The server is answering."
    daemon({ up: false });
    render(<App />);

    const panel = within(await session());
    expect(panel.queryByText("Ready for Codex")).toBeNull();
    expect(panel.queryByText("The server is answering.")).toBeNull();
    expect(
      panel.getByText(/(stopped answering|start the server to accept|waiting for the server)/i),
    ).toBeTruthy();
  });
});

describe("the control bar keeps its actions", () => {
  it("groups them without changing what they do", async () => {
    daemon();
    render(<App />);
    await session();

    const bar = screen.getByRole("region", { name: /server controls/i });
    for (const name of ["Start", "Stop", "Restart", "Unload model", "Clear cache", "Launch Codex"]) {
      expect(within(bar).getByRole("button", { name })).toBeTruthy();
    }
    // Three groups, told apart by spacing rather than by extra boxes.
    expect(bar.querySelectorAll(".control-group")).toHaveLength(3);
  });

  it("still reaches the backend for each lifecycle action", async () => {
    daemon();
    render(<App />);
    await session();

    await userEvent.click(screen.getByRole("button", { name: "Restart" }));
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("daemon_restart", { profile: undefined }));

    await userEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("daemon_stop"));
  });

  it("still reaches the backend for each maintenance action", async () => {
    daemon();
    render(<App />);
    await session();

    await userEvent.click(screen.getByRole("button", { name: "Clear cache" }));
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("cache_clear"));

    await userEvent.click(screen.getByRole("button", { name: "Unload model" }));
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("model_unload"));
  });

  it("keeps every disabled rule the grouping was supposed to leave alone", async () => {
    daemon();
    render(<App />);
    await session();

    // Start is refused while the daemon answers; Stop, Restart, Clear cache and
    // Unload are offered. Regrouping buttons is exactly the change that can
    // silently drop a `disabled` expression.
    expect(screen.getByRole("button", { name: "Start" })).toHaveProperty("disabled", true);
    for (const name of ["Stop", "Restart", "Clear cache", "Unload model"]) {
      expect(screen.getByRole("button", { name })).toHaveProperty("disabled", false);
    }
  });

  it("offers Start once the daemon has been declared offline", async () => {
    // Not on the first missed poll: three consecutive failures is what turns
    // reconnecting into offline, and that hysteresis is deliberate. Timers are
    // driven rather than waited on, so this stays a fast test of the real rule.
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      daemon({ up: false });
      render(<App />);
      await vi.advanceTimersByTimeAsync(8000);
      await waitFor(() =>
        expect(screen.getByRole("button", { name: "Start" })).toHaveProperty("disabled", false),
      );
      expect(screen.getByRole("button", { name: "Stop" })).toHaveProperty("disabled", true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("offers Launch Codex under the same rule as before — always, and reaching the server", async () => {
    // It has never been gated on the daemon: the generated configuration is
    // produced by the CLI, and a user may want it before starting anything.
    daemon({ up: false });
    render(<App />);

    const launch = await screen.findByRole("button", { name: "Launch Codex" });
    expect(launch).toHaveProperty("disabled", false);
    // Prominent, but by a border the app already uses rather than a new colour.
    expect(launch.className).toContain("primary");

    await userEvent.click(launch);
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("codex_launch_models"));
  });
});

describe("values long enough to break a layout", () => {
  const LONG_PATH =
    "/Volumes/Assets/AI Models/mlx/mlx-community/gpt-oss-120b-MXFP4-Q8-mxfp4-bf16-converted-2026-08/weights";
  const LONG_NAME = "GPT-OSS 120B MXFP4 Q8 Experimental Long Display Name";

  function withLongValues() {
    daemon({
      status: statusPayload({
        model: {
          served_name: "gpt-oss-120b-high-reasoning-experimental-build",
          path: LONG_PATH, quantization: "mxfp4-4bit", context_length: 131072, layers: 36,
        },
        lifecycle: {
          state: "ready", display_name: LONG_NAME,
          idle_timeout_seconds: 600, idle_seconds: 42, auto_unload_armed: true,
        },
      }),
    });
  }

  it("keeps the three-card structure whatever the values measure", async () => {
    withLongValues();
    render(<App />);
    await session();

    const grid = document.querySelector(".grid.dashboard")!;
    expect(grid.querySelectorAll(":scope > .panel")).toHaveLength(3);
    // Each card still has exactly one lead, so no value has been promoted or
    // demoted by its own length.
    for (const card of grid.querySelectorAll(":scope > .panel")) {
      expect(card.querySelectorAll(".dash-lead")).toHaveLength(1);
    }
  });

  it("keeps the path a single truncating element rather than a growing block", async () => {
    withLongValues();
    render(<App />);

    const path = within(await capabilities()).getByText(new RegExp("converted-2026-08"));
    expect(path.className).toContain("dash-path");
    expect(path.getAttribute("title")).toBe(LONG_PATH);
    // One element, not a wrapper that would grow the card with the string.
    expect(path.children).toHaveLength(0);
  });

  it("leaves a long model name in the lead rather than in a row", async () => {
    withLongValues();
    render(<App />);

    const panel = within(await session());
    expect(panel.getByText(LONG_NAME).className).toContain("dash-value");
    expect(panel.getAllByText(LONG_NAME)).toHaveLength(1);
  });
});

/** What is applied, not what was asked for.
 *
 * The Models tab shows the adapter a model is *configured* with. This row shows
 * what reached the weights that are answering right now, which is the only
 * place the two can be told apart — an adapter trained against another model
 * loads without error and applies to nothing.
 */
describe("a resident LoRA adapter", () => {
  const withAdapter = (over: Record<string, unknown>) =>
    statusPayload({
      model: {
        served_name: "gpt-oss-120b",
        path: "/Volumes/Weights/mlx/gpt-oss-120b-mxfp4-bf16",
        quantization: "mxfp4-4bit",
        context_length: 131072,
        layers: 36,
        adapter: over,
      },
    });

  it("reports how much of the adapter reached the weights", async () => {
    daemon({
      status: withAdapter({
        path: "/Volumes/Weights/adapters/style-fr",
        fine_tune_type: "lora",
        applied_tensors: 256,
        adapter_tensors: 256,
      }),
    });
    render(<App />);

    const card = await session();
    expect(within(card).getByText(/lora adapter/i)).toBeTruthy();
    expect(within(card).getByText(/256\/256 tensors applied/)).toBeTruthy();
  });

  it("shows a partial application as the partial thing it is", async () => {
    // Legitimate — an adapter trained over fewer blocks than the model has —
    // and the figure is the only thing that distinguishes it from a full one.
    daemon({
      status: withAdapter({
        path: "/Volumes/Weights/adapters/style-fr",
        fine_tune_type: "lora",
        applied_tensors: 64,
        adapter_tensors: 256,
      }),
    });
    render(<App />);

    const card = await session();
    expect(within(card).getByText(/64\/256 tensors applied/)).toBeTruthy();
  });

  it("says nothing at all when the base weights are serving", async () => {
    daemon();
    render(<App />);

    const card = await session();
    await within(card).findByText(/served as/i);
    expect(within(card).queryByText(/lora adapter/i)).toBeNull();
  });
});
