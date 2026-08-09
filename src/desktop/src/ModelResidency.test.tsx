/** Model residency on the dashboard.
 *
 * The server owns when a model is loaded and when it is released; this view
 * reports what it says. Three things must hold for that to be trustworthy:
 *
 * 1. an unloaded model never reads as a dead server;
 * 2. the Unload action is offered only when there is something to release, and
 *    calls the server rather than deciding anything itself;
 * 3. a refusal from the server is shown in the server's own words.
 */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const mocked = vi.mocked(invoke);

/** A daemon that is answering, with whatever lifecycle the test needs. */
function running(lifecycle: Record<string, unknown>, overrides: Record<string, unknown> = {}) {
  mocked.mockImplementation(async (command: string) => {
    if (command === "runtime_status") return { state: "READY", envPath: "/env", appVersion: "0" };
    if (command === "server_environment") return { resolved: { source: "managed environment" } };
    if (command === "model_catalog")
      return {
        models: [
          {
            slug: "gpt-oss-20b",
            display_name: "gpt-oss-20b",
            supported: true,
            installed: true,
            model: { name: "gpt-oss-20b-mxfp4-bf16", state: "READY" },
          },
        ],
      };
    if (command === "model_status") return { profiles: [{ name: "main", port: 8123 }] };
    if (command === "logs_tail") return [];
    if (command === "daemon_discover")
      return {
        connected: true,
        endpoint: "http://127.0.0.1:8123",
        model: null,
        detail: null,
        metadataError: null,
        adopted: false,
        manageable: true,
      };
    if (command === "daemon_status") return { lifecycle, server: {}, prompt_cache: {} };
    if (command in overrides) return overrides[command];
    return {};
  });
}

const READY = {
  state: "ready",
  display_name: "gpt-oss-20b",
  idle_timeout_seconds: 600,
  auto_unload_armed: true,
  idle_seconds: 42,
  unload_reason: null,
};

beforeEach(() => mocked.mockReset());
afterEach(cleanup);

describe("an unloaded model is not a failed server", () => {
  it("keeps reporting the daemon as running", async () => {
    running({ state: "idle", unload_reason: "idle_timeout" });
    render(<App />);

    // The connection pill and the Server panel both say so, and both are about
    // the daemon rather than about weights.
    expect(await screen.findAllByText("running")).toHaveLength(2);
    expect(screen.queryByText("stopped")).toBeNull();
  });

  it("says why the model was released rather than leaving it unexplained", async () => {
    running({ state: "idle", unload_reason: "idle_timeout" });
    render(<App />);

    expect(await screen.findByText(/released after being idle/)).toBeTruthy();
  });

  it("distinguishes a release the user asked for", async () => {
    running({ state: "idle", unload_reason: "manual" });
    render(<App />);

    expect(await screen.findByText(/released on request/)).toBeTruthy();
  });

  it("reports unloading as its own state, not as stopping", async () => {
    running({ state: "model_unloading", display_name: "gpt-oss-20b" });
    render(<App />);

    expect(await screen.findByText(/Unloading the model\. The server keeps running\./)).toBeTruthy();
    expect(screen.queryByText("stopped")).toBeNull();
  });
});

describe("the Unload action", () => {
  it("is offered when a model is resident", async () => {
    running(READY);
    render(<App />);

    expect(await screen.findByRole("button", { name: "Unload model" })).toBeTruthy();
  });

  it("is absent when there is nothing to release", async () => {
    running({ state: "idle" });
    render(<App />);

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("daemon_status"));
    expect(screen.queryByRole("button", { name: "Unload model" })).toBeNull();
  });

  it("is absent while the model is still loading", async () => {
    running({ state: "model_loading", display_name: "gpt-oss-20b", elapsed_seconds: 3 });
    render(<App />);

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("daemon_status"));
    expect(screen.queryByRole("button", { name: "Unload model" })).toBeNull();
  });

  it("invokes the server rather than deciding anything itself", async () => {
    running(READY, { model_unload: { released: true, lifecycle: { state: "idle" } } });
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Unload model" }));

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("model_unload"));
  });

  it("shows the server's refusal in its own words", async () => {
    // What the server answers with 409 while a generation is running. Replacing
    // it with a generic failure would lose the only actionable part.
    running(READY);
    const answering = mocked.getMockImplementation()!;
    mocked.mockImplementation((async (command: string) => {
      if (command === "model_unload") throw new Error("Model is currently in use");
      return answering(command);
    }) as never);
    render(<App />);

    await userEvent.click(await screen.findByRole("button", { name: "Unload model" }));

    expect(await screen.findByText(/Model is currently in use/)).toBeTruthy();
  });
});

describe("idle visibility", () => {
  it("reports the configured timeout and how long the model has been idle", async () => {
    running(READY);
    render(<App />);

    expect(await screen.findByText(/after 10 min idle — idle 42s/)).toBeTruthy();
  });

  it("says disabled rather than inventing a timeout of zero", async () => {
    running({ ...READY, idle_timeout_seconds: 0, auto_unload_armed: false });
    render(<App />);

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("daemon_status"));
    expect(screen.getByText("disabled")).toBeTruthy();
  });

  it("does not claim an unload is scheduled when none is armed", async () => {
    running({ ...READY, auto_unload_armed: false });
    render(<App />);

    expect(await screen.findByText("after 10 min idle")).toBeTruthy();
  });
});
