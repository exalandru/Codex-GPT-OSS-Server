/** Dashboard empty states.
 *
 * The smoke found "No GPT-OSS model is installed yet" beside two READY models.
 * The cause was that availability was read from the *profile* list: no profile
 * meant no model. They are independent facts — models are disk state — and
 * these tests hold them apart.
 */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const mocked = vi.mocked(invoke);

function scenario({
  models = [] as unknown[],
  profiles = [] as unknown[],
}) {
  mocked.mockImplementation(async (command: string) => {
    if (command === "runtime_status") return { state: "READY", envPath: "/env", appVersion: "0" };
    if (command === "server_environment") return { resolved: { source: "managed environment" } };
    if (command === "model_catalog") return { models };
    if (command === "model_status") return { profiles };
    if (command === "logs_tail") return [];
    if (command === "daemon_discover")
      return { connected: false, endpoint: null, model: null, detail: null, metadataError: null, adopted: false, manageable: false };
    return {};
  });
}

function entry(slug: string, state: string | null) {
  return {
    slug,
    display_name: slug,
    supported: true,
    installed: state !== null,
    model: state === null ? null : { name: `${slug}-mxfp4-bf16`, state },
  };
}

beforeEach(() => mocked.mockReset());
afterEach(cleanup);

describe("model availability is independent of profiles", () => {
  it("does not claim nothing is installed when models are READY", async () => {
    // The exact false state from the smoke: models present, no profile.
    scenario({ models: [entry("gpt-oss-20b", "READY"), entry("gpt-oss-120b", "READY")] });
    render(<App />);

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("model_catalog"));
    expect(screen.queryByText(/No GPT-OSS model is installed yet/)).toBeNull();
  });

  it("says a profile is missing instead, which is the real gap", async () => {
    scenario({ models: [entry("gpt-oss-20b", "READY")] });
    render(<App />);

    expect(await screen.findByText(/No profile configured yet/)).toBeTruthy();
  });

  it("does say nothing is installed when nothing is", async () => {
    scenario({ models: [entry("gpt-oss-20b", null), entry("gpt-oss-120b", null)] });
    render(<App />);

    expect(await screen.findByText(/No GPT-OSS model is installed yet/)).toBeTruthy();
  });

  it("distinguishes an unplugged volume from never installed", async () => {
    scenario({ models: [entry("gpt-oss-120b", "MISSING_VOLUME")] });
    render(<App />);

    expect(await screen.findByText(/volume is not mounted/)).toBeTruthy();
    expect(screen.queryByText(/No GPT-OSS model is installed yet/)).toBeNull();
  });

  it("warns about nothing when models and a profile both exist", async () => {
    scenario({
      models: [entry("gpt-oss-20b", "READY")],
      profiles: [{ name: "main", port: 8123 }],
    });
    render(<App />);

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("model_catalog"));
    expect(screen.queryByText(/No GPT-OSS model is installed yet/)).toBeNull();
    expect(screen.queryByText(/No profile configured yet/)).toBeNull();
  });
});

describe("navigation", () => {
  it("offers Logs between Diagnostics and Configuration", async () => {
    scenario({ models: [entry("gpt-oss-20b", "READY")] });
    render(<App />);

    const tabs = (await screen.findAllByRole("tab")).map((tab) => tab.textContent);

    expect(tabs).toEqual(["Dashboard", "Models", "Diagnostics", "Logs", "Configuration"]);
  });

  it("keeps the long server log off the dashboard", async () => {
    scenario({ models: [entry("gpt-oss-20b", "READY")] });
    render(<App />);

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("model_catalog"));
    expect(screen.queryByText("Server log")).toBeNull();
  });
});
