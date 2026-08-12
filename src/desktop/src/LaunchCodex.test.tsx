/** The Launch Codex panel must always name a model.
 *
 * Measured with the real client: given no `model`, Codex does not fall back to
 * a model this provider serves — it falls back to its own cloud model
 * selection, against a provider that has none of them. So a profile default of
 * None is a valid *server* state ("load on demand") and an invalid *launch*
 * configuration, and the panel must not quietly produce the second from the
 * first.
 *
 * Everything the selector shows — which models exist, which one is
 * preselected, what reasoning effort each carries — comes from the server. The
 * assertions below are therefore about what the form *does with* that answer,
 * never about what the answer should be.
 */

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const mocked = vi.mocked(invoke);

const MODELS = [
  { id: "gpt-oss-20b", slug: "gpt-oss-20b", display_name: "GPT-OSS 20B", reasoning_effort: "medium" },
  { id: "gpt-oss-120b", slug: "gpt-oss-120b", display_name: "GPT-OSS 120B", reasoning_effort: "high" },
  { id: "library-7f3a", slug: "codex-local", display_name: "My Local Model", reasoning_effort: "medium" },
];

/** A running daemon whose generator answers for whichever model it is given. */
function running(launchDefault: string | null) {
  mocked.mockImplementation((async (command: string, args?: Record<string, unknown>) => {
    if (command === "runtime_status") return { state: "READY", envPath: "/env", appVersion: "0" };
    if (command === "server_environment") return { resolved: { source: "managed environment" } };
    if (command === "model_catalog") return { models: [] };
    if (command === "model_status") return { profiles: [{ name: "main", port: 8123 }] };
    if (command === "logs_tail") return [];
    if (command === "daemon_discover")
      return { connected: true, endpoint: "http://127.0.0.1:8123", model: null, detail: null,
               metadataError: null, adopted: false, manageable: true };
    if (command === "daemon_status") return { lifecycle: { state: "idle" }, server: {}, prompt_cache: {} };
    if (command === "codex_launch_models") return { default: launchDefault, models: MODELS };
    if (command === "codex_launch_command") {
      const model = MODELS.find((m) => m.id === String(args?.model));
      return `codex -c model="${model?.slug}" -c model_reasoning_effort="${model?.reasoning_effort}"`;
    }
    if (command === "codex_launch_config") {
      const model = MODELS.find((m) => m.id === String(args?.model));
      return `model = "${model?.slug}"\nmodel_reasoning_effort = "${model?.reasoning_effort}"`;
    }
    return {};
  }) as never);
}

async function openPanel() {
  await userEvent.click(await screen.findByRole("button", { name: "Launch Codex" }));
  return screen.findByLabelText("Model");
}

beforeEach(() => mocked.mockReset());
afterEach(cleanup);

describe("the model the launch configuration names", () => {
  it("preselects the profile default and emits it with its effort", async () => {
    running("gpt-oss-20b");
    render(<App />);

    const select = await openPanel();

    expect(select).toHaveProperty("value", "gpt-oss-20b");
    expect(await screen.findByText(/model="gpt-oss-20b"/)).toBeTruthy();
    expect(screen.getByText(/model_reasoning_effort="medium"/)).toBeTruthy();
  });

  it("emits the other model's own effort, not the first one's", async () => {
    running("gpt-oss-120b");
    render(<App />);

    await openPanel();

    expect(await screen.findByText(/model="gpt-oss-120b"/)).toBeTruthy();
    expect(screen.getByText(/model_reasoning_effort="high"/)).toBeTruthy();
  });

  it("asks for a model instead of producing a command when there is no default", async () => {
    // The regression: a command with no `model` reads as runnable and sends the
    // session to a cloud model.
    running(null);
    render(<App />);

    const select = await openPanel();

    expect(select).toHaveProperty("value", "");
    expect(screen.getByText(/Choose a model to generate a command/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Copy" })).toBeNull();
    expect(screen.queryByText(/model_provider/)).toBeNull();
  });

  it("generates a runnable command once a model is chosen", async () => {
    running(null);
    render(<App />);
    const select = await openPanel();

    await userEvent.selectOptions(select, "gpt-oss-120b");

    expect(await screen.findByText(/model="gpt-oss-120b"/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Copy" })).toBeTruthy();
  });

  it("selects by stable id but generates the configured served name", async () => {
    running(null);
    render(<App />);
    const select = await openPanel();

    await userEvent.selectOptions(select, "library-7f3a");

    expect(await screen.findByText(/model="codex-local"/)).toBeTruthy();
    expect(mocked).toHaveBeenCalledWith("codex_launch_command", { model: "library-7f3a" });
    expect(
      screen.getByRole("option", {
        name: /My Local Model — served as codex-local — reasoning medium/,
      }),
    ).toBeTruthy();
  });

  it("does not change the profile's default by being chosen", async () => {
    running(null);
    render(<App />);
    const select = await openPanel();

    await userEvent.selectOptions(select, "gpt-oss-20b");
    await screen.findByText(/model="gpt-oss-20b"/);

    // Nothing that writes a profile was called.
    const writes = mocked.mock.calls.map(([c]) => c).filter((c) =>
      ["set_profile", "set_default_profile", "new_profile", "rename_profile"].includes(String(c)),
    );
    expect(writes).toEqual([]);
    expect(screen.getByText(/profile.s default model is unchanged/i)).toBeTruthy();
  });

  it("never renders a placeholder or the word None as a model", async () => {
    running(null);
    render(<App />);
    await openPanel();

    const body = document.body.textContent ?? "";
    expect(body).not.toContain('model="None"');
    expect(body).not.toContain("<model>");
  });

  it("gives the command and the config form the same model and effort", async () => {
    running("gpt-oss-120b");
    render(<App />);
    await openPanel();
    await screen.findByText(/model="gpt-oss-120b"/);

    await userEvent.click(screen.getByRole("tab", { name: "Global / VS Code config" }));

    expect(await screen.findByText(/model = "gpt-oss-120b"/)).toBeTruthy();
    expect(screen.getByText(/model_reasoning_effort = "high"/)).toBeTruthy();
  });

  it("offers every model the server reported, labelled with its effort", async () => {
    running("gpt-oss-20b");
    render(<App />);
    await openPanel();

    expect(screen.getByRole("option", { name: /GPT-OSS 20B — served as gpt-oss-20b — reasoning medium/ })).toBeTruthy();
    expect(screen.getByRole("option", { name: /GPT-OSS 120B — served as gpt-oss-120b — reasoning high/ })).toBeTruthy();
  });
});
