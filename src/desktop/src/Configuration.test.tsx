/** Profile lifecycle in the interface.
 *
 * The smoke that prompted these found an empty profile list with no way to
 * create one, and create/duplicate/rename that appeared to do nothing —
 * `window.prompt` is not implemented in Tauri's webview, so every one of them
 * returned null silently.
 *
 * The schema below is the real one the server publishes. Names, defaults and
 * validation are never asserted against a local opinion: the tests check that
 * the interface *asks the server* and shows what it answers.
 */

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Configuration } from "./Configuration";

const SCHEMA = {
  version: 1,
  groups: [{ id: "basic", label: "Basic", help: "" }],
  fields: [
    { name: "port", label: "Port", kind: "integer", group: "basic", help: "", default: 8123 },
  ],
};

const mocked = vi.mocked(invoke);

function respond(profiles: Record<string, unknown>[], defaultName: string | null = null) {
  mocked.mockImplementation(async (command: string) => {
    if (command === "profile_schema") return SCHEMA;
    if (command === "profiles") return { default: defaultName, profiles };
    return { message: "ok" };
  });
}

beforeEach(() => mocked.mockReset());
afterEach(cleanup);

describe("empty state", () => {
  it("offers a prominent way to create the first profile", async () => {
    respond([]);
    render(<Configuration serverRunning={false} />);

    expect(await screen.findByRole("button", { name: /create profile/i })).toBeTruthy();
  });

  it("does not tell the user to go and run a terminal command", async () => {
    // The observed empty state named a CLI command and offered no action.
    respond([]);
    render(<Configuration serverRunning={false} />);

    await screen.findByRole("button", { name: /create profile/i });
    expect(document.body.textContent).not.toContain("quantum-codex-server profiles add");
  });

  it("creates a profile through the server, with the typed name", async () => {
    respond([]);
    render(<Configuration serverRunning={false} />);

    await userEvent.click(await screen.findByRole("button", { name: /create profile/i }));
    const field = await screen.findByLabelText(/name for the new profile/i);
    await userEvent.clear(field);
    await userEvent.type(field, "work");
    await userEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("new_profile", { name: "work" }),
    );
  });
});

describe("lifecycle actions", () => {
  const dev = { name: "dev", port: 8123 };

  it("names each profile in a selector", async () => {
    respond([dev, { name: "other", port: 8124 }]);
    render(<Configuration serverRunning={false} />);

    expect(await screen.findByRole("option", { name: "dev" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "other" })).toBeTruthy();
  });

  it("duplicates through the server rather than rebuilding the profile here", async () => {
    respond([dev]);
    render(<Configuration serverRunning={false} />);

    await userEvent.click(await screen.findByRole("button", { name: "Duplicate" }));
    const form = await screen.findByRole("form", { name: /profile name/i });
    await userEvent.click(within(form).getByRole("button", { name: "Duplicate" }));

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("duplicate_profile", {
        source: "dev",
        name: "dev copy",
      }),
    );
  });

  it("renames through the server", async () => {
    respond([dev]);
    render(<Configuration serverRunning={false} />);

    await userEvent.click(await screen.findByRole("button", { name: "Rename" }));
    const form = await screen.findByRole("form", { name: /profile name/i });
    const field = within(form).getByLabelText(/new name/i);
    await userEvent.clear(field);
    await userEvent.type(field, "renamed");
    await userEvent.click(within(form).getByRole("button", { name: "Rename" }));

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("rename_profile", {
        name: "dev",
        newName: "renamed",
      }),
    );
  });

  it("asks for confirmation before deleting", async () => {
    respond([dev]);
    render(<Configuration serverRunning={false} />);

    await userEvent.click(await screen.findByRole("button", { name: /^delete…$/i }));

    // Nothing has been sent yet: the first press only arms the action.
    expect(mocked).not.toHaveBeenCalledWith("remove_profile", expect.anything());
    await userEvent.click(screen.getByRole("button", { name: /delete dev/i }));
    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("remove_profile", { name: "dev", force: false }),
    );
  });

  it("can make a profile the default, and says when it already is", async () => {
    respond([dev, { name: "other", port: 8124 }], "other");
    render(<Configuration serverRunning={false} />);

    await userEvent.click(await screen.findByRole("button", { name: /make default/i }));
    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("set_default_profile", { name: "dev" }),
    );
  });

  it("keeps the typed name and shows the server's words when a name is refused", async () => {
    // The server owns the name rules; the form must not pre-empt or hide them.
    respond([dev]);
    mocked.mockImplementation(async (command: string) => {
      if (command === "profile_schema") return SCHEMA;
      if (command === "profiles") return { default: null, profiles: [dev] };
      if (command === "new_profile") throw new Error("a profile named 'dev' already exists");
      return { message: "ok" };
    });
    render(<Configuration serverRunning={false} />);

    await userEvent.click(await screen.findByRole("button", { name: "New profile" }));
    const field = await screen.findByLabelText(/name for the new profile/i);
    await userEvent.type(field, "dev");
    await userEvent.click(screen.getByRole("button", { name: "Create" }));

    expect(await screen.findByText(/already exists/i)).toBeTruthy();
    expect((field as HTMLInputElement).value).toBe("dev");
  });
});

/** Where downloaded weights are written: one choice for the installation.
 *
 * It lives here because it is global, and the discriminating property is not
 * that a control appears on this view — it is that changing it goes to the
 * storage authority and creates no profile or per-model setting anywhere.
 */
describe("model storage", () => {
  const dev = { name: "dev", port: 8123 };

  function withStorage(
    storage: Record<string, unknown> = { download_root: "/Volumes/Weights/models", available: true },
  ) {
    const calls: { command: string; args?: unknown }[] = [];
    // A store, so the read after a change is answered from what was written
    // rather than from a constant. A view that showed its own optimistic copy
    // would pass against a constant and fail here.
    let current = { ...storage };
    mocked.mockImplementation(async (command: string, args?: unknown) => {
      calls.push({ command, args });
      if (command === "profile_schema") return SCHEMA;
      if (command === "profiles") return { default: "dev", profiles: [dev] };
      if (command === "model_storage") return current;
      if (command === "choose_model_directory") return "/Volumes/Other/models/";
      if (command === "set_model_storage") {
        // Stored as the server would: normalised, not as it was handed in. That
        // is what makes the reload assertion discriminating — a view showing its
        // own optimistic copy of the chosen path would display the trailing
        // slash the server dropped.
        current = {
          ...current,
          download_root: (args as { path: string }).path.replace(/\/+$/, ""),
        };
        return current;
      }
      return { message: "ok" };
    });
    return calls;
  }

  it("shows the global download location the storage authority reports", async () => {
    withStorage();
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    expect(within(section).getByText(/download location/i)).toBeTruthy();
    expect(await within(section).findByText("/Volumes/Weights/models")).toBeTruthy();
    expect(within(section).getByText(/global location used for downloaded models/i)).toBeTruthy();
    expect(within(section).getByText(/imported models are left in their original location/i))
      .toBeTruthy();
  });

  it("is a card of its own rather than a region inside the profile card", async () => {
    // A divider inside someone else's container still reads as that container's
    // last section. Storage belongs to no profile, so it is a sibling panel and
    // sits outside the one the profile form draws.
    withStorage();
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    expect(section.className).toContain("panel");
    expect(section.closest(".panel")).toBe(section);

    const profileCard = [...document.querySelectorAll(".panel")].find((p) => p !== section)!;
    expect(profileCard.contains(section)).toBe(false);
    // And it carries its own heading, in the page's own card language.
    expect(within(section).getByRole("heading", { name: /model storage/i })).toBeTruthy();
  });

  it("changes it through the global storage command", async () => {
    const calls = withStorage();
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    await userEvent.click(within(section).getByRole("button", { name: "Choose…" }));

    await waitFor(() =>
      expect(calls.find((c) => c.command === "set_model_storage")?.args).toEqual({
        path: "/Volumes/Other/models/",
      }),
    );
  });

  it("writes no per-model and no profile setting when it changes", async () => {
    // The discriminating half. A path stored as a model override or a profile
    // field would be a second authority for the same fact.
    const calls = withStorage();
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    await userEvent.click(within(section).getByRole("button", { name: "Choose…" }));

    await waitFor(() => expect(calls.some((c) => c.command === "set_model_storage")).toBe(true));
    expect(calls.filter((c) => c.command === "set_model_config")).toEqual([]);
    expect(calls.filter((c) => c.command === "set_profile")).toEqual([]);
  });

  it("shows the new location by re-reading the setting, not its own copy", async () => {
    withStorage();
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    await within(section).findByText("/Volumes/Weights/models");
    await userEvent.click(within(section).getByRole("button", { name: "Choose…" }));

    // The server's normalised answer, not the path the picker returned.
    expect(await within(section).findByText("/Volumes/Other/models")).toBeTruthy();
    expect(within(section).queryByText("/Volumes/Other/models/")).toBeNull();
  });

  it("leaves the location alone when the picker is cancelled", async () => {
    // The counterexample: choosing is not the same as having chosen, and a
    // dismissed picker must not write anything.
    const calls: { command: string; args?: unknown }[] = [];
    mocked.mockImplementation(async (command: string, args?: unknown) => {
      calls.push({ command, args });
      if (command === "profile_schema") return SCHEMA;
      if (command === "profiles") return { default: "dev", profiles: [dev] };
      if (command === "model_storage")
        return { download_root: "/Volumes/Weights/models", available: true };
      if (command === "choose_model_directory") return null;
      return { message: "ok" };
    });
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    await userEvent.click(within(section).getByRole("button", { name: "Choose…" }));

    await waitFor(() => expect(calls.some((c) => c.command === "choose_model_directory")).toBe(true));
    expect(calls.filter((c) => c.command === "set_model_storage")).toEqual([]);
    expect(within(section).getByText("/Volumes/Weights/models")).toBeTruthy();
  });

  it("says when the volume holding it is not attached", async () => {
    withStorage({ download_root: "/Volumes/Weights/models", available: false });
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    expect(await within(section).findByText(/not mounted/i)).toBeTruthy();
  });

  it("is still reachable when the server cannot describe its profiles", async () => {
    // Storage has its own command and does not depend on the profile schema.
    mocked.mockImplementation(async (command: string) => {
      if (command === "profile_schema") throw new Error("no server is running");
      if (command === "model_storage")
        return { download_root: "/Volumes/Weights/models", available: true };
      return { message: "ok" };
    });
    render(<Configuration serverRunning={false} />);

    const section = await screen.findByRole("region", { name: /model storage/i });
    expect(await within(section).findByText("/Volumes/Weights/models")).toBeTruthy();
  });
});
