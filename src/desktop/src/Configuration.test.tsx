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
