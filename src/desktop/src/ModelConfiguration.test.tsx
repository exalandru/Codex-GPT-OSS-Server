/** Per-model settings in the interface.
 *
 * The 20B and the 120B are configured independently. These tests hold the two
 * properties that matter: each form reads and writes its own model's settings
 * by stable id, and neither the defaults nor the validation live here.
 */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ModelConfiguration } from "./ModelConfiguration";
import { Configuration } from "./Configuration";

const MODEL_SCHEMA = {
  version: 1,
  groups: [{ id: "basic", label: "Basic", help: "" }],
  fields: [
    {
      name: "reasoning_effort",
      label: "Reasoning effort",
      kind: "choice",
      group: "basic",
      help: "",
      choices: ["low", "medium", "high"],
    },
    { name: "served_model_name", label: "Served as", kind: "string", group: "basic", help: "" },
    { name: "context_length", label: "Context length", kind: "integer", group: "basic", help: "" },
  ],
};

/** What the backend resolves for the presets, before any override. */
const DEFAULTS: Record<string, Record<string, unknown>> = {
  "gpt-oss-20b": {
    served_model_name: "gpt-oss-20b",
    reasoning_effort: "medium",
    context_length: 131072,
  },
  "gpt-oss-120b": {
    served_model_name: "gpt-oss-120b",
    reasoning_effort: "medium",
    context_length: 131072,
  },
};

const mocked = vi.mocked(invoke);

function withSettings(bySlug: Record<string, Record<string, unknown>>) {
  mocked.mockImplementation(async (command: string, args?: unknown) => {
    if (command === "model_config_schema") return MODEL_SCHEMA;
    if (command === "model_config") {
      const slug = String((args as { slug?: string } | undefined)?.slug);
      const overrides = bySlug[slug] ?? {};
      const defaults = DEFAULTS[slug] ?? {};
      const effective = { ...defaults, ...overrides };
      return {
        model: slug,
        settings: overrides,
        defaults,
        effective,
        inherited: Object.keys(effective).filter((k) => !(k in overrides)),
      };
    }
    return { message: "ok" };
  });
}

beforeEach(() => mocked.mockReset());
afterEach(cleanup);

describe("a model's own settings", () => {
  it("shows the fields the server declares for a model", async () => {
    withSettings({});
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    expect(await screen.findByLabelText(/reasoning effort/i)).toBeTruthy();
    expect(screen.getByLabelText(/served as/i)).toBeTruthy();
  });

  it("never offers the filesystem path as a preference", async () => {
    withSettings({});
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    await screen.findByLabelText(/reasoning effort/i);
    expect(screen.queryByLabelText(/path/i)).toBeNull();
  });

  it("reads the settings of the model it was opened for", async () => {
    withSettings({ "gpt-oss-120b": { reasoning_effort: "high" } });
    render(
      <ModelConfiguration slug="gpt-oss-120b" displayName="GPT-OSS 120B" onClose={() => {}} />,
    );

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("model_config", { slug: "gpt-oss-120b" }),
    );
    expect(((await screen.findByLabelText(/reasoning effort/i)) as HTMLSelectElement).value).toBe(
      "high",
    );
  });

  it("shows each model its own value", async () => {
    withSettings({
      "gpt-oss-20b": { reasoning_effort: "low" },
      "gpt-oss-120b": { reasoning_effort: "high" },
    });

    const twenty = render(
      <ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />,
    );
    expect(((await screen.findByLabelText(/reasoning effort/i)) as HTMLSelectElement).value).toBe(
      "low",
    );
    twenty.unmount();

    render(<ModelConfiguration slug="gpt-oss-120b" displayName="GPT-OSS 120B" onClose={() => {}} />);
    expect(((await screen.findByLabelText(/reasoning effort/i)) as HTMLSelectElement).value).toBe(
      "high",
    );
  });

  it("saves against the model it is configuring, not another", async () => {
    withSettings({ "gpt-oss-20b": { reasoning_effort: "medium" } });
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    await userEvent.selectOptions(await screen.findByLabelText(/reasoning effort/i), "low");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("set_model_config", {
        slug: "gpt-oss-20b",
        assignments: ["reasoning_effort=low"],
      }),
    );
  });

  it("leaves a field blank when the backend has no value to show", async () => {
    // A preset shows its catalogue default; a model the user imported has no
    // shipped opinion, and blank is the honest answer.
    withSettings({});
    render(<ModelConfiguration slug="my-own" displayName="My own" onClose={() => {}} />);

    const field = (await screen.findByLabelText(/served as/i)) as HTMLInputElement;

    expect(field.value).toBe("");
  });

  it("shows the server's refusal rather than validating here", async () => {
    mocked.mockImplementation(async (command: string, args?: unknown) => {
      if (command === "model_config_schema") return MODEL_SCHEMA;
      if (command === "model_config") return { model: (args as { slug?: string } | undefined)?.slug, settings: {} };
      if (command === "set_model_config")
        throw new Error("context_length: Context length must be at most 131072");
      return { message: "ok" };
    });
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    await userEvent.type(await screen.findByLabelText(/context length/i), "999999");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText(/must be at most/i)).toBeTruthy();
  });
});

describe("the profile form after the split", () => {
  it("no longer exposes model-specific settings", async () => {
    // They belong to a model now; two owners would be two sources of truth.
    const PROFILE_SCHEMA = {
      version: 1,
      groups: [{ id: "basic", label: "Basic", help: "" }],
      fields: [
        { name: "port", label: "Port", kind: "integer", group: "basic", help: "" },
        {
          name: "model",
          label: "Default model",
          kind: "choice",
          group: "basic",
          help: "",
          choices: ["", "gpt-oss-20b"],
          choice_labels: { "": "None — load on demand" },
        },
      ],
    };
    mocked.mockImplementation(async (command: string) => {
      if (command === "profile_schema") return PROFILE_SCHEMA;
      if (command === "profiles") return { default: "dev", profiles: [{ name: "dev", port: 8123 }] };
      return { message: "ok" };
    });
    render(<Configuration serverRunning={false} />);

    expect(await screen.findByLabelText(/default model/i)).toBeTruthy();
    for (const gone of [/reasoning effort/i, /served as/i, /temperature/i, /maximum output/i]) {
      expect(screen.queryByLabelText(gone)).toBeNull();
    }
  });
});


describe("effective defaults on first open", () => {
  it("shows the 20B's resolved defaults with no override stored", async () => {
    withSettings({});
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    expect(((await screen.findByLabelText(/served as/i)) as HTMLInputElement).value).toBe(
      "gpt-oss-20b",
    );
    expect((screen.getByLabelText(/reasoning effort/i) as HTMLSelectElement).value).toBe("medium");
    expect((screen.getByLabelText(/context length/i) as HTMLInputElement).value).toBe("131072");
  });

  it("shows the 120B's own resolved defaults", async () => {
    withSettings({});
    render(
      <ModelConfiguration slug="gpt-oss-120b" displayName="GPT-OSS 120B" onClose={() => {}} />,
    );

    expect(((await screen.findByLabelText(/served as/i)) as HTMLInputElement).value).toBe(
      "gpt-oss-120b",
    );
  });

  it("does not write anything merely by opening", async () => {
    withSettings({});
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    await screen.findByLabelText(/served as/i);
    expect(mocked).not.toHaveBeenCalledWith("set_model_config", expect.anything());
  });

  it("invents nothing for a model the backend has no defaults for", async () => {
    withSettings({});
    render(<ModelConfiguration slug="my-own" displayName="My own" onClose={() => {}} />);

    expect(((await screen.findByLabelText(/served as/i)) as HTMLInputElement).value).toBe("");
  });

  it("offers Reset only for a value the user actually set", async () => {
    withSettings({ "gpt-oss-20b": { reasoning_effort: "low" } });
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    await screen.findByLabelText(/reasoning effort/i);
    // One override -> exactly one reset control.
    expect(screen.getAllByRole("button", { name: /reset to default/i })).toHaveLength(1);
  });

  it("resets an override by clearing it on the server", async () => {
    withSettings({ "gpt-oss-20b": { reasoning_effort: "low" } });
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    await userEvent.click(await screen.findByRole("button", { name: /reset to default/i }));

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("set_model_config", {
        slug: "gpt-oss-20b",
        assignments: ["reasoning_effort="],
      }),
    );
  });

  it("marks an inherited value as not the user's own", async () => {
    withSettings({});
    render(<ModelConfiguration slug="gpt-oss-20b" displayName="GPT-OSS 20B" onClose={() => {}} />);

    const field = await screen.findByLabelText(/served as/i);
    expect(field.closest(".setting")?.className).toContain("setting-inherited");
  });
});

/** An imported model is configured with this same form, and its edits stick.
 *
 * The store here answers `model_config` from what `set_model_config` wrote, so
 * a form that only appeared to save -- writing an assignment and then showing
 * its own optimistic copy -- fails on reopen.
 */
describe("an imported model", () => {
  const SCHEMA_WITH_DISPLAY = {
    ...MODEL_SCHEMA,
    fields: [
      { name: "display_name", label: "Display name", kind: "string", group: "basic", help: "" },
      ...MODEL_SCHEMA.fields,
    ],
  };

  function withStore(store: Record<string, Record<string, unknown>>) {
    mocked.mockImplementation(async (command: string, args?: unknown) => {
      const slug = String((args as { slug?: string } | undefined)?.slug);
      if (command === "model_config_schema") return SCHEMA_WITH_DISPLAY;
      if (command === "set_model_config") {
        const assignments = (args as { assignments?: string[] }).assignments ?? [];
        const current = { ...(store[slug] ?? {}) };
        for (const assignment of assignments) {
          const [name, value] = [
            assignment.slice(0, assignment.indexOf("=")),
            assignment.slice(assignment.indexOf("=") + 1),
          ];
          if (value === "") delete current[name];
          else current[name] = value;
        }
        store[slug] = current;
        return { model: slug, settings: current };
      }
      if (command === "model_config") {
        const overrides = store[slug] ?? {};
        return {
          model: slug,
          settings: overrides,
          defaults: {},
          effective: { ...overrides },
          inherited: [],
        };
      }
      return { message: "ok" };
    });
  }

  it("keeps a display name and a served name after a save and a reopen", async () => {
    const store: Record<string, Record<string, unknown>> = {};
    withStore(store);

    const first = render(
      <ModelConfiguration slug="library-7f3a" displayName="my-own-gpt-oss" onClose={() => {}} />,
    );
    await userEvent.type(await screen.findByLabelText(/display name/i), "My Local Model");
    await userEvent.type(screen.getByLabelText(/served as/i), "codex-local");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("set_model_config", {
        slug: "library-7f3a",
        assignments: ["display_name=My Local Model", "served_model_name=codex-local"],
      }),
    );
    first.unmount();

    // Reopened by the same stable id, which is what the identity is for.
    render(
      <ModelConfiguration slug="library-7f3a" displayName="My Local Model" onClose={() => {}} />,
    );

    expect(((await screen.findByLabelText(/display name/i)) as HTMLInputElement).value).toBe(
      "My Local Model",
    );
    expect((screen.getByLabelText(/served as/i) as HTMLInputElement).value).toBe("codex-local");
    expect(store["library-7f3a"]).toEqual({
      display_name: "My Local Model",
      served_model_name: "codex-local",
    });
  });

  it("shows the server's refusal of a name rather than deciding for itself", async () => {
    withStore({});
    mocked.mockImplementationOnce(async () => SCHEMA_WITH_DISPLAY);
    render(
      <ModelConfiguration slug="library-7f3a" displayName="my-own-gpt-oss" onClose={() => {}} />,
    );
    await screen.findByLabelText(/served as/i);
    mocked.mockImplementation(async (command: string) => {
      if (command === "set_model_config") throw new Error("served name 'codex-local' is claimed");
      return SCHEMA_WITH_DISPLAY;
    });

    await userEvent.type(screen.getByLabelText(/served as/i), "codex-local");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText(/is claimed/)).toBeTruthy();
  });
});
