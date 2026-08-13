/** The Models tab.
 *
 * The manual smoke found an installed 20B rendered twice: once as its catalog
 * card, correctly READY, and again as a generic library row. Reconciliation was
 * working; rendering was not using it.
 *
 * The catalogue is the server's answer to "which installed directory is which
 * model". These tests hold the interface to that answer rather than letting it
 * form a second opinion from names or paths.
 */

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Models } from "./Models";

const mocked = vi.mocked(invoke);

/** An installed model, in the shape the library reports. */
function installed(name: string, state = "READY") {
  return {
    name,
    path: `/models/${name}`,
    state,
    usable: state === "READY",
    quantization: "mxfp4-4bit",
    context_length: 131072,
    disk_bytes: 65_318_000_000,
  };
}

function preset(slug: string, display: string, model: unknown = null) {
  return {
    slug,
    display_name: display,
    repo: `mlx-community/${slug}-MXFP4-Q8`,
    parameters: slug.includes("120") ? "120B" : "20B",
    note: "",
    supported: true,
    installed: model !== null,
    model,
  };
}

function respond(catalog: unknown[], library: unknown[] = []) {
  mocked.mockImplementation(async (command: string) => {
    if (command === "model_catalog") return { models: catalog };
    if (command === "list_models") return { models: library, roots: [] };
    if (command === "download_status") return { state: "idle" };
    if (command === "model_config_schema") return { version: 1, groups: [], fields: [] };
    if (command === "model_config") return { settings: {}, effective: {}, inherited: [] };
    return { message: "ok" };
  });
}

beforeEach(() => mocked.mockReset());
afterEach(cleanup);

/** How many times a model's directory name appears on screen. */
function occurrences(text: string): number {
  return document.body.textContent?.split(text).length ?? 1;
}

describe("catalog and library reconciliation", () => {
  it("shows both supported models even with nothing installed", async () => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")]);
    render(<Models />);

    expect(await screen.findByText("GPT-OSS 20B")).toBeTruthy();
    expect(screen.getByText("GPT-OSS 120B")).toBeTruthy();
  });

  it("renders an installed 20B once, as its catalog card", async () => {
    // The regression: the same model as a READY card *and* a library row.
    const local = installed("gpt-oss-20b-mxfp4-bf16");
    respond(
      [preset("gpt-oss-20b", "GPT-OSS 20B", local), preset("gpt-oss-120b", "GPT-OSS 120B")],
      [local],
    );
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    // Once, on the card, as read-only metadata about the installed copy —
    // never a second time as a generic library row.
    expect(occurrences("gpt-oss-20b-mxfp4-bf16") - 1).toBe(1);
    expect(screen.getAllByText("GPT-OSS 20B")).toHaveLength(1);
    expect(screen.queryByRole("list", { name: /other installed models/i })).toBeNull();
  });

  it("renders an installed 120B once, as its catalog card", async () => {
    const local = installed("gpt-oss-120b-mxfp4-bf16");
    respond(
      [preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B", local)],
      [local],
    );
    render(<Models />);

    await screen.findByText("GPT-OSS 120B");
    expect(occurrences("gpt-oss-120b-mxfp4-bf16") - 1).toBe(1);
    expect(screen.getAllByText("GPT-OSS 120B")).toHaveLength(1);
    expect(screen.queryByRole("list", { name: /other installed models/i })).toBeNull();
  });

  it("still lists a model that is not one of the presets", async () => {
    const other = installed("some-other-model");
    respond(
      [
        preset("gpt-oss-20b", "GPT-OSS 20B"),
        preset("gpt-oss-120b", "GPT-OSS 120B"),
        {
          slug: "some-other-model",
          display_name: "some-other-model",
          supported: false,
          installed: true,
          model: other,
        },
      ],
      [other],
    );
    render(<Models />);

    const list = await screen.findByRole("list", { name: /other installed models/i });
    expect(within(list).getByText("some-other-model", { selector: "strong" })).toBeTruthy();
  });

  it("configures an imported model by stable id and shows its served name", async () => {
    const other = installed("some-other-model");
    respond(
      [
        {
          id: "library-7f3a",
          slug: "some-other-model",
          display_name: "My Local Model",
          served_name: "codex-local",
          supported: false,
          installed: true,
          model: other,
        },
      ],
      [other],
    );
    render(<Models />);

    const list = await screen.findByRole("list", { name: /other installed models/i });
    expect(within(list).getByText("My Local Model")).toBeTruthy();
    expect(within(list).getByText("codex-local")).toBeTruthy();
    await userEvent.click(within(list).getByRole("button", { name: "Configure…" }));

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("model_config", { slug: "library-7f3a" }),
    );
  });

  it("keeps MISSING_VOLUME visible on the card rather than hiding the model", async () => {
    const local = installed("gpt-oss-120b-mxfp4-bf16", "MISSING_VOLUME");
    respond(
      [preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B", local)],
      [local],
    );
    render(<Models />);

    expect(await screen.findByText(/MISSING_VOLUME/)).toBeTruthy();
  });

  it("says so when two installed models claim one served name", async () => {
    // Neither is served, and the only other place that fact appears is a line
    // in the server log.
    const other = installed("some-other-model");
    respond(
      [
        {
          ...preset("gpt-oss-20b", "GPT-OSS 20B", installed("gpt-oss-20b-mxfp4-bf16")),
          id: "gpt-oss-20b",
          served_name: "gpt-oss-20b",
          served_conflict: true,
        },
        {
          id: "some-other-model",
          slug: "some-other-model",
          display_name: "Some other model",
          served_name: "gpt-oss-20b",
          served_conflict: true,
          supported: false,
          installed: true,
          model: other,
        },
      ],
      [other],
    );
    render(<Models />);

    const warnings = await screen.findAllByRole("alert");
    expect(warnings).toHaveLength(2);
    expect(warnings[0].textContent).toContain("also served as");
  });
});

describe("a preset that is not installed", () => {
  beforeEach(() => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")]);
  });

  it("offers Download and Locate…, named exactly that", async () => {
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    const cards = within(screen.getByRole("list", { name: /supported models/i }));
    expect(cards.getAllByRole("button", { name: "Download" })).toHaveLength(2);
    expect(cards.getAllByRole("button", { name: "Locate…" })).toHaveLength(2);
    // The card title already says which model this is.
    expect(screen.queryByRole("button", { name: /Download 20B/ })).toBeNull();
  });

  it("downloads the repository the server named, not one held here", async () => {
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    const cards = within(screen.getByRole("list", { name: /supported models/i }));
    await userEvent.click(cards.getAllByRole("button", { name: "Download" })[0]);

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("start_download", {
        repo: "mlx-community/gpt-oss-20b-MXFP4-Q8",
        destination: undefined,
      }),
    );
  });

  it("locates a directory scoped to the card it was pressed on", async () => {
    mocked.mockImplementation(async (command: string) => {
      if (command === "model_catalog")
        return { models: [preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")] };
      if (command === "list_models") return { models: [], roots: [] };
      if (command === "download_status") return { state: "idle" };
      if (command === "choose_model_directory") return "/models/gpt-oss-120b-mxfp4-bf16";
      return { message: "ok" };
    });
    render(<Models />);

    await screen.findByText("GPT-OSS 120B");
    await userEvent.click(screen.getAllByRole("button", { name: "Locate…" })[1]);

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("import_model_for", {
        path: "/models/gpt-oss-120b-mxfp4-bf16",
        expect: "gpt-oss-120b",
      }),
    );
  });

  it("shows the server's refusal when the wrong directory is chosen", async () => {
    // Identity is the server's judgement; the interface reports what it said.
    mocked.mockImplementation(async (command: string) => {
      if (command === "model_catalog")
        return { models: [preset("gpt-oss-120b", "GPT-OSS 120B")] };
      if (command === "list_models") return { models: [], roots: [] };
      if (command === "download_status") return { state: "idle" };
      if (command === "choose_model_directory") return "/models/gpt-oss-20b-mxfp4-bf16";
      if (command === "import_model_for")
        throw new Error("that directory is 'gpt-oss-20b', not 'gpt-oss-120b'");
      return { message: "ok" };
    });
    render(<Models />);

    await screen.findByText("GPT-OSS 120B");
    await userEvent.click(screen.getByRole("button", { name: "Locate…" }));

    expect(await screen.findByText(/not 'gpt-oss-120b'/)).toBeTruthy();
  });
});

describe("the manual Hugging Face download", () => {
  it("is not pre-filled with a preset's repository id", async () => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B")]);
    render(<Models />);

    const field = (await screen.findByPlaceholderText(
      "Paste the HugginFace ID",
    )) as HTMLInputElement;

    expect(field.value).toBe("");
  });

  it("sits below the catalog cards", async () => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")]);
    render(<Models />);

    const card = await screen.findByText("GPT-OSS 120B");
    const field = screen.getByPlaceholderText("Paste the HugginFace ID");

    // DOCUMENT_POSITION_FOLLOWING: the field comes after the card.
    expect(card.compareDocumentPosition(field) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});


describe("installed metadata on a preset card", () => {
  it("shows the directory, quantization, context and size, read-only", async () => {
    respond(
      [preset("gpt-oss-120b", "GPT-OSS 120B", installed("gpt-oss-120b-mxfp4-bf16"))],
      [],
    );
    render(<Models />);

    // The title stays the model; the facts describe the copy on disk.
    expect(await screen.findByText("GPT-OSS 120B")).toBeTruthy();
    expect(screen.getByText("gpt-oss-120b-mxfp4-bf16")).toBeTruthy();
    expect(screen.getByText(/mxfp4-4bit/)).toBeTruthy();
    expect(screen.getByText(/ctx/)).toBeTruthy();
  });

  it("does not make any of it editable", async () => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B", installed("gpt-oss-20b-mxfp4-bf16"))], []);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    // No inputs on the card itself; settings live behind Configure….
    expect(screen.queryByLabelText(/quantization/i)).toBeNull();
    expect(screen.queryByLabelText(/path/i)).toBeNull();
    expect(screen.getByRole("button", { name: "Configure…" })).toBeTruthy();
  });

  it("shows the note instead when the model is not installed", async () => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B")], []);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    expect(screen.queryByText(/mxfp4-4bit/)).toBeNull();
  });
});

/** Download cancellation.
 *
 * The regression: `onCancel` held a bare `return` before the call that reached
 * the server, so automatic semicolon insertion made the request dead code. The
 * button said "Cancelling…" and disabled itself — both driven by local state —
 * while the transfer ran to completion untouched. Every symptom a user could
 * see was correct; the only thing missing was the request.
 *
 * So the assertion that matters is not what the button says. It is that
 * `cancel_download` was invoked.
 */
describe("cancelling a download", () => {
  /** A daemon with a transfer in flight. */
  function downloading(state = "DOWNLOADING") {
    mocked.mockImplementation(async (command: string) => {
      if (command === "model_catalog") return { models: [] };
      if (command === "list_models") return { models: [], roots: [] };
      if (command === "download_status")
        return {
          active: {
            repo: "mlx-community/gpt-oss-20b-MXFP4-Q8",
            state,
            total_bytes: 1000,
            downloaded_bytes: 250,
            fraction: 0.25,
          },
          last: null,
        };
      return { message: "ok" };
    });
  }

  it("invokes the backend cancel command", async () => {
    downloading();
    render(<Models />);

    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(mocked).toHaveBeenCalledWith("cancel_download"));
  });

  it("shows CANCELLING and disables the button on the first click", async () => {
    downloading();
    render(<Models />);
    const cancel = await screen.findByRole("button", { name: "Cancel" });

    await userEvent.click(cancel);

    const cancelling = await screen.findByRole("button", { name: "Cancelling…" });
    expect(cancelling).toHaveProperty("disabled", true);
  });

  it("does not send a second request from a second click", async () => {
    // Not because the backend would refuse — it is idempotent — but because a
    // disabled button that still fires is a control lying about its state.
    downloading();
    render(<Models />);
    const cancel = await screen.findByRole("button", { name: "Cancel" });

    await userEvent.click(cancel);
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("cancel_download"));
    await userEvent.click(cancel);

    const calls = mocked.mock.calls.filter(([command]) => command === "cancel_download");
    expect(calls).toHaveLength(1);
  });

  it("stops offering Cancel once the server reports CANCELLING itself", async () => {
    // The optimistic flag has done its job by now; the server's own state
    // carries it, so the two must not fight over the label.
    downloading("CANCELLING");
    render(<Models />);

    expect(await screen.findByRole("button", { name: "Cancelling…" })).toHaveProperty(
      "disabled",
      true,
    );
  });

  it("reaches the cancelled state with its partial download named", async () => {
    mocked.mockImplementation(async (command: string) => {
      if (command === "model_catalog") return { models: [] };
      if (command === "list_models") return { models: [], roots: [] };
      if (command === "download_status")
        return {
          active: null,
          last: {
            repo: "mlx-community/gpt-oss-20b-MXFP4-Q8",
            state: "CANCELLED",
            detail: "Cancelled. The partial download was kept and can be resumed.",
            total_bytes: 1000,
            downloaded_bytes: 250,
            fraction: 0.25,
          },
        };
      return { message: "ok" };
    });
    render(<Models />);

    expect(await screen.findByText("CANCELLED")).toBeTruthy();
    // Resume is the point of keeping the tree; saying so is what stops a user
    // deleting it and starting sixty gigabytes again.
    expect(await screen.findByText(/can be resumed/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
  });
});

/** Every control that asks "is a download running" must ask the same way.
 *
 * The regression these pin is not "the button is disabled". It is that the
 * question was asked of the wrong object. The server answers `{active, last}`;
 * three separate sites read `download.state` off the top level, where nothing
 * lives. Two were fixed; the third — the guard on each catalogue card's
 * Download button — was not, and it compared against `"running"`, a state the
 * server has never produced. `undefined === "running"` is false, so the guard
 * did nothing at all and no test noticed, because every test asserted on the
 * Downloader panel, which computes the answer for itself.
 *
 * So these assert the property the guard exists for, from both sides: a running
 * transfer must disable the cards, and a decoy state at the top level must not.
 */
describe("controls agree about whether a download is running", () => {
  /** The catalogue with nothing installed, so both cards offer Download. */
  function catalogueWith(downloadStatus: unknown) {
    mocked.mockImplementation(async (command: string) => {
      if (command === "model_catalog")
        return {
          models: [preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")],
        };
      if (command === "list_models") return { models: [], roots: [] };
      if (command === "download_status") return downloadStatus;
      return { message: "ok" };
    });
  }

  /** The catalogue cards' own Download buttons, once the cards exist.
   *
   * Awaited on a card title first: the Downloader panel's button is on screen
   * before the catalogue resolves, so querying immediately finds that one alone
   * and every assertion below would be about the wrong control.
   */
  async function cardDownloadButtons(): Promise<HTMLElement[]> {
    await screen.findByText("GPT-OSS 120B");
    return screen
      .getAllByRole("button", { name: "Download" })
      .filter((button) => button.className.includes("primary"));
  }

  const transfer = {
    active: {
      repo: "mlx-community/gpt-oss-120b-MXFP4-Q8",
      state: "DOWNLOADING",
      total_bytes: 1000,
      downloaded_bytes: 250,
      fraction: 0.25,
    },
    last: null,
  };

  it("disables every catalogue Download button while a transfer is in flight", async () => {
    catalogueWith(transfer);
    render(<Models />);

    const cards = await cardDownloadButtons();
    expect(cards).toHaveLength(2);
    for (const button of cards) expect(button).toHaveProperty("disabled", true);
  });

  it("offers them again once nothing is active, even with a finished transfer recorded", async () => {
    // The counterexample: without it, a guard that disabled the cards
    // unconditionally would pass the test above.
    catalogueWith({ active: null, last: { ...transfer.active, state: "COMPLETED" } });
    render(<Models />);

    const cards = await cardDownloadButtons();
    expect(cards).toHaveLength(2);
    for (const button of cards) expect(button).toHaveProperty("disabled", false);
  });

  it("ignores a `state` sitting at the top level of the envelope", async () => {
    // Exactly the shape a wrong read is fooled by. `active` is null, so nothing
    // is running; a control consulting `download.state` would see DOWNLOADING
    // and disable itself.
    catalogueWith({ state: "DOWNLOADING", active: null, last: null });
    render(<Models />);

    const cards = await cardDownloadButtons();
    expect(cards).toHaveLength(2);
    for (const button of cards) expect(button).toHaveProperty("disabled", false);
  });

  it("treats a state this build does not recognise as still running", async () => {
    // Fail closed: `active` being present is the server saying a transfer
    // exists. Reading an unknown state as "not running" would re-enable every
    // control in the middle of a sixty-gigabyte download.
    catalogueWith({ active: { ...transfer.active, state: "VERIFYING" }, last: null });
    render(<Models />);

    const cards = await cardDownloadButtons();
    for (const button of cards) expect(button).toHaveProperty("disabled", true);
  });
});

/** A download must be startable from a cold app.
 *
 * `daemon::ensure_running` starts the daemon before forwarding the request,
 * precisely so a user never has to visit the Dashboard and press Start first —
 * that is a recorded architecture decision, not a convenience. The interface
 * defeated it: when the daemon is unreachable the view shows a notice, and the
 * notice also disabled the Download button. The one action that would have
 * started the daemon was the one its absence switched off.
 */
describe("starting a download from a cold app", () => {
  /** A daemon that is not answering: every management call fails. */
  function noDaemon() {
    mocked.mockImplementation(async (command: string) => {
      if (command === "model_catalog") return { models: [] };
      if (command === "list_models") return { models: [], roots: [] };
      if (command === "download_status") throw new Error("no server is running");
      if (command === "start_download") return { repo: "owner/model", state: "PENDING" };
      return { message: "ok" };
    });
  }

  async function manualDownloadButton(): Promise<HTMLElement> {
    const buttons = await screen.findAllByRole("button", { name: "Download" });
    const manual = buttons.filter((b) => !b.className.includes("primary"));
    expect(manual).toHaveLength(1);
    return manual[0];
  }

  it("offers Download even though the daemon is not answering", async () => {
    noDaemon();
    render(<Models />);
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("download_status"));

    await userEvent.type(screen.getByPlaceholderText(/HugginFace ID/i), "owner/model");

    expect(await manualDownloadButton()).toHaveProperty("disabled", false);
  });

  it("reaches the backend, which is what starts the daemon", async () => {
    noDaemon();
    render(<Models />);
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("download_status"));
    await userEvent.type(screen.getByPlaceholderText(/HugginFace ID/i), "owner/model");

    await userEvent.click(await manualDownloadButton());

    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("start_download", {
        repo: "owner/model",
        destination: undefined,
      }),
    );
  });

  it("still refuses an empty repository id", async () => {
    // The counterexample: the button is not simply always enabled.
    noDaemon();
    render(<Models />);
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("download_status"));

    expect(await manualDownloadButton()).toHaveProperty("disabled", true);
  });

  it("does not tell the user to go and start the server", async () => {
    noDaemon();
    render(<Models />);

    expect(await screen.findByText(/starting a download will start it/i)).toBeTruthy();
    expect(screen.queryByText(/Start the server to fetch a model/i)).toBeNull();
  });
});

/** The "No models yet" line describes the *custom* list, not the library.
 *
 * It sat directly beneath two READY catalogue cards and said no models were
 * installed. Both statements were on screen at once and only one was true. The
 * emptiness of the non-preset list says nothing about whether anything is
 * installed, so it may not be phrased as though it does.
 */
describe("the empty-state under the catalogue", () => {
  it("says nothing about models when a built-in is installed", async () => {
    const local = installed("gpt-oss-20b-mxfp4-bf16");
    respond(
      [preset("gpt-oss-20b", "GPT-OSS 20B", local), preset("gpt-oss-120b", "GPT-OSS 120B")],
      [local],
    );
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    expect(screen.queryByText(/No models yet/)).toBeNull();
  });

  it("says nothing when both built-ins are installed and READY", async () => {
    // The exact packaged state that produced the contradiction.
    const twenty = installed("gpt-oss-20b-mxfp4-bf16");
    const oneTwenty = installed("gpt-oss-120b-mxfp4-bf16");
    respond(
      [
        preset("gpt-oss-20b", "GPT-OSS 20B", twenty),
        preset("gpt-oss-120b", "GPT-OSS 120B", oneTwenty),
      ],
      [twenty, oneTwenty],
    );
    render(<Models />);

    await screen.findByText("GPT-OSS 120B");
    expect(screen.getAllByText("READY")).toHaveLength(2);
    expect(screen.queryByText(/No models yet/)).toBeNull();
  });

  it("still explains an genuinely empty library", async () => {
    // The counterexample. With nothing installed the message is the only thing
    // telling a user what to do, so removing it outright would be worse.
    respond([preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")]);
    render(<Models />);

    expect(await screen.findByText(/No models yet/)).toBeTruthy();
  });

  it("keeps the real no-model state on the cards themselves", async () => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")]);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    // Download and Locate… are how a missing model is actioned; the sentence is
    // context, not the mechanism.
    expect(screen.getAllByRole("button", { name: "Download" }).length).toBeGreaterThan(1);
    expect(screen.getAllByRole("button", { name: "Locate…" })).toHaveLength(2);
  });
});

/** Configuring a model, over the library rather than below it.
 *
 * The store here is the server: `set_model_config` writes to it and
 * `model_catalog` is answered *from* it. That is what makes the display-name
 * test discriminating -- a card that renders the name the form sent, or one
 * that never re-reads the catalogue, both fail.
 */
describe("the configuration dialog", () => {
  const SCHEMA = {
    version: 1,
    groups: [{ id: "basic", label: "Basic", help: "" }],
    fields: [
      { name: "display_name", label: "Display name", kind: "string", group: "basic", help: "" },
      { name: "served_model_name", label: "Served as", kind: "string", group: "basic", help: "" },
    ],
  };

  /** The library as the server would report it, from the stored overrides. */
  function server(options: { refuse?: string } = {}) {
    const store: Record<string, Record<string, string>> = {};
    const model = installed("my-own-gpt-oss");
    const calls: { command: string; args?: unknown }[] = [];

    const entry = () => ({
      id: "library-7f3a",
      slug: "my-own-gpt-oss",
      display_name: store["library-7f3a"]?.display_name ?? "my-own-gpt-oss",
      served_name: store["library-7f3a"]?.served_model_name ?? "my-own-gpt-oss",
      supported: false,
      installed: true,
      model,
    });

    mocked.mockImplementation(async (command: string, args?: unknown) => {
      calls.push({ command, args });
      const slug = String((args as { slug?: string } | undefined)?.slug);
      if (command === "model_catalog") return { models: [entry()] };
      if (command === "list_models") return { models: [model], roots: [] };
      if (command === "download_status") return { state: "idle" };
      if (command === "model_config_schema") return SCHEMA;
      if (command === "model_storage")
        return { download_root: "/Volumes/Weights/models", available: true };
      if (command === "choose_model_directory") return "/Volumes/Other/models";
      if (command === "set_model_storage") return { message: "ok" };
      if (command === "set_model_config") {
        if (options.refuse) throw new Error(options.refuse);
        const current = { ...(store[slug] ?? {}) };
        for (const assignment of (args as { assignments?: string[] }).assignments ?? []) {
          const at = assignment.indexOf("=");
          const [name, value] = [assignment.slice(0, at), assignment.slice(at + 1)];
          if (value === "") delete current[name];
          else current[name] = value;
        }
        store[slug] = current;
        return { model: slug, settings: current };
      }
      if (command === "model_config") {
        const overrides = store[slug] ?? {};
        return { model: slug, settings: overrides, defaults: {}, effective: { ...overrides }, inherited: [] };
      }
      return { message: "ok" };
    });
    return { store, calls };
  }

  async function openConfigure() {
    const list = await screen.findByRole("list", { name: /other installed models/i });
    await userEvent.click(within(list).getByRole("button", { name: "Configure…" }));
    return screen.findByRole("dialog");
  }

  it("opens a dialog over the page rather than an inline section", async () => {
    server();
    render(<Models />);

    // Nothing before the click: the editor is not part of the page's flow.
    await screen.findByRole("list", { name: /other installed models/i });
    expect(screen.queryByRole("dialog")).toBeNull();

    const dialog = await openConfigure();

    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(dialog.getAttribute("aria-label")).toMatch(/settings/i);
    // The library it belongs to is still on screen behind it.
    expect(screen.getByRole("list", { name: /other installed models/i })).toBeTruthy();
  });

  it("edits an imported model by its stable library id", async () => {
    const { calls } = server();
    render(<Models />);
    await openConfigure();

    expect(calls.filter((c) => c.command === "model_config")).toEqual([
      { command: "model_config", args: { slug: "library-7f3a" } },
    ]);
  });

  it("shows the new display name on the card as soon as the save lands", async () => {
    // The regression: the value persisted, the dialog closed, and the card kept
    // the name the catalogue had been read with. It took a tab switch -- a
    // remount -- to show the truth.
    server();
    render(<Models />);
    const list = await screen.findByRole("list", { name: /other installed models/i });
    expect(within(list).getByText("my-own-gpt-oss", { selector: "strong" })).toBeTruthy();

    await openConfigure();
    await userEvent.type(screen.getByLabelText(/display name/i), "Fable 120B");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(await within(list).findByText("Fable 120B", { selector: "strong" })).toBeTruthy();
    expect(within(list).queryByText("my-own-gpt-oss", { selector: "strong" })).toBeNull();
  });

  it("shows the new served name where the served name is shown", async () => {
    server();
    render(<Models />);
    const list = await screen.findByRole("list", { name: /other installed models/i });

    await openConfigure();
    await userEvent.type(screen.getByLabelText(/served as/i), "codex-local");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(await within(list).findByText("codex-local")).toBeTruthy();
  });

  it("keeps the stable id across a rename of either name", async () => {
    const { store, calls } = server();
    render(<Models />);

    await openConfigure();
    await userEvent.type(screen.getByLabelText(/display name/i), "Renamed");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await openConfigure();

    // Same id before and after; the settings never moved to a name-shaped key.
    expect(Object.keys(store)).toEqual(["library-7f3a"]);
    for (const call of calls.filter((c) => c.command.endsWith("model_config"))) {
      expect((call.args as { slug: string }).slug).toBe("library-7f3a");
    }
  });

  it("writes nothing when it is closed without saving", async () => {
    const { store, calls } = server();
    render(<Models />);
    await openConfigure();

    await userEvent.type(screen.getByLabelText(/display name/i), "Never saved");
    await userEvent.click(screen.getByRole("button", { name: "Close" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(calls.some((c) => c.command === "set_model_config")).toBe(false);
    expect(store).toEqual({});
  });

  it("opens for a built-in model too, keyed by its own id", async () => {
    // Presets are configurable for the same reason imports are: served name and
    // reasoning effort are theirs. The dialog is reached from the card.
    const local = installed("gpt-oss-20b-mxfp4-bf16");
    respond([{ ...preset("gpt-oss-20b", "GPT-OSS 20B", local), id: "gpt-oss-20b" }], [local]);
    render(<Models />);

    const cards = await screen.findByRole("list", { name: /supported models/i });
    await userEvent.click(within(cards).getByRole("button", { name: "Configure…" }));

    expect(await screen.findByRole("dialog")).toBeTruthy();
    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("model_config", { slug: "gpt-oss-20b" }),
    );
  });

  it("stays open, with the server's words, when a value is refused", async () => {
    server({ refuse: "served name 'codex-local' is claimed by 'gpt-oss-20b'" });
    render(<Models />);
    await openConfigure();

    await userEvent.type(screen.getByLabelText(/served as/i), "codex-local");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText(/is claimed by/)).toBeTruthy();
    expect(screen.getByRole("dialog")).toBeTruthy();
    // And the refused value is still in the form to be corrected.
    expect((screen.getByLabelText(/served as/i) as HTMLInputElement).value).toBe("codex-local");
  });
});

describe("where downloads go", () => {
  it("is not a control on the Models page itself", async () => {
    respond([preset("gpt-oss-20b", "GPT-OSS 20B")]);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    expect(screen.queryByText(/download location/i)).toBeNull();
    expect(mocked).not.toHaveBeenCalledWith("model_storage", expect.anything());
  });

  it("is not reachable through the configure dialog either", async () => {
    // It is global, so it belongs with the global settings. Reaching it from a
    // dialog named after one model is what made it look like the model's.
    const other = installed("some-other-model");
    respond(
      [
        {
          id: "library-7f3a",
          slug: "some-other-model",
          display_name: "My Local Model",
          supported: false,
          installed: true,
          model: other,
        },
      ],
      [other],
    );
    render(<Models />);

    const list = await screen.findByRole("list", { name: /other installed models/i });
    await userEvent.click(within(list).getByRole("button", { name: "Configure…" }));
    const dialog = await screen.findByRole("dialog");

    expect(within(dialog).queryByText(/download location/i)).toBeNull();
    expect(mocked).not.toHaveBeenCalledWith("model_storage", expect.anything());
    expect(mocked).not.toHaveBeenCalledWith("model_storage");
  });
});

/** The catalogue is what you have; installing is how you get more.
 *
 * Before this, "Import existing…" and "Scan roots" sat in the same action row a
 * user reaches for after picking a model, and the Hugging Face field trailed the
 * page with no heading tying the three together. The property these hold is not
 * "the section exists": it is that each acquisition control is inside it and
 * none of them is inside the catalogue.
 */
describe("install more models", () => {
  const both = [preset("gpt-oss-20b", "GPT-OSS 20B"), preset("gpt-oss-120b", "GPT-OSS 120B")];

  function install(): HTMLElement {
    return screen.getByRole("region", { name: /install more models/i });
  }

  it("gives the acquisition workflows a section of their own", async () => {
    respond(both);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    expect(install()).toBeTruthy();
  });

  it("puts the Hugging Face download in it", async () => {
    respond(both);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    const section = within(install());
    expect(section.getByLabelText(/repository id/i)).toBeTruthy();
    expect(section.getByPlaceholderText(/HugginFace ID/i)).toBeTruthy();
    expect(section.getByRole("button", { name: "Download" })).toBeTruthy();
  });

  it("puts Import existing in it", async () => {
    respond(both);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    const section = within(install());
    expect(section.getByText(/import existing/i)).toBeTruthy();
    expect(section.getByRole("button", { name: /choose folder/i })).toBeTruthy();
  });

  it("puts Scan roots in it", async () => {
    respond(both);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    const section = within(install());
    expect(section.getByText(/scan roots/i)).toBeTruthy();
    expect(section.getByRole("button", { name: "Scan" })).toBeTruthy();
  });

  it("keeps every one of them out of the catalogue", async () => {
    // Containment in both directions: the acquisition controls are not among
    // the cards, and no card's own actions leaked into the section.
    respond(both);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    const catalog = within(screen.getByRole("list", { name: /supported models/i }));
    expect(catalog.queryByPlaceholderText(/HugginFace ID/i)).toBeNull();
    expect(catalog.queryByRole("button", { name: /choose folder/i })).toBeNull();
    expect(catalog.queryByRole("button", { name: "Scan" })).toBeNull();

    const section = within(install());
    expect(section.queryByText("GPT-OSS 20B")).toBeNull();
    expect(section.queryByRole("button", { name: "Locate…" })).toBeNull();
    expect(section.queryByRole("button", { name: "Configure…" })).toBeNull();
  });

  it("puts the three tiles in one stretching row, so none is shorter than another", async () => {
    // Equal height is CSS, but it only works if the three tiles are siblings of
    // one grid container: a tile nested a level deeper, or a wrapper around one
    // of them, sizes itself and the row goes ragged again. That is the
    // structural precondition, and it is what this asserts.
    respond(both);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    const row = install().querySelector(".install-tiles")!;
    expect(row.querySelectorAll(":scope > .install-tile")).toHaveLength(3);
    expect(row.querySelectorAll(".install-tile")).toHaveLength(3);
  });

  it("ends every tile with its action, so the buttons share a line", async () => {
    // `margin-top: auto` on the action row is what pulls the buttons to the
    // bottom of a stretched tile. It can only do that if the action really is
    // the last thing in each tile.
    respond(both);
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    for (const tile of install().querySelectorAll(".install-tile")) {
      const last = tile.lastElementChild!;
      expect(last.className).toContain("actions");
      expect(last.querySelector("button")).toBeTruthy();
    }
  });

  it("sits below the catalogue, not among it", async () => {
    respond(both);
    render(<Models />);

    const card = await screen.findByText("GPT-OSS 120B");
    // DOCUMENT_POSITION_FOLLOWING: the whole section comes after the last card.
    expect(card.compareDocumentPosition(install()) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
  });

  it("still imports and scans through the same backend commands", async () => {
    mocked.mockImplementation(async (command: string) => {
      if (command === "model_catalog") return { models: both };
      if (command === "list_models") return { models: [], roots: [] };
      if (command === "download_status") return { state: "idle" };
      if (command === "choose_model_directory") return "/models/my-own-gpt-oss";
      return { message: "ok" };
    });
    render(<Models />);

    await screen.findByText("GPT-OSS 20B");
    await userEvent.click(within(install()).getByRole("button", { name: /choose folder/i }));
    await waitFor(() =>
      expect(mocked).toHaveBeenCalledWith("import_model", { path: "/models/my-own-gpt-oss" }),
    );

    await userEvent.click(within(install()).getByRole("button", { name: "Scan" }));
    await waitFor(() => expect(mocked).toHaveBeenCalledWith("scan_models"));
  });
});
