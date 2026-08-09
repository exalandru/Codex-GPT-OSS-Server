// Tauri's `invoke` is the only thing these components reach the outside world
// through, so it is the single seam the tests replace. Everything else — the
// server's schema, its validation, its wording — is exercised as real data
// captured from the CLI, not re-invented here.
import { vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
