// The Rust command surface, and nothing else.
//
// Server responses are deliberately typed as `unknown` and read defensively.
// Declaring their shape here would be a second definition of what the Python
// server already owns, and the two would drift the first time a capability is
// added — which is exactly what the desktop app is supposed to avoid.

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface Discovery {
  connected: boolean;
  endpoint: string | null;
  model: string | null;
  detail: string | null;
  /** A problem reading `runtime.json`, which is not the same as an unhealthy
   *  daemon and must never be shown as one. */
  metadataError: string | null;
  /** Found by probing rather than through the runtime file. */
  adopted: boolean;
  /** Management actions need the token the runtime file carries. */
  manageable: boolean;
}

export type RuntimeState =
  | "UNINITIALIZED"
  | "INITIALIZING"
  | "READY"
  | "UPDATE_REQUIRED"
  | "BROKEN";

export interface RuntimeStatus {
  state: RuntimeState;
  /** Which executable would be used: managed environment, checkout, override. */
  source: string | null;
  envPath: string;
  appVersion: string;
  installedVersion: string | null;
  expectedFingerprint: string | null;
  installedFingerprint: string | null;
  /** Whether this build carries the resources needed to initialise. */
  installable: boolean;
  detail: string | null;
}

export interface SearchedLocation {
  description: string;
  found: boolean;
}

export interface ServerEnvironment {
  resolved: { program: string; args: string[]; source: string } | null;
  searched: SearchedLocation[];
  inheritedPath: string;
}

export type BootstrapEvent =
  | { kind: "step"; message: string }
  | { kind: "output"; line: string }
  | { kind: "done" }
  | { kind: "failed"; message: string };

export const discover = () => invoke<Discovery>("daemon_discover");
export const serverEnvironment = () => invoke<ServerEnvironment>("server_environment");
export const runtimeStatus = () => invoke<RuntimeStatus>("runtime_status");
export const runtimeInitialize = () => invoke<void>("runtime_initialize");
/** Model configuration, straight from the CLI: the app forms no opinion on it. */
export const modelStatus = () => invoke<unknown>("model_status");
export const onBootstrap = (handler: (event: BootstrapEvent) => void) =>
  listen<BootstrapEvent>("bootstrap", (event) => handler(event.payload));
export const status = () => invoke<unknown>("daemon_status");
export const start = (profile?: string) => invoke<void>("daemon_start", { profile });
export const stop = () => invoke<void>("daemon_stop");
export const restart = (profile?: string) => invoke<void>("daemon_restart", { profile });
export const clearCache = () => invoke<unknown>("cache_clear");
/** Release the resident model. The daemon keeps running; the next request
 *  loads it again. Refused by the server while inference is in flight. */
export const unloadModel = () => invoke<unknown>("model_unload");
export const tailLogs = (lines = 300) => invoke<string[]>("logs_tail", { lines });
/** One model this server can be pointed at, as the server resolves it. */
export interface LaunchModel {
  /** Stable library identity used by desktop selectors and profiles. */
  id: string;
  /** The served id `/v1/models` publishes and the server routes on. */
  slug: string;
  /** Human-facing name shown by QCS. */
  display_name: string | null;
  /** Effective effort: catalogue default, then the per-model override. */
  reasoning_effort: string | null;
}

export interface LaunchModels {
  /** What the generator would pick unprompted — the profile's default, or the
   *  single installed model. `null` when the user must choose. */
  /** Stable library id, not the mutable served name. */
  default: string | null;
  models: LaunchModel[];
}

/** The choices for the Launch Codex selector, resolved by the backend.
 *
 *  Asked for rather than assembled here: which models exist and what effort
 *  each runs at are the library's and the per-model configuration's answers.
 */
export const codexLaunchModels = () => invoke<LaunchModels>("codex_launch_models");
/** `model` pins the generated configuration to one model. Passed through to
 *  the generator, never spliced into its output: the reasoning effort that
 *  travels with the model is resolved server-side. */
export const codexLaunchCommand = (model?: string) =>
  invoke<string>("codex_launch_command", { model });
/** The persistent `~/.codex/config.toml` fragment, for the Codex CLI's global
 *  configuration and the VS Code extension. Same generator as the command. */
export const codexLaunchConfig = (model?: string) =>
  invoke<string>("codex_launch_config", { model });

// -- model library ----------------------------------------------------------
//
// All of these return the server's own JSON. The library's schema — states,
// volume facts, validation verdicts — lives on the server, so a state added
// there needs no type change here.

export const listModels = () => invoke<unknown>("list_models");
/** Supported GPT-OSS models joined with what is installed, decided by the server. */
export const modelCatalog = () => invoke<unknown>("model_catalog");
/** Where downloads are written. Application-wide, server-owned. */
export const modelStorage = () => invoke<unknown>("model_storage");
export const setModelStorage = (path: string) => invoke<unknown>("set_model_storage", { path });
/** Per-model settings: schema, current overrides, and changes. All server-owned. */
export const modelConfigSchema = () => invoke<ProfileSchema>("model_config_schema");
export const modelConfig = (slug: string) => invoke<unknown>("model_config", { slug });
export const setModelConfig = (slug: string, assignments: string[]) =>
  invoke<unknown>("set_model_config", { slug, assignments });
export const scanModels = () => invoke<unknown>("scan_models");
export const importModel = (path: string) => invoke<unknown>("import_model", { path });
/** Import a directory that must be a particular catalog model. The server
 *  decides whether it is; a mismatch is refused rather than misfiled. */
export const importModelFor = (path: string, expect: string) =>
  invoke<unknown>("import_model_for", { path, expect });
export const forgetModel = (path: string) => invoke<unknown>("forget_model", { path });
export const chooseModelDirectory = () => invoke<string | null>("choose_model_directory");
export const revealInFinder = (path: string) => invoke<void>("reveal_in_finder", { path });

// -- profiles ---------------------------------------------------------------
//
// The schema is fetched, never declared here. That is what lets a setting be
// added on the server and appear in the form without a change to this file.

export interface SchemaField {
  name: string;
  label: string;
  kind: "string" | "integer" | "number" | "choice" | "path";
  group: string;
  help: string;
  default?: unknown;
  choices?: string[];
  choice_labels?: Record<string, string>;
  minimum?: number;
  maximum?: number;
  unit?: string;
  restart_required?: boolean;
  caution?: string;
  required?: boolean;
  nullable?: boolean;
}

export interface ProfileSchema {
  version: number;
  groups: { id: string; label: string; help: string }[];
  fields: SchemaField[];
}

export const profileSchema = () => invoke<ProfileSchema>("profile_schema");
export const profiles = () => invoke<unknown>("profiles");
export const setProfile = (name: string, assignments: string[]) =>
  invoke<unknown>("set_profile", { name, assignments });
export const newProfile = (name: string) => invoke<unknown>("new_profile", { name });
export const duplicateProfile = (source: string, name: string) =>
  invoke<unknown>("duplicate_profile", { source, name });
export const renameProfile = (name: string, newName: string) =>
  invoke<unknown>("rename_profile", { name, newName });
export const removeProfile = (name: string, force = false) =>
  invoke<unknown>("remove_profile", { name, force });
export const setDefaultProfile = (name: string) =>
  invoke<unknown>("set_default_profile", { name });

/** Diagnostics, already interpreted by the server. */
export const requestDiagnostics = (limit = 50) =>
  invoke<unknown>("request_diagnostics", { limit });

export const downloadStatus = () => invoke<unknown>("download_status");
export const startDownload = (repo: string, destination?: string) =>
  invoke<unknown>("start_download", { repo, destination });
export const cancelDownload = () => invoke<unknown>("cancel_download");

/** The states `library/downloads.py: DownloadState` can report.
 *
 * Named here, as a union, for one reason: comparisons against it are then
 * checked. A guard once read `=== "running"` — a value the server has never
 * produced — and because it was compared against `unknown` it was simply always
 * false, so a control that should have been disabled never was. The compiler
 * now rejects that spelling instead of the interface silently doing nothing.
 */
export const DOWNLOAD_STATES = [
  "PENDING",
  "DOWNLOADING",
  "CANCELLING",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
] as const;

export type DownloadState = (typeof DOWNLOAD_STATES)[number];

/** The transfer that is running right now, or `null`.
 *
 * The server answers `{active, last}`: `active` is the one still moving bytes
 * and becomes `null` the moment it finishes, `last` is the outcome of the one
 * before. Reading `state` off the envelope's top level — which is not where it
 * lives — yields `undefined`, which compares false against everything and so
 * disables nothing and clears every optimistic flag. Every caller goes through
 * here so there is one place that can be wrong, and it has a test.
 */
export function activeDownload(status: unknown): Record<string, unknown> | null {
  const active = pick(status, "active");
  return active !== null && typeof active === "object"
    ? (active as Record<string, unknown>)
    : null;
}

/** Whether a transfer is in flight at all.
 *
 * Deliberately independent of the state string: `active` being present *is* the
 * server saying a transfer exists, so a state this build has not heard of still
 * counts as running rather than silently reading as idle.
 */
export function isDownloading(status: unknown): boolean {
  return activeDownload(status) !== null;
}

/** The active transfer's state, or `null` when nothing is running.
 *
 * An unrecognised value reads as `null` so a comparison cannot accidentally
 * succeed; use {@link isDownloading} when the question is "is anything running".
 */
export function downloadState(status: unknown): DownloadState | null {
  const value = activeDownload(status)?.state;
  return DOWNLOAD_STATES.includes(value as DownloadState) ? (value as DownloadState) : null;
}

/** Seconds as "4m 20s", or "—" when the server did not report an estimate. */
/** A rate, or "—" when the server reported none. Never invents a zero. */
export function rate(value: number | undefined, unit: string): string {
  if (value === undefined || value === null) return "—";
  return `${value.toFixed(1)} ${unit}`;
}

export function percent(value: number | undefined): string {
  if (value === undefined || value === null) return "—";
  return `${Math.round(value * 100)}%`;
}

export function duration(seconds: number | undefined): string {
  if (seconds === undefined) return "—";
  const whole = Math.round(seconds);
  if (whole < 60) return `${whole}s`;
  return `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, "0")}s`;
}

/** Read a nested value without asserting a shape over the whole response. */
export function pick(source: unknown, ...path: string[]): unknown {
  let current: unknown = source;
  for (const key of path) {
    if (current === null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

export function text(source: unknown, ...path: string[]): string {
  const value = pick(source, ...path);
  if (value === undefined || value === null) return "—";
  return String(value);
}

export function count(source: unknown, ...path: string[]): number | undefined {
  const value = pick(source, ...path);
  return typeof value === "number" ? value : undefined;
}

export function bytes(value: number | undefined): string {
  if (value === undefined) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = value;
  for (const unit of units) {
    if (size < 1024 || unit === "TiB") {
      return unit === "B" ? `${size} B` : `${size.toFixed(1)} ${unit}`;
    }
    size /= 1024;
  }
  return `${size} B`;
}

export function tokens(value: number | undefined): string {
  if (value === undefined) return "—";
  return value.toLocaleString("en-US");
}
