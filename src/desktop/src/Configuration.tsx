// Server profiles.
//
// The form is *generated* from the schema the server publishes: which settings
// exist, what they accept, what they mean, and which need a restart. Nothing
// here decides whether a value is acceptable — that judgement stays where it is
// enforced, so the form cannot allow something the server would then refuse.
//
// The practical consequence: adding a setting on the server makes it appear
// here with its label, help and bounds, and this file does not change.
//
// One thing on this view is not part of a profile: where downloaded weights are
// written. It is a single choice for the installation, it has its own server
// command, and it is rendered below the form behind a rule so it cannot read as
// a field of whichever profile happens to be selected.

import { useCallback, useEffect, useState } from "react";

import * as api from "./api";

type PendingKind = "new" | "duplicate" | "rename";

const PENDING_LABEL: Record<PendingKind, string> = {
  new: "Name for the new profile",
  duplicate: "Name for the copy",
  rename: "New name",
};

const PENDING_ACTION: Record<PendingKind, string> = {
  new: "Create",
  duplicate: "Duplicate",
  rename: "Rename",
};

export function Configuration({ serverRunning }: { serverRunning: boolean }) {
  const [schema, setSchema] = useState<api.ProfileSchema | null>(null);
  const [profiles, setProfiles] = useState<Record<string, unknown>[]>([]);
  const [defaultProfile, setDefaultProfile] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  // Naming happens in an inline form, not `window.prompt`: Tauri's webview does
  // not implement it, so every create/duplicate/rename silently did nothing.
  const [pending, setPending] = useState<{ kind: PendingKind; name: string } | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [description, listing] = await Promise.all([api.profileSchema(), api.profiles()]);
      setSchema(description);
      const found = (api.pick(listing, "profiles") as Record<string, unknown>[]) ?? [];
      setDefaultProfile((api.pick(listing, "default") as string | null) ?? null);
      setProfiles(found);
      // A selection that no longer exists must not survive a delete or a
      // rename: the form would keep editing a profile that is gone.
      const names = found.map((entry) => String(entry.name));
      setSelected((current) =>
        current && names.includes(current) ? current : (names[0] ?? null),
      );
      setFailure(null);
    } catch (cause) {
      setFailure(String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const profile = profiles.find((entry) => entry.name === selected);

  /** The value shown: an unsaved edit if there is one, else what is stored. */
  function shown(field: api.SchemaField): string {
    if (field.name in edits) return edits[field.name];
    const stored = profile?.[field.name];
    return stored === null || stored === undefined ? "" : String(stored);
  }

  /** Run one lifecycle operation and report what actually happened.
   *
   * Every one of these is a call to the server, which owns the defaults, the
   * name rules and the collision checks. Nothing here decides what a profile
   * contains or what a valid name is.
   */
  async function lifecycle(
    label: string,
    run: () => Promise<unknown>,
    { select, message }: { select?: string; message: string },
  ): Promise<boolean> {
    setBusy(label);
    setResult(null);
    setFailure(null);
    try {
      await run();
      if (select !== undefined) {
        setSelected(select);
        setEdits({});
      }
      setResult(message);
      await refresh();
      return true;
    } catch (cause) {
      // The server's own words: it explains which rule was broken. The form
      // stays open with what the user typed, so a rejected name can be edited
      // rather than retyped.
      setFailure(String(cause));
      return false;
    } finally {
      setBusy(null);
      setConfirmDelete(false);
    }
  }


  function begin(kind: PendingKind, initial: string) {
    setPending({ kind, name: initial });
    setConfirmDelete(false);
    setResult(null);
    setFailure(null);
  }

  /** Run the pending create/duplicate/rename.
   *
   * The name is sent as typed. Trimming, character rules and collisions are the
   * server's to enforce, and a second opinion here would eventually disagree
   * with it.
   */
  async function submitPending() {
    if (!pending) return;
    const name = pending.name.trim();
    if (!name) return;
    const source = selected;

    const run =
      pending.kind === "new"
        ? () => api.newProfile(name)
        : pending.kind === "duplicate"
          ? () => api.duplicateProfile(source ?? "", name)
          : () => api.renameProfile(source ?? "", name);

    const message =
      pending.kind === "new"
        ? `Created ${name}.`
        : pending.kind === "duplicate"
          ? `Duplicated ${source} as ${name}.`
          : `Renamed to ${name}.`;

    const ok = await lifecycle(pending.kind, run, { select: name, message });
    if (ok) setPending(null);
  }

  async function save() {
    if (!selected) return;
    setSaving(true);
    setResult(null);
    setFailure(null);
    try {
      // Sent as `field=value` pairs so the server parses, coerces and validates
      // them — the same path the CLI takes.
      const assignments = Object.entries(edits).map(([name, value]) => `${name}=${value}`);
      if (assignments.length === 0) {
        setResult("Nothing changed.");
        return;
      }
      await api.setProfile(selected, assignments);
      setEdits({});
      await refresh();

      const restarts = (schema?.fields ?? [])
        .filter((field) => field.restart_required && field.name in edits)
        .map((field) => field.label);
      setResult(
        restarts.length > 0 && serverRunning
          ? // Said plainly: otherwise a saved-but-not-applied setting reads as
            // one that did not save.
            `Saved. ${restarts.join(", ")} ${restarts.length === 1 ? "takes" : "take"} effect after a restart.`
          : "Saved.",
      );
    } catch (cause) {
      // The server's own words, listing every field it rejected.
      setFailure(String(cause));
    } finally {
      setSaving(false);
    }
  }

  const dirty = Object.keys(edits).length > 0;

  // One return rather than an early one, so `ModelStorage` below is mounted
  // once for the life of the view. Rendering it in two branches instead gave it
  // two lifetimes: it was unmounted and remounted the moment the profile schema
  // arrived, and re-read a global setting because a profile-shaped answer had
  // landed. Its lifetime does not depend on the profile schema, so its position
  // in the tree must not either.
  return (
    <>
      <section className="panel">
        <h2>Configuration</h2>

        {!schema ? (
          failure ? (
            <div className="notice notice-error">{failure}</div>
          ) : (
            <p className="empty">Reading the settings the server publishes…</p>
          )
        ) : (
          <>
          <div className="actions">
            <button
              disabled={busy !== null || pending !== null}
              onClick={() => begin("new", "")}
            >
              New profile
            </button>
            <button
              disabled={busy !== null || pending !== null || !selected}
              onClick={() => begin("duplicate", `${selected} copy`)}
            >
              Duplicate
            </button>
            <button
              disabled={busy !== null || pending !== null || !selected}
              onClick={() => begin("rename", selected ?? "")}
            >
              Rename
            </button>
            {/* Two steps, because deleting a profile discards settings that exist
                nowhere else. */}
            {confirmDelete ? (
              <>
                <button
                  className="danger"
                  disabled={busy !== null || !selected}
                  onClick={() => {
                    if (!selected) return;
                    void lifecycle("delete", () => api.removeProfile(selected), {
                      message: `Deleted ${selected}.`,
                    });
                  }}
                >
                  {busy === "delete" ? "Deleting…" : `Delete ${selected}`}
                </button>
                <button disabled={busy !== null} onClick={() => setConfirmDelete(false)}>
                  Cancel
                </button>
              </>
            ) : (
              <button
                disabled={busy !== null || pending !== null || !selected}
                onClick={() => {
                  setConfirmDelete(true);
                  setResult(null);
                  setFailure(null);
                }}
              >
                Delete…
              </button>
            )}
            <button
              disabled={busy !== null || !selected || selected === defaultProfile}
              onClick={() => {
                if (!selected) return;
                void lifecycle("default", () => api.setDefaultProfile(selected), {
                  message: `${selected} is now the default.`,
                });
              }}
            >
              {selected && selected === defaultProfile ? "Is default" : "Make default"}
            </button>
          </div>

          {pending && (
            <form
              className="actions"
              aria-label="Profile name"
              onSubmit={(event) => {
                event.preventDefault();
                void submitPending();
              }}
            >
              <label htmlFor="profile-name">{PENDING_LABEL[pending.kind]}</label>
              <input
                id="profile-name"
                className="repo-input"
                autoFocus
                value={pending.name}
                onChange={(event) =>
                  setPending((current) => (current ? { ...current, name: event.target.value } : null))
                }
              />
              <button type="submit" disabled={busy !== null || !pending.name.trim()}>
                {busy ? "Working…" : PENDING_ACTION[pending.kind]}
              </button>
              <button type="button" disabled={busy !== null} onClick={() => setPending(null)}>
                Cancel
              </button>
            </form>
          )}

          {profiles.length === 0 ? (
            <>
              <div className="empty-cta">
                <p>No profiles yet. A profile holds this server's settings — port, limits, defaults.</p>
                <button
                  className="primary"
                  disabled={busy !== null || pending !== null}
                  onClick={() => begin("new", "dev")}
                >
                  Create profile
                </button>
              </div>
              {result && <div className="notice notice-ok">{result}</div>}
              {failure && (
                <div className="notice notice-error" role="alert">
                  {failure}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="actions">
                <select
                  className="repo-input"
                  value={selected ?? ""}
                  onChange={(event) => {
                    setSelected(event.target.value);
                    setEdits({});
                    setResult(null);
                    setFailure(null);
                  }}
                >
                  {profiles.map((entry) => (
                    <option key={String(entry.name)} value={String(entry.name)}>
                      {String(entry.name)}
                    </option>
                  ))}
                </select>
                <button disabled={!dirty || saving} onClick={() => void save()}>
                  {saving ? "Saving…" : "Save"}
                </button>
                <button disabled={!dirty || saving} onClick={() => setEdits({})}>
                  Discard
                </button>
              </div>

              {result && <div className="notice notice-ok">{result}</div>}
              {failure && (
                <div className="notice notice-error" role="alert">
                  {failure}
                </div>
              )}

              {schema.groups.map((group) => {
                const fields = schema.fields.filter((field) => field.group === group.id);
                if (fields.length === 0) return null;
                return (
                  <fieldset key={group.id} className="settings-group">
                    <legend>{group.label}</legend>
                    <p className="empty">{group.help}</p>
                    {fields.map((field) => (
                      <Setting
                        key={field.name}
                        field={field}
                        value={shown(field)}
                        serverRunning={serverRunning}
                        onChange={(value) => setEdits((current) => ({ ...current, [field.name]: value }))}
                      />
                    ))}
                  </fieldset>
                );
              })}
            </>
          )}
          </>
        )}
      </section>

      {/* A sibling card, not a ruled-off region inside the profile card. It
          belongs to no profile, and a divider drawn inside someone else's
          container still reads as part of that container. */}
      <ModelStorage />
    </>
  );
}

/** Where downloaded weights are written. One choice for the installation.
 *
 * Not a profile field and not a model setting: it is read and written through
 * the server's own storage command, which is the single authority for it. It is
 * on this view because this is where global settings are, and it was briefly in
 * the per-model dialog, where it read as something a model could own.
 */
function ModelStorage() {
  const [storage, setStorage] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  // A daemon that is not answering is not an error worth alarming about here:
  // the location is still shown as unknown and the button still works, because
  // choosing one is what a user came to do.
  const refresh = useCallback(async () => {
    setStorage((await api.modelStorage().catch(() => null)) as Record<string, unknown> | null);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function choose() {
    setBusy(true);
    setFailure(null);
    try {
      const chosen = await api.chooseModelDirectory();
      // A cancelled picker changes nothing; re-reading is still right, because
      // it is the server's answer either way.
      if (chosen) await api.setModelStorage(chosen);
      await refresh();
    } catch (cause) {
      // The server's own words about the directory it refused.
      setFailure(String(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel model-storage" aria-label="Model storage">
      <h2>Model storage</h2>
      <div className="setting">
        <span className="setting-label">Download location</span>
        <div className="storage">
          <code className="library-path">{String(storage?.download_root ?? "—")}</code>
          <button disabled={busy} onClick={() => void choose()}>
            Choose…
          </button>
          {storage && storage.available === false && (
            <span className="pill pill-down">volume not mounted</span>
          )}
        </div>
        <p className="setting-help">
          Global location used for downloaded models. Imported models are left in their
          original location.
        </p>
      </div>
      {failure && (
        <div className="notice notice-error" role="alert">
          {failure}
        </div>
      )}
    </section>
  );
}

export function Setting({
  field,
  value,
  serverRunning,
  onChange,
  inherited = false,
}: {
  field: api.SchemaField;
  value: string;
  serverRunning: boolean;
  onChange: (value: string) => void;
  /** The value shown came from the backend, not from this user. Rendered
   *  quieter so a default is visibly not a choice someone made. */
  inherited?: boolean;
}) {
  return (
    <div className={inherited ? "setting setting-inherited" : "setting"}>
      <label className="setting-label" htmlFor={field.name}>
        {field.label}
        {field.unit && <span className="setting-unit"> ({field.unit})</span>}
        {field.restart_required && serverRunning && (
          <span className="pill pill-warn">restart</span>
        )}
      </label>

      {field.choices ? (
        <select
          id={field.name}
          className="repo-input"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          {field.choices.map((choice) => (
            <option key={choice} value={choice}>
              {/* The server may name a choice better than its raw value —
                  "" is a real selection meaning "load nothing", not a blank. */}
              {field.choice_labels?.[choice] ?? choice}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={field.name}
          className="repo-input"
          value={value}
          spellCheck={false}
          /* `inputMode` rather than `type="number"`. It gives a numeric field
             the numeric affordance without handing the browser authority over
             the value: `type="number"` reports an unparseable entry as the
             empty string, so a mistyped digit would silently erase what the
             user had, and its spinner clamps to `max` instead of letting the
             server say why a value was refused. Bounds stay the server's to
             enforce, which is what keeps the form from becoming a second
             opinion about what is valid. */
          inputMode={
            field.kind === "integer" ? "numeric" : field.kind === "number" ? "decimal" : undefined
          }
          placeholder={field.nullable ? "inherit" : ""}
          onChange={(event) => onChange(event.target.value)}
        />
      )}

      <p className="setting-help">{field.help}</p>
      {field.caution && <p className="setting-caution">{field.caution}</p>}
    </div>
  );
}
