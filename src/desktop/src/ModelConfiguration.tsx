// One model's own settings, in a dialog over the library.
//
// Rendered from the schema the server publishes for models, with the same
// `Setting` component the profile form uses: two forms, one description of what
// a setting is and how to show it. Nothing here knows a default or a bound.
//
// Keyed by the model's stable id, so these values follow the model rather than
// the directory it currently occupies.
//
// Everything in this dialog is owned by this model. Where downloads are written
// is not: it is one setting for the installation, so it lives with the other
// global settings on the Configuration view and nothing here reads or writes
// it. A control that changes global state from a dialog titled after one model
// invites the reading that the two are connected, and they are not.

import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import { Setting } from "./Configuration";

export function ModelConfiguration({
  slug,
  displayName,
  onClose,
  onSaved,
}: {
  slug: string;
  displayName: string;
  onClose: () => void;
  /** A write landed on the server. The owner of the model list re-reads it.
   *
   *  Called rather than the parent guessing: a saved display name changes what
   *  the library says about this model, and the list behind this dialog is
   *  showing the answer from before the write.
   */
  onSaved?: (summary: string) => void;
}) {
  const [schema, setSchema] = useState<api.ProfileSchema | null>(null);
  // `effective` is what the model will actually use; `inherited` names the
  // fields whose value came from the backend rather than from this user. The
  // form shows the effective value so it is never blank when a default exists,
  // and knows not to write an inherited value back as an override.
  const [effective, setEffective] = useState<Record<string, unknown>>({});
  const [inherited, setInherited] = useState<string[]>([]);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [described, current] = await Promise.all([
        api.modelConfigSchema(),
        api.modelConfig(slug),
      ]);
      setSchema(described);
      setEffective((api.pick(current, "effective") as Record<string, unknown>) ?? {});
      setInherited((api.pick(current, "inherited") as string[]) ?? []);
      setFailure(null);
    } catch (cause) {
      setFailure(String(cause));
    }
  }, [slug]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  /** An unsaved edit if there is one, else what is stored, else empty.
   *
   * Empty means "inherit the server default" — not zero, and not a default
   * copied into this form. The server decides what an absent value means.
   */
  function shown(name: string): string {
    if (name in edits) return edits[name];
    const value = effective[name];
    return value === undefined || value === null ? "" : String(value);
  }

  /** Whether this field's value comes from the backend rather than the user. */
  function isInherited(name: string): boolean {
    return !(name in edits) && inherited.includes(name);
  }

  /** Clear one override, returning the field to the backend default. */
  async function reset(name: string) {
    setSaving(true);
    setResult(null);
    setFailure(null);
    try {
      await api.setModelConfig(slug, [`${name}=`]);
      setEdits((current) => {
        const next = { ...current };
        delete next[name];
        return next;
      });
      await refresh();
      setResult(`Reset ${name.replace(/_/g, " ")} to the default.`);
      // A reset changes what the library reports just as a save does; the
      // dialog stays open because clearing one field is not finishing.
      onSaved?.(`Reset ${name.replace(/_/g, " ")} for ${displayName}.`);
    } catch (cause) {
      setFailure(String(cause));
    } finally {
      setSaving(false);
    }
  }

  const dirty = Object.keys(edits).length > 0;

  async function save() {
    setSaving(true);
    setResult(null);
    setFailure(null);
    try {
      const assignments = Object.entries(edits).map(([name, value]) => `${name}=${value}`);
      await api.setModelConfig(slug, assignments);
      setEdits({});
      setSaving(false);
      // Told before closing, so the list behind this dialog is re-read from the
      // server rather than left showing what it read before the save.
      onSaved?.(`Saved ${displayName}.`);
      onClose();
    } catch (cause) {
      // The server's words about which value it refused. The dialog stays open:
      // a refused value is still in the form, and closing would discard it
      // along with the explanation.
      setFailure(String(cause));
      setSaving(false);
    }
  }

  // Escape closes, and the first control takes focus on open. Not a native
  // `<dialog>`: `showModal` is unimplemented in the DOM these tests run in, and
  // a modal whose behaviour cannot be asserted is a modal nobody is checking.
  //
  // Both effects run once, on mount, and neither depends on `onClose`. The
  // owner of this dialog re-renders on a timer -- it polls download state --
  // and a focus effect keyed on a prop the parent recreates each render would
  // take the caret out of whichever field was being typed into, every two
  // seconds. The handler is read through a ref so the listener can be
  // installed once and still call the current one.
  const panel = useRef<HTMLDivElement | null>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close.current();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    panel.current?.querySelector("button")?.focus();
  }, []);

  return (
    // The list stays visible underneath: what is being configured is one row of
    // it, and losing sight of that was the reason to stop pushing this into the
    // page below.
    <div className="modal-scrim" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        ref={panel}
        className="panel model-config modal-panel"
        role="dialog"
        aria-modal="true"
        aria-label={`${displayName} settings`}
      >
        <div className="catalog-head">
          <h3>{displayName} settings</h3>
          <button onClick={onClose}>Close</button>
        </div>

        <p className="catalog-note">
          These apply to {displayName} only. Values shown in grey are the defaults this model
          would use; changing one saves an override for this model alone.
        </p>

        <div className="actions">
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

        {schema?.fields.map((field) => (
          <div key={field.name} className="model-setting">
            <Setting
              field={field}
              value={shown(field.name)}
              serverRunning={false}
              onChange={(value) => setEdits((current) => ({ ...current, [field.name]: value }))}
              inherited={isInherited(field.name)}
            />
            {!isInherited(field.name) && effective[field.name] !== undefined && (
              <button
                className="link"
                disabled={saving}
                onClick={() => void reset(field.name)}
              >
                Reset to default
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
