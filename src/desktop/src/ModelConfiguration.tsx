// One model's own settings.
//
// Rendered from the schema the server publishes for models, with the same
// `Setting` component the profile form uses: two forms, one description of what
// a setting is and how to show it. Nothing here knows a default or a bound.
//
// Keyed by the model's stable id, so these values follow the model rather than
// the directory it currently occupies.

import { useCallback, useEffect, useState } from "react";

import * as api from "./api";
import { Setting } from "./Configuration";

export function ModelConfiguration({
  slug,
  displayName,
  onClose,
}: {
  slug: string;
  displayName: string;
  onClose: () => void;
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
      setResult(`Saved ${displayName}.`);
      await refresh();
    } catch (cause) {
      // The server's words about which value it refused.
      setFailure(String(cause));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel model-config" aria-label={`${displayName} settings`}>
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
    </section>
  );
}
