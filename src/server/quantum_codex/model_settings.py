"""Per-model configuration, keyed by stable model identity.

Some settings belong to the daemon -- the port it listens on, how large the
prompt cache may grow -- and some belong to a *model*. A reasoning effort that
suits the 20B is not the one that suits the 120B, and an alias names one set of
weights. Holding both kinds in one profile meant whichever model happened to be
loaded inherited settings chosen for another.

Keyed by slug, never by path. A path is library metadata: it changes when a
volume is remounted or a directory renamed, and settings keyed by one would be
lost by an operation that changed nothing about the model. The slug is derived
from the model directory by the same rule the catalogue reconciles with, so a
model that disappears and comes back is recognised as itself.

**Lifetime.** Overrides outlive the library entry deliberately. Removing a model
from the library is an inventory action -- it leaves the files alone -- and an
unplugged volume is temporary by nature. Discarding a user's settings on either
would make "forget and re-import" a destructive operation for something they
never asked to change. Settings are removed only when a caller asks explicitly.

Absence is meaningful: no override means *use the backend default*, not "the
default was chosen". Storing resolved defaults would freeze them, so a later
improvement to a default would never reach anyone who had opened the form once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import ConfigError, app_support_dir, read_json, write_json

logger = logging.getLogger(__name__)

MODEL_SETTINGS_VERSION = 1


def model_settings_path():
    return app_support_dir() / "model-settings.json"


@dataclass
class ModelSettings:
    """Every model's overrides, by slug.

    Values are stored *sparsely*: only what a user actually set. Reading a
    setting nobody chose returns nothing, and the caller falls back to the
    backend default.
    """

    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def for_model(self, slug: str) -> dict[str, Any]:
        return dict(self.overrides.get(slug, {}))

    def set(self, slug: str, values: dict[str, Any]) -> dict[str, Any]:
        """Merge ``values`` into one model's overrides.

        ``None`` clears a setting rather than storing a null, so "inherit the
        default" and "explicitly set to nothing" cannot be confused.
        """
        current = dict(self.overrides.get(slug, {}))
        for key, value in values.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        if current:
            self.overrides[slug] = current
        else:
            self.overrides.pop(slug, None)
        return dict(current)

    def clear(self, slug: str) -> None:
        """Forget one model's settings. Only ever on an explicit request."""
        self.overrides.pop(slug, None)


def load_model_settings() -> ModelSettings:
    data = read_json(model_settings_path(), expected_version=MODEL_SETTINGS_VERSION)
    if data is None:
        return ModelSettings()

    raw = data.get("models") or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{model_settings_path()} has no `models` object")

    overrides: dict[str, dict[str, Any]] = {}
    for slug, values in raw.items():
        if not isinstance(values, dict):
            raise ConfigError(f"settings for model {slug!r} are not an object")
        overrides[slug] = dict(values)
    return ModelSettings(overrides=overrides)


def save_model_settings(settings: ModelSettings) -> None:
    write_json(
        model_settings_path(),
        {"version": MODEL_SETTINGS_VERSION, "models": settings.overrides},
    )



def migrate_from_profiles(profiles: Any, settings: ModelSettings) -> list[str]:
    """Move model-specific values out of profiles and onto their model.

    A profile that names exactly one model is unambiguous: whatever
    model-specific settings it carried were chosen while that model was the one
    being served, so they belong to it.

    A profile that names *no* model is ambiguous, and the values stay where they
    are. Guessing would attach settings chosen for one model to another, which
    is worse than leaving them parked -- they are preserved on the profile and
    move automatically once it names a model.

    Existing overrides win: a value the user has already set on the model is
    their current intent, and a migration must not walk it back.

    Returns the names of profiles whose settings could not be attributed.
    """
    unattributed: list[str] = []
    for profile in profiles.profiles.values():
        legacy = getattr(profile, "legacy_model_settings", None)
        if not legacy:
            continue

        slug = profile.default_model
        if not slug:
            unattributed.append(profile.name)
            logger.warning(
                "profile %r carries model settings (%s) but names no model; they are kept "
                "and will move once a default model is chosen",
                profile.name,
                ", ".join(sorted(legacy)),
            )
            continue

        existing = settings.for_model(slug)
        moved = {key: value for key, value in legacy.items() if key not in existing}
        if moved:
            settings.set(slug, moved)
            logger.info(
                "moved %s from profile %r to model %r",
                ", ".join(sorted(moved)),
                profile.name,
                slug,
            )
        profile.legacy_model_settings = {}
    return unattributed
