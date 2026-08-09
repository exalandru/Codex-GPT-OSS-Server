"""Per-model configuration.

The 20B and the 120B are different models with different useful settings. These
tests pin the two properties that make that safe: settings follow the *model*,
not the directory it happens to live in, and neither model's values can be
changed by editing the other's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from quantum_codex.config import ConfigError, Profiles, load_profiles, save_profiles
from quantum_codex.library.registry import ModelState
from quantum_codex.model_settings import (
    ModelSettings,
    load_model_settings,
    migrate_from_profiles,
    save_model_settings,
)
from quantum_codex.models import served_models_from_library
from quantum_codex.profile_schema import validate_model


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    return tmp_path


@dataclass
class FakeEntry:
    name: str
    path: str


@dataclass
class FakeReport:
    entry: FakeEntry
    state: ModelState = ModelState.READY
    context_length: int = 131072
    quantization: str = "mxfp4-4bit"


def report(name: str, path: str | None = None) -> FakeReport:
    return FakeReport(entry=FakeEntry(name=name, path=path or f"/models/{name}"))


BOTH = [report("gpt-oss-20b-mxfp4-bf16"), report("gpt-oss-120b-mxfp4-bf16")]


# -- independence ------------------------------------------------------------


def test_two_models_hold_different_settings_at_once() -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low"})
    settings.set("gpt-oss-120b", {"reasoning_effort": "high"})

    assert settings.for_model("gpt-oss-20b")["reasoning_effort"] == "low"
    assert settings.for_model("gpt-oss-120b")["reasoning_effort"] == "high"


def test_editing_one_model_leaves_the_other_untouched() -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low", "served_model_name": "fast"})
    settings.set("gpt-oss-120b", {"reasoning_effort": "high"})

    settings.set("gpt-oss-120b", {"reasoning_effort": "medium", "temperature": 0.2})

    assert settings.for_model("gpt-oss-20b") == {
        "reasoning_effort": "low",
        "served_model_name": "fast",
    }


def test_an_unset_value_means_inherit_rather_than_a_stored_default() -> None:
    """Storing a resolved default would freeze it against later improvement."""
    settings = ModelSettings()

    assert settings.for_model("gpt-oss-20b") == {}


def test_clearing_one_setting_leaves_the_rest() -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low", "temperature": 0.5})

    settings.set("gpt-oss-20b", {"temperature": None})

    assert settings.for_model("gpt-oss-20b") == {"reasoning_effort": "low"}


# -- persistence and identity ------------------------------------------------


def test_settings_survive_a_reload(home) -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-120b", {"reasoning_effort": "high"})
    save_model_settings(settings)

    assert load_model_settings().for_model("gpt-oss-120b")["reasoning_effort"] == "high"


def test_settings_are_keyed_by_model_not_by_path() -> None:
    """The property that makes remounting a volume harmless."""
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low"})

    moved = served_models_from_library(
        [report("gpt-oss-20b-mxfp4-bf16", "/somewhere/else/entirely")],
        overrides=settings.overrides,
    )

    assert moved[0].default_reasoning_effort.value == "low"
    assert moved[0].path == "/somewhere/else/entirely"


def test_a_removed_model_keeps_its_settings_for_re_import(home) -> None:
    """Forgetting a model is an inventory action; it leaves the files alone.

    Discarding settings would make "forget and re-import" destructive for
    something the user never asked to change.
    """
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low"})
    save_model_settings(settings)

    # The library forgets the model; nothing touches its settings.
    reloaded = load_model_settings()

    assert reloaded.for_model("gpt-oss-20b") == {"reasoning_effort": "low"}
    # And they apply again the moment it comes back.
    restored = served_models_from_library(BOTH[:1], overrides=reloaded.overrides)
    assert restored[0].default_reasoning_effort.value == "low"


def test_settings_are_only_forgotten_when_asked(home) -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low"})

    settings.clear("gpt-oss-20b")

    assert settings.for_model("gpt-oss-20b") == {}


# -- resolution --------------------------------------------------------------


def test_each_model_resolves_its_own_values() -> None:
    """The witness: distinct configuration reaching the inference path."""
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low", "served_model_name": "fast-oss"})
    settings.set("gpt-oss-120b", {"reasoning_effort": "high", "served_model_name": "deep-oss"})

    resolved = {m.slug: m for m in served_models_from_library(BOTH, overrides=settings.overrides)}

    assert resolved["gpt-oss-20b"].default_reasoning_effort.value == "low"
    assert resolved["gpt-oss-20b"].display_name == "fast-oss"
    assert resolved["gpt-oss-120b"].default_reasoning_effort.value == "high"
    assert resolved["gpt-oss-120b"].display_name == "deep-oss"


def test_a_model_with_no_override_uses_the_server_default() -> None:
    from quantum_codex.canonical import ReasoningEffort

    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low"})

    resolved = {
        m.slug: m
        for m in served_models_from_library(
            BOTH, default_effort=ReasoningEffort.MEDIUM, overrides=settings.overrides
        )
    }

    assert resolved["gpt-oss-20b"].default_reasoning_effort.value == "low"
    assert resolved["gpt-oss-120b"].default_reasoning_effort.value == "medium"


def test_generation_overrides_reach_the_served_model() -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-120b", {"max_output_tokens": 4096, "temperature": 0.3, "top_p": 0.8})

    resolved = {m.slug: m for m in served_models_from_library(BOTH, overrides=settings.overrides)}

    assert resolved["gpt-oss-120b"].max_output_tokens == 4096
    assert resolved["gpt-oss-120b"].temperature == 0.3
    assert resolved["gpt-oss-20b"].max_output_tokens is None


def test_a_context_override_narrows_the_window() -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"context_length": 32768})

    resolved = {m.slug: m for m in served_models_from_library(BOTH, overrides=settings.overrides)}

    assert resolved["gpt-oss-20b"].context_window == 32768
    assert resolved["gpt-oss-120b"].context_window == 131072


# -- validation --------------------------------------------------------------


def test_validation_stays_on_the_server() -> None:
    problems = validate_model({"reasoning_effort": "xhigh"})

    assert [p.field for p in problems] == ["reasoning_effort"]


def test_every_problem_is_reported_at_once() -> None:
    problems = validate_model({"temperature": 9.0, "top_p": -1.0})

    assert {p.field for p in problems} == {"temperature", "top_p"}


# -- migration ---------------------------------------------------------------


def test_settings_move_from_a_profile_that_names_one_model(home) -> None:
    (home).mkdir(exist_ok=True)
    (home / "profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default": "dev",
                "profiles": {
                    "dev": {
                        "model": "gpt-oss-120b",
                        "reasoning_effort": "high",
                        "served_model_name": "deep",
                        "port": 8123,
                    }
                },
            }
        )
    )

    profiles = load_profiles()
    settings = ModelSettings()
    unattributed = migrate_from_profiles(profiles, settings)

    assert unattributed == []
    assert settings.for_model("gpt-oss-120b") == {
        "reasoning_effort": "high",
        "served_model_name": "deep",
    }
    # And the profile no longer owns them: one source of truth.
    assert profiles.get("dev").legacy_model_settings == {}
    assert not hasattr(profiles.get("dev"), "reasoning_effort")


def test_an_ambiguous_profile_keeps_its_values_rather_than_guessing(home) -> None:
    """A profile naming no model cannot say which model the settings were for.

    Assigning them anyway would attach one model's configuration to another.
    """
    (home / "profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default": "dev",
                "profiles": {"dev": {"model": "", "reasoning_effort": "high"}},
            }
        )
    )

    profiles = load_profiles()
    settings = ModelSettings()
    unattributed = migrate_from_profiles(profiles, settings)

    assert unattributed == ["dev"]
    assert settings.overrides == {}
    # Nothing discarded: still there, ready to move once a model is chosen.
    assert profiles.get("dev").legacy_model_settings == {"reasoning_effort": "high"}


def test_migration_never_overwrites_a_value_the_user_already_set(home) -> None:
    (home / "profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default": "dev",
                "profiles": {"dev": {"model": "gpt-oss-20b", "reasoning_effort": "high"}},
            }
        )
    )

    profiles = load_profiles()
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"reasoning_effort": "low"})
    migrate_from_profiles(profiles, settings)

    assert settings.for_model("gpt-oss-20b")["reasoning_effort"] == "low"


def test_a_profile_with_nothing_to_migrate_is_left_alone(home) -> None:
    profiles = Profiles()
    profiles.create("dev")
    save_profiles(profiles)
    settings = ModelSettings()

    assert migrate_from_profiles(load_profiles(), settings) == []
    assert settings.overrides == {}


def test_an_unknown_setting_in_a_profile_is_still_refused(home) -> None:
    """Only the settings that genuinely moved are treated as legacy."""
    (home / "profiles.json").write_text(
        json.dumps(
            {"version": 1, "default": "dev", "profiles": {"dev": {"invented": 1}}}
        )
    )

    with pytest.raises(ConfigError, match="unknown field"):
        load_profiles()


# -- catalogue defaults ------------------------------------------------------
#
# What the two supported models are configured as before anyone changes
# anything. A field the catalogue deliberately leaves alone inherits the server
# default, and that omission is itself declared.


def test_a_clean_store_resolves_the_presets_canonical_names() -> None:
    """The public id, not the directory the weights happen to occupy."""
    resolved = {m.slug: m for m in served_models_from_library(BOTH)}

    assert resolved["gpt-oss-20b"].display_name == "gpt-oss-20b"
    assert resolved["gpt-oss-120b"].display_name == "gpt-oss-120b"


def test_a_quantisation_suffix_never_reaches_the_public_name() -> None:
    resolved = {m.slug: m for m in served_models_from_library(BOTH)}

    for model in resolved.values():
        assert "mxfp4" not in model.display_name
        assert "bf16" not in model.display_name


def test_a_preset_installed_under_an_unusual_directory_keeps_its_name() -> None:
    """The discriminator: the default is the catalogue's, not the disk's."""
    odd = [report("gpt-oss-20b", "/somewhere/gpt-oss-20b")]

    resolved = served_models_from_library(odd)

    assert resolved[0].display_name == "gpt-oss-20b"


def test_a_custom_model_still_falls_back_to_its_directory_name() -> None:
    """No shipped opinion exists for a model the user imported themselves."""
    resolved = served_models_from_library([report("my-own-gpt-oss")])

    assert resolved[0].display_name == "my-own-gpt-oss"


def test_the_full_resolved_configuration_of_the_20b_from_a_clean_store() -> None:
    from quantum_codex.canonical import ReasoningEffort

    model = {m.slug: m for m in served_models_from_library(BOTH)}["gpt-oss-20b"]

    assert model.display_name == "gpt-oss-20b"
    assert model.default_reasoning_effort is ReasoningEffort.MEDIUM
    assert model.context_window == 131072
    # Declared as inherited: no catalogue opinion, so the server default applies.
    assert model.max_output_tokens is None
    assert model.temperature is None
    assert model.top_p is None


def test_the_full_resolved_configuration_of_the_120b_from_a_clean_store() -> None:
    from quantum_codex.canonical import ReasoningEffort

    model = {m.slug: m for m in served_models_from_library(BOTH)}["gpt-oss-120b"]

    assert model.display_name == "gpt-oss-120b"
    assert model.default_reasoning_effort is ReasoningEffort.MEDIUM
    assert model.context_window == 131072
    assert model.max_output_tokens is None


def test_every_inherited_field_is_declared_rather_than_merely_absent() -> None:
    """An omission has to be a decision someone can read."""
    from quantum_codex.library.catalog import DEFAULTS_INHERITED, SUPPORTED
    from quantum_codex.profile_schema import model_field_names

    for entry in SUPPORTED:
        covered = set(entry.defaults) | set(DEFAULTS_INHERITED)
        missing = model_field_names() - covered
        assert not missing, f"{entry.slug} leaves {missing} neither defaulted nor declared inherited"


def test_a_user_override_still_beats_the_catalogue_default() -> None:
    settings = ModelSettings()
    settings.set("gpt-oss-20b", {"served_model_name": "mine"})

    resolved = {m.slug: m for m in served_models_from_library(BOTH, overrides=settings.overrides)}

    assert resolved["gpt-oss-20b"].display_name == "mine"
    assert resolved["gpt-oss-120b"].display_name == "gpt-oss-120b"
