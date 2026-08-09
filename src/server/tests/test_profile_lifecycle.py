"""Creating, copying, renaming and deleting profiles.

The lifecycle lives on the server so the CLI and the desktop cannot disagree
about what a valid name is or what a new profile contains. These tests pin the
rules that a second implementation would be most likely to get subtly wrong.
"""

from __future__ import annotations

import json

import pytest

from quantum_codex.config import (
    ConfigError,
    Profiles,
    ServerProfile,
    load_profiles,
    validate_profile_name,
)
from quantum_codex.profile_schema import schema


def populated() -> Profiles:
    profiles = Profiles()
    profiles.create("work")
    return profiles


# -- names -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a", "work", "gpt oss 120b", "my-profile", "v1.2_final"])
def test_ordinary_names_are_accepted(name: str) -> None:
    assert validate_profile_name(name) == name


@pytest.mark.parametrize(
    "name", ["", "   ", "/etc/passwd", "../escape", "has/slash", "-leading", "x" * 65]
)
def test_names_that_would_not_survive_a_shell_or_a_path_are_refused(name: str) -> None:
    with pytest.raises(ConfigError):
        validate_profile_name(name)


def test_a_name_is_trimmed_rather_than_rejected_for_stray_spaces() -> None:
    assert validate_profile_name("  work  ") == "work"


# -- create ------------------------------------------------------------------


def test_a_new_profile_uses_the_dataclass_defaults() -> None:
    """The same defaults the schema publishes; nothing is invented per caller."""
    created = Profiles().create("fresh")
    reference = ServerProfile(name="fresh")

    assert created == reference


def test_a_new_profile_needs_no_model() -> None:
    """The daemon serves every installed model and loads on demand."""
    created = Profiles().create("fresh")

    # `None` is the value for "load on demand"; the form saves it explicitly.
    assert created.default_model is None
    created.validate()


def test_the_first_profile_becomes_the_default() -> None:
    profiles = Profiles()
    profiles.create("first")
    profiles.create("second")

    assert profiles.default == "first"


def test_creating_over_an_existing_name_is_refused() -> None:
    profiles = populated()

    with pytest.raises(ConfigError, match="already exists"):
        profiles.create("work")


# -- duplicate ---------------------------------------------------------------


def test_duplicate_copies_every_field_except_the_name() -> None:
    profiles = populated()
    profiles.get("work").port = 9001
    profiles.get("work").log_level = "DEBUG"

    copy = profiles.duplicate("work", "work copy")

    assert copy.name == "work copy"
    assert copy.port == 9001
    assert copy.log_level == "DEBUG"


def test_duplicate_does_not_alias_the_original() -> None:
    """A copy that shared state would make editing one edit both."""
    profiles = populated()
    copy = profiles.duplicate("work", "copy")

    copy.port = 9111

    assert profiles.get("work").port != 9111


def test_duplicating_onto_an_existing_name_is_refused() -> None:
    profiles = populated()
    profiles.create("other")

    with pytest.raises(ConfigError, match="already exists"):
        profiles.duplicate("work", "other")


def test_duplicating_an_unknown_profile_is_refused() -> None:
    with pytest.raises(ConfigError, match="no profile named"):
        populated().duplicate("nope", "copy")


# -- rename ------------------------------------------------------------------


def test_rename_moves_the_profile_and_its_key() -> None:
    profiles = populated()
    profiles.rename("work", "renamed")

    assert "work" not in profiles.profiles
    assert profiles.get("renamed").name == "renamed"


def test_rename_follows_the_default_selection() -> None:
    """Otherwise renaming the default profile silently unsets the default."""
    profiles = populated()
    assert profiles.default == "work"

    profiles.rename("work", "renamed")

    assert profiles.default == "renamed"


def test_rename_preserves_position() -> None:
    profiles = Profiles()
    for name in ("one", "two", "three"):
        profiles.create(name)

    profiles.rename("two", "middle")

    assert list(profiles.profiles) == ["one", "middle", "three"]


def test_renaming_to_the_same_name_is_a_no_op() -> None:
    profiles = populated()

    assert profiles.rename("work", "work").name == "work"


def test_renaming_onto_an_existing_name_is_refused() -> None:
    profiles = populated()
    profiles.create("other")

    with pytest.raises(ConfigError, match="already exists"):
        profiles.rename("work", "other")


# -- remove ------------------------------------------------------------------


def test_removing_the_default_selects_another() -> None:
    profiles = populated()
    profiles.create("second")

    profiles.remove("work")

    assert profiles.default == "second"


def test_removing_the_last_profile_leaves_no_default() -> None:
    profiles = populated()
    profiles.remove("work")

    assert profiles.default is None
    assert profiles.profiles == {}


def test_removing_an_unknown_profile_is_refused() -> None:
    with pytest.raises(ConfigError, match="no profile named"):
        populated().remove("nope")


# -- model identity ----------------------------------------------------------
#
# A profile records *which model*, not *where the weights are*. A path changes
# when a volume is remounted or a directory renamed; the id does not.


def test_a_profile_written_with_a_model_path_is_migrated_to_a_model_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default": "dev",
                "profiles": {
                    "dev": {"model": "/Volumes/Models/mlx/gpt-oss-120b-mxfp4-bf16"}
                },
            }
        )
    )

    loaded = load_profiles()

    assert loaded.get("dev").model == "gpt-oss-120b"
    assert loaded.migrated is True


def test_a_profile_already_holding_a_model_id_is_left_alone(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        json.dumps(
            {"version": 1, "default": "dev", "profiles": {"dev": {"model": "gpt-oss-20b"}}}
        )
    )

    loaded = load_profiles()

    assert loaded.get("dev").model == "gpt-oss-20b"
    assert loaded.migrated is False


def test_the_model_field_offers_installed_models_and_an_explicit_none() -> None:
    """The form is a choice over the library, never a filesystem picker."""
    described = schema(["gpt-oss-20b", "gpt-oss-120b"])
    model = next(f for f in described["fields"] if f["name"] == "model")

    assert model["kind"] == "choice"
    assert model["choices"] == ["", "gpt-oss-20b", "gpt-oss-120b"]
    assert model["choice_labels"][""].startswith("None")


def test_the_model_choices_follow_the_library_rather_than_a_fixed_list() -> None:
    described = schema(["something-imported"])
    model = next(f for f in described["fields"] if f["name"] == "model")

    assert "something-imported" in model["choices"]
