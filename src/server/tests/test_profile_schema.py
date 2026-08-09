"""The profile schema, and its agreement with what the server actually honours.

The first test is the one that matters. A configuration form is generated from
this schema, so a field listed here that the server ignores would let a user
believe they had configured something. Checking the schema against the profile
dataclass — and against what `serve` actually consumes — is what keeps the form
truthful.
"""

from __future__ import annotations

import pytest

from quantum_codex.config import ServerProfile
from quantum_codex.profile_schema import (
    FIELDS,
    GROUPS,
    MODEL_FIELDS,
    coerce,
    coerce_model,
    field_names,
    schema,
    validate,
    validate_model,
)


def test_every_declared_setting_exists_on_the_profile() -> None:
    """A form field with nowhere to store its value would be a lie."""
    stored = set(ServerProfile.__dataclass_fields__) - {"name"}

    assert field_names() <= stored, f"declared but not stored: {field_names() - stored}"


def test_every_stored_setting_is_declared() -> None:
    """The converse: a setting nobody can reach from the interface.

    `legacy_model_settings` is excluded on purpose: it is a holding pen for
    model settings found in an old profile, never a preference, and rendering it
    would offer to edit something that is about to move elsewhere.
    """
    stored = set(ServerProfile.__dataclass_fields__) - {"name", "legacy_model_settings"}

    assert stored <= field_names(), f"stored but not declared: {stored - field_names()}"


def test_no_setting_belongs_to_both_the_profile_and_a_model() -> None:
    """One owner each. Two would mean two live sources of truth."""
    server = {field.name for field in FIELDS}
    per_model = {field.name for field in MODEL_FIELDS}

    assert server & per_model == set()


def test_the_profile_no_longer_stores_model_specific_settings() -> None:
    """They moved to `model_settings`, keyed by model id."""
    stored = set(ServerProfile.__dataclass_fields__)

    for moved in ("served_model_name", "reasoning_effort", "context_length", "temperature"):
        assert moved not in stored, f"{moved} is still owned by the profile"


def test_the_defaults_match_the_dataclass() -> None:
    """A form that opened on different defaults than the server uses would
    silently rewrite settings the moment it was saved."""
    profile = ServerProfile(name="x", model="/m")

    for field in FIELDS:
        if field.default is None or field.name == "model":
            continue
        assert getattr(profile, field.name) == field.default, field.name


def test_settings_that_need_a_restart_are_marked() -> None:
    # These decide what is loaded or where it listens, so changing them while
    # running cannot take effect. Saying so is the difference between a setting
    # that appears broken and one that is merely deferred.
    restart = {field.name for field in FIELDS if field.restart_required}

    assert {"model", "port", "host"} <= restart
    # Sampling defaults are read per request, so they need no restart.
    assert "temperature" not in {field.name for field in MODEL_FIELDS if field.restart_required}


def test_risky_settings_carry_a_caution() -> None:
    caution = {field.name for field in FIELDS if field.caution}

    # Binding beyond loopback exposes a server with no authentication.
    assert "host" in caution


def test_every_field_belongs_to_a_declared_group() -> None:
    groups = {group["id"] for group in GROUPS}

    for field in FIELDS:
        assert field.group in groups, field.name


def test_the_schema_serialises_without_nulls() -> None:
    # Absent means "not applicable"; a null would make a form render an empty
    # constraint as though one existed.
    payload = schema()

    for field in payload["fields"]:
        assert None not in field.values()


# -- coercion ----------------------------------------------------------------


def test_text_input_becomes_the_stored_type() -> None:
    # Every form value arrives as a string; the server decides what its own
    # settings mean.
    assert coerce("port", "8124") == 8124
    # Generation settings belong to a model now, so they are coerced against the
    # model schema. Same rules, different owner.
    assert coerce_model("temperature", "0.7") == 0.7
    assert coerce_model("reasoning_effort", " high ") == "high"


def test_an_empty_nullable_field_means_inherit() -> None:
    assert coerce_model("temperature", "") is None
    assert coerce_model("served_model_name", "") is None


def test_an_empty_required_field_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        coerce("port", "")


def test_an_unknown_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        coerce_model("resoning_effort", "high")


# -- validation --------------------------------------------------------------


def test_bounds_are_enforced() -> None:
    problems = validate({"port": 99999}) + validate_model({"temperature": 5.0})

    assert {problem.field for problem in problems} == {"port", "temperature"}


def test_every_problem_is_reported_not_just_the_first() -> None:
    """So a form can mark all the offending fields at once."""
    problems = validate({"port": 0}) + validate_model({"temperature": -1.0, "top_p": 9.0})

    assert len(problems) == 3


def test_a_choice_outside_the_list_is_refused() -> None:
    problems = validate_model({"reasoning_effort": "xhigh"})

    assert len(problems) == 1
    assert "low, medium, high" in problems[0].message


def test_valid_values_produce_no_problems() -> None:
    assert validate({"port": 8123}) == []
    assert validate_model({"temperature": 0.7, "reasoning_effort": "high"}) == []


def test_a_null_in_a_nullable_field_is_fine() -> None:
    assert validate_model({"temperature": None}) == []


def test_a_null_in_a_required_field_is_not() -> None:
    problems = validate({"port": None})

    assert len(problems) == 1
    assert "required" in problems[0].message
