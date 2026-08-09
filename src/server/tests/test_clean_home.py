"""A deleted Application Support directory is a supported state.

Someone who removes it is doing a factory reset, not breaking the install.
Everything that reads configuration must cope with nothing being there, and
everything that writes must create what it needs. No step may require a human
to restore a JSON file by hand.
"""

from __future__ import annotations

import json

import pytest

from quantum_codex import config
from quantum_codex.library.registry import load_registry, save_registry
from quantum_codex.profile_schema import schema


@pytest.fixture
def clean_home(tmp_path, monkeypatch):
    """An Application Support directory that does not exist at all."""
    home = tmp_path / "gone"
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(home))
    assert not home.exists()
    return home


def test_reading_profiles_from_nothing_yields_an_empty_set(clean_home) -> None:
    profiles = config.load_profiles()

    assert profiles.profiles == {}
    assert profiles.default is None
    # Reading must not create anything: a factory-clean state stays clean until
    # the user actually saves something.
    assert not clean_home.exists()


def test_the_schema_is_available_before_any_file_exists(clean_home) -> None:
    """The configuration form is generated from this; it cannot depend on state."""
    described = schema()

    assert described["fields"]
    assert not clean_home.exists()


def test_creating_the_first_profile_creates_the_directory(clean_home) -> None:
    profiles = config.load_profiles()
    profiles.create("dev")
    config.save_profiles(profiles)

    assert config.profiles_path().is_file()
    assert config.load_profiles().get("dev").name == "dev"


def test_a_profile_created_from_nothing_needs_no_further_editing(clean_home) -> None:
    """The empty-state path must produce a usable profile, not a stub."""
    profiles = config.load_profiles()
    created = profiles.create("dev")
    config.save_profiles(profiles)

    created.validate()
    assert created.port == config.DEFAULT_PORT


def test_the_model_library_starts_empty_rather_than_failing(clean_home) -> None:
    registry = load_registry()

    assert registry.report() == []
    save_registry(registry)
    assert config.app_support_dir().is_dir()


def test_there_is_no_runtime_state_and_that_is_not_an_error(clean_home) -> None:
    assert config.load_runtime_state() is None


def test_nothing_claims_to_be_configured_after_a_wipe(clean_home) -> None:
    """No stale "configured" assumption may survive the directory being removed."""
    profiles = config.load_profiles()
    profiles.create("dev")
    config.save_profiles(profiles)
    assert config.load_profiles().profiles

    # The user deletes the directory.
    import shutil

    shutil.rmtree(clean_home)

    assert config.load_profiles().profiles == {}
    assert config.load_runtime_state() is None


def test_runtime_state_and_user_configuration_recover_independently(clean_home) -> None:
    """Losing one must not take the other with it."""
    profiles = config.load_profiles()
    profiles.create("dev")
    config.save_profiles(profiles)

    config.write_runtime_state(
        config.RuntimeState(
            pid=1, host="127.0.0.1", port=8123, model=None,
            management_token="t", started_at=1.0,
        )
    )
    config.runtime_path().unlink()

    assert config.load_runtime_state() is None
    assert config.load_profiles().get("dev").name == "dev"


def test_a_runtime_file_written_today_records_a_null_model(clean_home) -> None:
    """The shape the desktop has to read.

    A daemon holds no weights until a request names one, so `model` is null on
    every start. Declaring it non-nullable is what made a healthy server read as
    stopped.
    """
    config.write_runtime_state(
        config.RuntimeState(
            pid=1, host="127.0.0.1", port=8123, model=None,
            management_token="t", started_at=1.0,
        )
    )

    written = json.loads(config.runtime_path().read_text())
    assert written["model"] is None
    assert config.load_runtime_state().model is None
