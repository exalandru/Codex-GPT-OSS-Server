"""Moving to the named bundle identifier.

Renaming the product moved the application-data root. User configuration
follows; the managed Python runtime does not, because its scripts carry
absolute shebangs pointing at the old directory and a copy would look installed
while being unable to run.
"""

from __future__ import annotations

import json

import pytest

from quantum_codex import config


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A fake Application Support holding both identifiers."""
    monkeypatch.delenv("QUANTUM_CODEX_HOME", raising=False)
    monkeypatch.setattr(config, "_migration_checked", False)
    monkeypatch.setattr(config.Path, "home", staticmethod(lambda: tmp_path))
    support = tmp_path / "Library" / "Application Support"
    old = support / config.PREVIOUS_BUNDLE_ID
    new = support / config.BUNDLE_ID
    old.mkdir(parents=True)
    return old, new


def write(directory, name, payload) -> None:
    (directory / name).write_text(json.dumps(payload))


def test_the_identifier_is_the_named_one() -> None:
    assert config.BUNDLE_ID == "com.exalandru.qcs"
    assert config.PREVIOUS_BUNDLE_ID == "com.exalandru.quantum-codex"


def test_configuration_is_carried_into_the_new_root(roots) -> None:
    old, new = roots
    write(old, "profiles.json", {"version": 1, "default": "main", "profiles": {"main": {}}})
    write(old, "model-settings.json", {"version": 1, "models": {"gpt-oss-20b": {}}})

    moved = config.migrate_app_support_root()

    assert set(moved) == {"profiles.json", "model-settings.json"}
    assert (new / "profiles.json").is_file()
    assert config.load_profiles().get("main").name == "main"


def test_every_kind_of_user_state_comes_over(roots) -> None:
    old, _ = roots
    for name in config.MIGRATED_FILES:
        write(old, name, {"version": 1})

    moved = config.migrate_app_support_root()

    assert set(moved) == set(config.MIGRATED_FILES)


def test_the_managed_runtime_is_never_copied(roots) -> None:
    """Its shebangs point at the old path; a copy would not execute."""
    old, new = roots
    for directory in ("env", "python", "server", "uv-cache"):
        (old / directory).mkdir()
    (old / ".runtime-stamp.json").write_text("{}")

    config.migrate_app_support_root()

    for directory in ("env", "python", "server", "uv-cache"):
        assert not (new / directory).exists(), f"{directory} must be rebuilt, not carried"
    assert not (new / ".runtime-stamp.json").exists()


def test_process_metadata_is_not_carried(roots) -> None:
    """A runtime file describes a process that is not running; adopting it
    would point the app at a dead endpoint."""
    old, new = roots
    write(old, "runtime.json", {"version": 1, "pid": 1, "port": 8123})

    config.migrate_app_support_root()

    assert not (new / "runtime.json").exists()


def test_a_root_that_has_been_used_is_left_alone(roots) -> None:
    """Never undo work done after the rename."""
    old, new = roots
    new.mkdir(parents=True)
    write(old, "profiles.json", {"version": 1, "default": "old", "profiles": {"old": {}}})
    write(new, "profiles.json", {"version": 1, "default": "new", "profiles": {"new": {}}})

    assert config.migrate_app_support_root() == []
    assert config.load_profiles().default == "new"


def test_a_clean_install_with_no_previous_directory_does_nothing(roots) -> None:
    old, new = roots
    import shutil

    shutil.rmtree(old)

    assert config.migrate_app_support_root() == []
    assert config.load_profiles().profiles == {}


def test_the_old_directory_is_left_intact(roots) -> None:
    """Copied, not moved: an older build must still find its own state."""
    old, _ = roots
    write(old, "profiles.json", {"version": 1, "default": "main", "profiles": {"main": {}}})

    config.migrate_app_support_root()

    assert (old / "profiles.json").is_file()


def test_it_runs_once_per_process(roots) -> None:
    old, _ = roots
    write(old, "profiles.json", {"version": 1, "default": "main", "profiles": {"main": {}}})

    assert config.migrate_app_support_root() != []
    assert config.migrate_app_support_root() == []


def test_an_explicit_home_is_never_migrated_into(tmp_path, monkeypatch) -> None:
    """A caller who set QUANTUM_CODEX_HOME asked for that directory, not for
    whatever another installation left behind."""
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(config, "_migration_checked", False)

    assert config.migrate_app_support_root() == []


def test_model_files_are_not_touched(roots) -> None:
    """They live where the user put them; the library records them by path."""
    old, _ = roots
    write(old, "models.json", {"version": 1, "roots": ["/Volumes/Models/mlx"], "models": []})

    config.migrate_app_support_root()

    from quantum_codex.library.registry import load_registry

    assert "/Volumes/Models/mlx" in load_registry().roots
