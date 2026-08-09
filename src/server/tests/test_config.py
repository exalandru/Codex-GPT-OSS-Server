"""Configuration files: atomicity, versioning and profile rules.

Configuration is state a user edits by hand and expects to survive. The
assertions here are mostly about what happens when something goes wrong —
an interrupted write, a hand-edited typo, a file from a newer build — because
those are the paths that silently lose what somebody configured.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from quantum_codex import config
from quantum_codex.config import (
    ConfigError,
    Profiles,
    RuntimeState,
    ServerProfile,
    load_profiles,
    load_runtime_state,
    profiles_path,
    runtime_path,
    save_profiles,
    write_json,
    write_runtime_state,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real application-support directory."""
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    return tmp_path


def profile(name: str = "20b", **kwargs) -> ServerProfile:
    kwargs.setdefault("model", "/models/gpt-oss-20b")
    return ServerProfile(name=name, **kwargs)


# -- atomic writes ------------------------------------------------------------


def test_a_failed_write_leaves_the_previous_file_intact(isolated_home) -> None:
    """The point of writing through a temporary file.

    A half-written profiles file would be read at the next start, so the failure
    mode is not "lost edit" but "unusable configuration".
    """
    target = isolated_home / "settings.json"
    write_json(target, {"version": 1, "value": "original"})

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        write_json(target, {"version": 1, "value": Unserialisable()})

    assert json.loads(target.read_text())["value"] == "original"
    # And no debris left behind for the next reader to trip over.
    assert [p.name for p in isolated_home.iterdir()] == ["settings.json"]


def test_secret_files_are_owner_only(isolated_home) -> None:
    # The runtime file carries the management token, which is what stands
    # between another local process and this server's control surface.
    write_json(isolated_home / "runtime.json", {"version": 1}, private=True)
    mode = stat.S_IMODE(os.stat(isolated_home / "runtime.json").st_mode)

    assert mode == 0o600


def test_ordinary_files_are_readable(isolated_home) -> None:
    write_json(isolated_home / "settings.json", {"version": 1})
    mode = stat.S_IMODE(os.stat(isolated_home / "settings.json").st_mode)

    assert mode == 0o644


# -- schema versioning --------------------------------------------------------


def test_a_file_from_a_newer_build_is_refused_not_downgraded() -> None:
    """Reading it with older rules would silently drop what it added."""
    write_json(profiles_path(), {"version": 99, "profiles": {}})

    with pytest.raises(ConfigError, match="newer version"):
        load_profiles()


def test_a_missing_file_is_not_an_error() -> None:
    assert load_profiles().profiles == {}
    assert load_runtime_state() is None


def test_malformed_json_names_the_file() -> None:
    profiles_path().parent.mkdir(parents=True, exist_ok=True)
    profiles_path().write_text("{not json")

    with pytest.raises(ConfigError, match="not valid JSON"):
        load_profiles()


# -- profiles -----------------------------------------------------------------


def test_profiles_round_trip() -> None:
    profiles = Profiles()
    profiles.put(profile("20b", port=8123))
    profiles.put(profile("120b", model="/models/gpt-oss-120b", port=8124))
    save_profiles(profiles)

    loaded = load_profiles()

    assert set(loaded.profiles) == {"20b", "120b"}
    assert loaded.get("120b").port == 8124
    # The first profile added becomes the default, so a single-profile setup
    # needs no further ceremony.
    assert loaded.default == "20b"


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    """A typo in a hand-edited file must not silently do nothing."""
    write_json(
        profiles_path(),
        {
            "version": config.PROFILES_VERSION,
            "default": "20b",
            "profiles": {"20b": {"model": "/m", "reasoning_efort": "high"}},
        },
    )

    with pytest.raises(ConfigError, match="unknown field"):
        load_profiles()


def test_a_dangling_default_is_refused() -> None:
    write_json(
        profiles_path(),
        {"version": config.PROFILES_VERSION, "default": "gone", "profiles": {}},
    )

    with pytest.raises(ConfigError, match="does not exist"):
        load_profiles()


def test_removing_the_default_picks_another() -> None:
    profiles = Profiles()
    profiles.put(profile("a"))
    profiles.put(profile("b"))
    assert profiles.default == "a"

    profiles.remove("a")

    assert profiles.default == "b"


def test_resolving_without_a_default_says_what_to_do() -> None:
    profiles = Profiles()
    profiles.put(profile("a"))
    profiles.put(profile("b"))
    profiles.default = None

    with pytest.raises(ConfigError, match="--profile"):
        profiles.resolve(None)


def test_a_lone_profile_needs_no_default() -> None:
    profiles = Profiles(profiles={"only": profile("only")}, default=None)

    assert profiles.resolve(None).name == "only"


def test_the_profile_names_a_model_by_id_not_by_path() -> None:
    # Clients should see a stable id, never a filesystem path. The *alias* a
    # model is served as is the model's own setting now, so all a profile
    # carries is which model it preloads.
    assert profile(model="gpt-oss-20b").default_model == "gpt-oss-20b"
    assert profile(model="").default_model is None


# -- runtime state ------------------------------------------------------------


def test_runtime_state_round_trips() -> None:
    state = RuntimeState(
        pid=os.getpid(),
        host="127.0.0.1",
        port=8123,
        model="gpt-oss-20b",
        management_token="secret",
        started_at=1.0,
    )
    write_runtime_state(state)

    loaded = load_runtime_state()

    assert loaded == state
    assert loaded.base_url == "http://127.0.0.1:8123"
    assert loaded.is_running is True


def test_a_stale_runtime_file_reports_a_dead_process() -> None:
    """A crash leaves this file behind; readers must be able to tell."""
    write_runtime_state(
        RuntimeState(
            pid=2**22,  # above the pid ceiling, so it cannot exist
            host="127.0.0.1",
            port=8123,
            model="gpt-oss-20b",
            management_token="secret",
            started_at=1.0,
        )
    )

    assert load_runtime_state().is_running is False


def test_an_incomplete_runtime_file_is_refused() -> None:
    write_json(runtime_path(), {"version": config.RUNTIME_VERSION, "pid": 1}, private=True)

    with pytest.raises(ConfigError, match="incomplete"):
        load_runtime_state()


# -- the model a profile preloads --------------------------------------------
#
# `None` is a real value: it is what "None — load on demand" saves. A crash
# here made the whole configuration surface unreachable and the profile look
# corrupted, so every shape a file can legitimately hold is pinned.


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, None),          # "None — load on demand", as the form saves it
        ("", None),            # written by an earlier build
        ("   ", None),         # hand-edited
        ("gpt-oss-20b", "gpt-oss-20b"),
        ("  gpt-oss-120b  ", "gpt-oss-120b"),
    ],
)
def test_every_shape_of_default_model_resolves(stored, expected) -> None:
    assert profile(model=stored).default_model == expected


def test_a_profile_whose_model_is_null_loads(tmp_path, monkeypatch) -> None:
    """The exact regression: `TypeError: argument of type 'NoneType'`."""
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        json.dumps({"version": 1, "default": "main", "profiles": {"main": {"model": None}}})
    )

    loaded = load_profiles()

    assert loaded.get("main").default_model is None


def test_a_legacy_profile_with_no_model_key_loads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        json.dumps({"version": 1, "default": "main", "profiles": {"main": {"port": 8123}}})
    )

    assert load_profiles().get("main").default_model is None


def test_a_path_valued_profile_still_migrates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default": "main",
                "profiles": {"main": {"model": "/Volumes/A/mlx/gpt-oss-120b-mxfp4-bf16"}},
            }
        )
    )

    loaded = load_profiles()

    assert loaded.get("main").default_model == "gpt-oss-120b"
    assert loaded.migrated is True


def test_null_is_never_replaced_by_some_arbitrary_model(tmp_path, monkeypatch) -> None:
    """"Load on demand" is a choice, not a gap to be filled in."""
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        json.dumps({"version": 1, "default": "main", "profiles": {"main": {"model": None}}})
    )

    loaded = load_profiles()

    assert loaded.get("main").model is None
    assert loaded.migrated is False


def test_a_non_string_model_does_not_crash_the_loader(tmp_path, monkeypatch) -> None:
    """A hand-edited file must not make `profiles list` unusable."""
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        json.dumps({"version": 1, "default": "main", "profiles": {"main": {"model": 42}}})
    )

    # It loads, and the unusable value simply reads as "no default model".
    assert load_profiles().get("main").default_model is None


def test_one_odd_profile_does_not_hide_the_others(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default": "good",
                "profiles": {
                    "good": {"model": "gpt-oss-20b"},
                    "odd": {"model": None},
                },
            }
        )
    )

    loaded = load_profiles()

    assert set(loaded.profiles) == {"good", "odd"}
    assert loaded.get("good").default_model == "gpt-oss-20b"
