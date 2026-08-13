"""One product version, in every place that declares one.

The version is not a string in a file; it is a fact about a release, and six
files repeat it: two package manifests, two lockfiles, the Tauri configuration
the macOS bundle is built from, and the README badge. A bump that reaches five
of them produces a `.dmg` whose name and whose `CFBundleShortVersionString`
disagree, which is exactly the kind of thing nobody notices until a user reports
the wrong version.

`quantum_codex.__version__` is taken as the authority here -- not because it is
special, but because a comparison needs one, and it is the value `--version`
prints.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from quantum_codex import __version__

ROOT = Path(__file__).resolve().parents[3]


def read_toml(relative: str) -> dict:
    return tomllib.loads((ROOT / relative).read_text())


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_the_repository_root_is_where_this_test_thinks_it_is() -> None:
    """Everything below is a path assertion; a wrong root makes them vacuous."""
    assert (ROOT / "Makefile").is_file()
    assert (ROOT / "src" / "desktop" / "package.json").is_file()


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("src/server/pyproject.toml", lambda: read_toml("src/server/pyproject.toml")["project"]["version"]),
        ("src/desktop/package.json", lambda: read_json("src/desktop/package.json")["version"]),
        (
            "src/desktop/package-lock.json",
            lambda: read_json("src/desktop/package-lock.json")["version"],
        ),
        (
            "src/desktop/package-lock.json (root package)",
            lambda: read_json("src/desktop/package-lock.json")["packages"][""]["version"],
        ),
        (
            "src/desktop/src-tauri/Cargo.toml",
            lambda: read_toml("src/desktop/src-tauri/Cargo.toml")["package"]["version"],
        ),
        (
            "src/desktop/src-tauri/tauri.conf.json",
            lambda: read_json("src/desktop/src-tauri/tauri.conf.json")["version"],
        ),
    ],
)
def test_every_version_source_agrees_with_the_package(source: str, value) -> None:
    assert value() == __version__, f"{source} disagrees with quantum_codex.__version__"


def test_the_cargo_lock_carries_the_crate_version_it_will_build() -> None:
    """`cargo` rewrites this itself; a stale entry means a dirty tree in CI."""
    lock = tomllib.loads((ROOT / "src/desktop/src-tauri/Cargo.lock").read_text())
    crate = next(p for p in lock["package"] if p["name"] == "quantum-codex-desktop")

    assert crate["version"] == __version__


def test_the_readme_states_the_version_being_shipped() -> None:
    line = next(
        line
        for line in (ROOT / "README.md").read_text().splitlines()
        if line.startswith("> **Version")
    )

    assert f"**Version {__version__}**" in line


def test_the_bundle_identity_is_not_a_version_and_does_not_move_with_one() -> None:
    """What a version bump must leave alone.

    The bundle id and the app name are how macOS and every existing install
    recognise this application. Changing one during a patch release orphans
    preferences and produces a second app beside the first.
    """
    config = read_json("src/desktop/src-tauri/tauri.conf.json")

    assert config["identifier"] == "com.exalandru.qcs"
    assert config["productName"] == "Codex GPT-OSS Server"
    assert config["mainBinaryName"] == "Codex GPT-OSS Server"
