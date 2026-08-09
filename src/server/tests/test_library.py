"""The model library: states, volume handling, and what a scan may not do.

The load-bearing property here is negative: **a scan never removes an entry**.
Unplugging an external SSD makes tens of gigabytes unreadable for a while, and a
registry that pruned what it could not see would erase the record of weights
that are perfectly intact on a disk sitting on the desk.

So most of these tests are about what happens when things are *absent*.
"""

from __future__ import annotations

import json

import pytest

from quantum_codex.config import ConfigError
from quantum_codex.library import (
    ModelEntry,
    ModelRegistry,
    ModelState,
    evaluate,
    load_registry,
    save_registry,
    volume_for,
)
from quantum_codex.library.volumes import free_bytes_for

GPT_OSS_CONFIG = {
    "model_type": "gpt_oss",
    "architectures": ["GptOssForCausalLM"],
    "num_hidden_layers": 24,
    "num_local_experts": 32,
    "max_position_embeddings": 131072,
    "quantization": {"mode": "mxfp4", "bits": 4},
}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.fixture
def model_dir(tmp_path):
    def build(name="gpt-oss-20b", config=GPT_OSS_CONFIG, *, incomplete=False, tokenizer=True):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "model-00001.safetensors").write_bytes(b"w" * 2048)
        if tokenizer:
            (directory / "tokenizer.json").write_text("{}")
        if incomplete:
            (directory / "model-00002.safetensors.incomplete").write_bytes(b"x")
        return directory

    return build


# -- volumes -----------------------------------------------------------------


def test_a_path_on_the_startup_disk_is_always_available(tmp_path) -> None:
    volume = volume_for(tmp_path / "anything")

    assert volume.mounted is True
    assert volume.is_external is False


def test_an_unmounted_external_volume_is_recognised_without_existing() -> None:
    # The question is asked precisely when the path cannot be reached, so this
    # must work on a path that does not exist.
    volume = volume_for("/Volumes/DefinitelyNotAttached/models/gpt-oss-120b")

    assert volume.is_external is True
    assert volume.name == "DefinitelyNotAttached"
    assert volume.mounted is False


def test_free_space_is_reported_for_a_directory_that_does_not_exist_yet(tmp_path) -> None:
    # Asked before a download creates the target directory (cahier 24).
    free = free_bytes_for(tmp_path / "not" / "created" / "yet")

    assert free is not None and free > 0


# -- states ------------------------------------------------------------------


def test_a_valid_model_is_ready(model_dir) -> None:
    report = evaluate(ModelEntry(path=str(model_dir())))

    assert report.state is ModelState.READY
    assert report.state.usable is True
    assert report.quantization == "mxfp4-4bit"
    assert report.context_length == 131072


def test_an_absent_volume_is_not_a_missing_model() -> None:
    """The distinction that decides what the user is told to do."""
    report = evaluate(ModelEntry(path="/Volumes/DefinitelyNotAttached/gpt-oss-120b"))

    assert report.state is ModelState.MISSING_VOLUME
    assert report.state.recoverable_by_reattaching is True
    assert "reattach" in report.detail.lower()
    # And it must not be confused with deletion.
    assert report.state is not ModelState.MISSING


def test_a_deleted_directory_on_a_present_volume_is_missing(tmp_path) -> None:
    report = evaluate(ModelEntry(path=str(tmp_path / "was-here")))

    assert report.state is ModelState.MISSING
    assert report.state.recoverable_by_reattaching is False


def test_an_interrupted_download_is_distinguishable(model_dir) -> None:
    # It has a config.json, so without this check it would read as merely
    # invalid — hiding the fact that resuming would fix it.
    report = evaluate(ModelEntry(path=str(model_dir(incomplete=True))))

    assert report.state is ModelState.PARTIAL_DOWNLOAD
    assert "resum" in report.detail.lower()


def test_another_family_is_incompatible_not_invalid(model_dir) -> None:
    directory = model_dir(config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]})

    assert evaluate(ModelEntry(path=str(directory))).state is ModelState.INCOMPATIBLE


def test_a_broken_gpt_oss_directory_is_invalid(model_dir) -> None:
    # Our family, but unusable: worth re-fetching, unlike an incompatible one.
    directory = model_dir(tokenizer=False)

    assert evaluate(ModelEntry(path=str(directory))).state is ModelState.INVALID


# -- the registry does not forget --------------------------------------------


def test_a_scan_never_removes_an_entry(model_dir, tmp_path) -> None:
    """The invariant this module exists for."""
    registry = ModelRegistry()
    registry.add(model_dir("gpt-oss-20b"))
    registry.add_root(tmp_path / "empty-root")
    (tmp_path / "empty-root").mkdir()

    registry.discover()

    assert len(registry.report()) == 1


def test_a_root_on_an_absent_volume_yields_nothing_and_loses_nothing(model_dir) -> None:
    registry = ModelRegistry()
    registry.add(model_dir("gpt-oss-20b"))
    registry.add_root("/Volumes/DefinitelyNotAttached/models")

    registry.discover()

    assert len(registry.report()) == 1


def test_an_entry_on_an_absent_volume_survives_a_round_trip() -> None:
    # Saving and reloading while the drive is away must keep the record.
    registry = ModelRegistry(
        entries=[ModelEntry(path="/Volumes/DefinitelyNotAttached/gpt-oss-120b")]
    )
    save_registry(registry)

    reloaded = load_registry()
    reports = reloaded.report()

    assert len(reports) == 1
    assert reports[0].state is ModelState.MISSING_VOLUME


def test_forgetting_a_model_leaves_its_files_alone(model_dir) -> None:
    directory = model_dir()
    registry = ModelRegistry()
    registry.add(directory)

    registry.forget(directory)

    assert registry.report() == []
    assert directory.is_dir()
    assert (directory / "config.json").is_file()


# -- importing ---------------------------------------------------------------


def test_importing_validates_at_the_moment_of_choice(tmp_path) -> None:
    # Refused here, where the user still knows which directory they picked —
    # not at load time when the context is gone.
    empty = tmp_path / "not-a-model"
    empty.mkdir()

    with pytest.raises(ConfigError, match="cannot be used"):
        ModelRegistry().add(empty)


def test_importing_from_an_absent_volume_says_so() -> None:
    with pytest.raises(ConfigError, match="not mounted"):
        ModelRegistry().add("/Volumes/DefinitelyNotAttached/gpt-oss-120b")


def test_discovery_finds_models_under_a_root(model_dir, tmp_path) -> None:
    model_dir("gpt-oss-20b")
    model_dir("gpt-oss-120b")
    registry = ModelRegistry(roots=[str(tmp_path)])

    found = registry.discover()

    assert {entry.name for entry in found} == {"gpt-oss-20b", "gpt-oss-120b"}


def test_discovery_skips_directories_that_are_not_models(model_dir, tmp_path) -> None:
    model_dir("gpt-oss-20b")
    (tmp_path / "notes").mkdir()
    registry = ModelRegistry(roots=[str(tmp_path)])

    assert {entry.name for entry in registry.discover()} == {"gpt-oss-20b"}


def test_unknown_fields_in_the_stored_registry_are_refused(tmp_path, monkeypatch) -> None:
    from quantum_codex.config import write_json
    from quantum_codex.library.registry import MODELS_VERSION, models_path

    write_json(
        models_path(),
        {"version": MODELS_VERSION, "roots": [], "models": [{"path": "/x", "typo": 1}]},
    )

    with pytest.raises(ConfigError, match="unknown field"):
        load_registry()
