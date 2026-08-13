"""One model, three names, one authority for each.

An imported model is a first-class configurable model here, which means three
values that are deliberately not the same thing:

``id``            the immutable library identity. Profiles, settings and the
                  desktop's selectors store this, and nothing the user can edit
                  changes it.
``display_name``  what QCS calls the model. Mutable, and load-bearing for
                  nobody.
``served_name``   what Codex asks for. Mutable, validated, unique among the
                  models that are actually served, and the *only* name that
                  reaches the wire.

These tests cross the real boundaries rather than checking each layer's own
opinion: the CLI writes the settings, the daemon builds its catalogue from the
same files, and `/v1/models`, `/v1/responses` and the generated Codex
configuration are read from a running app. A layer that started deriving a name
of its own would pass its own unit tests and fail here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from quantum_codex import app as app_module
from quantum_codex.canonical import FinishReason, GenerationTiming
from quantum_codex.cli import main
from quantum_codex.inference.engine import EngineState, GenerationOutcome
from quantum_codex.library.registry import load_registry, models_path
from quantum_codex.model_settings import load_model_settings

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
    def build(name: str, *, under: str = "weights"):
        directory = tmp_path / under / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(json.dumps(GPT_OSS_CONFIG))
        (directory / "model-00001.safetensors").write_bytes(b"w" * 2048)
        (directory / "tokenizer.json").write_text("{}")
        return directory

    return build


def run(*argv: str) -> int:
    return main(list(argv))


def imported_id(capsys, path) -> str:
    assert run("models", "import", "--json", str(path)) == 0
    return json.loads(capsys.readouterr().out)["id"]


# -- the fake engine ---------------------------------------------------------


@dataclass
class Loaded:
    served_name: str
    quantization: str = "mxfp4-4bit"
    context_length: int = 131072
    adapter: object | None = None


class RecordingEngine:
    """Loads instantly and records the *path* it was given.

    The path is the discriminating observation: every name in this file can be
    changed, and the question each test is really asking is which set of weights
    a request ended up addressing.
    """

    def __init__(self) -> None:
        self.state = EngineState.UNLOADED
        self.loaded_paths: list[str] = []
        self.load_elapsed_seconds = None
        self.completion: list[int] = []

    async def load(self, path, served_name, context_length, *, adapter_path=None):  # noqa: ANN001
        self.loaded_paths.append(str(path))
        self.state = EngineState.READY
        return Loaded(served_name=served_name, context_length=context_length)

    async def unload(self) -> None:
        self.state = EngineState.UNLOADED

    def shutdown(self) -> None:
        return None

    async def generate(self, prompt_tokens, **kwargs):  # noqa: ANN001, ARG002
        return GenerationOutcome(
            tokens=list(self.completion),
            input_tokens=len(prompt_tokens),
            finish_reason=FinishReason.STOP,
            timing=GenerationTiming(prefill_seconds=0.01, decode_seconds=0.01),
        )


@pytest.fixture
def engine(monkeypatch) -> RecordingEngine:
    """Every app built in this module gets this engine instead of MLX."""
    recorder = RecordingEngine()
    monkeypatch.setattr(app_module, "MlxEngine", lambda **_: recorder)
    return recorder


def completion_tokens(text: str) -> list[int]:
    from quantum_codex.harmony.render import load_encoding

    return load_encoding().encode(
        f"<|channel|>final<|message|>{text}<|return|>", allowed_special="all"
    )


@pytest.fixture
def client(engine):
    engine.completion = completion_tokens("done")
    app = app_module.create_app(host="127.0.0.1", port=8123)
    with TestClient(app) as running:
        yield running


# -- identity ----------------------------------------------------------------


def test_an_imported_model_is_configured_by_id_and_served_by_name(
    capsys, model_dir, client, engine
) -> None:
    """Import, rename both names, reload, and follow the id to the weights."""
    directory = model_dir("my-own-gpt-oss")
    model_id = imported_id(capsys, directory)

    assert run(
        "models",
        "config",
        "--json",
        model_id,
        "display_name=My Local Model",
        "served_model_name=codex-local",
    ) == 0
    capsys.readouterr()

    # Persisted, and keyed by the id rather than by either name.
    assert load_model_settings().overrides[model_id] == {
        "display_name": "My Local Model",
        "served_model_name": "codex-local",
    }
    # The library entry is untouched by a rename: identity is not metadata.
    entries = load_registry().report()
    assert [entry.entry.id for entry in entries] == [model_id]
    assert entries[0].entry.path == str(directory)

    # `/v1/models` publishes the served name and the display name, and no id.
    app_module.refresh_registry(client.app.state.context)
    published = client.get("/v1/models").json()
    entry = next(m for m in published["models"] if m["slug"] == "codex-local")
    assert entry["display_name"] == "My Local Model"
    assert [m["slug"] for m in published["models"]] == ["codex-local"]

    # `/v1/responses` accepts that same name and reaches the imported weights.
    answered = client.post(
        "/v1/responses",
        json={"model": "codex-local", "input": "hello", "stream": False},
    )
    assert answered.status_code == 200, answered.text
    assert answered.json()["model"] == "codex-local"
    assert engine.loaded_paths == [str(directory)]


def test_the_wire_never_accepts_the_library_id(capsys, model_dir, client) -> None:
    """The counterfactual: one Codex-facing name, not two.

    If `/v1/responses` also answered to the stable id, every consumer below
    would appear to agree while actually publishing one name and routing on
    another.
    """
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    assert run("models", "config", "--json", model_id, "served_model_name=codex-local") == 0
    capsys.readouterr()
    app_module.refresh_registry(client.app.state.context)

    refused = client.post(
        "/v1/responses", json={"model": model_id, "input": "hello", "stream": False}
    )

    assert refused.status_code == 400
    assert "codex-local" in refused.json()["error"]["message"]


def test_launch_configuration_and_profile_agree_on_one_model(
    capsys, model_dir, engine
) -> None:
    """Profile default is the id; everything Codex reads is the served name."""
    directory = model_dir("my-own-gpt-oss")
    model_id = imported_id(capsys, directory)
    assert run(
        "models",
        "config",
        "--json",
        model_id,
        "display_name=My Local Model",
        "served_model_name=codex-local",
    ) == 0
    assert run("profiles", "new", "default") == 0
    assert run("profiles", "set", "default", f"model={model_id}") == 0
    capsys.readouterr()

    assert run("codex", "launch", "--models-json") == 0
    offered = json.loads(capsys.readouterr().out)
    assert offered["default"] == model_id
    assert offered["models"] == [
        {
            "id": model_id,
            "slug": "codex-local",
            "display_name": "My Local Model",
            "reasoning_effort": "medium",
        }
    ]

    assert run("codex", "launch", "--config") == 0
    config = capsys.readouterr().out
    assert 'model = "codex-local"' in config
    assert model_id not in config

    assert run("codex", "launch") == 0
    command = capsys.readouterr().out
    assert '-c model="codex-local"' in command
    assert model_id not in command


def test_a_renamed_model_is_still_the_profile_default_at_preload(
    capsys, model_dir, engine
) -> None:
    """The regression: a profile stores an id, and preload must resolve it.

    Fails if anything on this path interprets the stored id as a served name --
    the server then starts, reports nothing worse than a warning, and simply
    does not have the model the user asked it to hold.
    """
    directory = model_dir("my-own-gpt-oss")
    model_id = imported_id(capsys, directory)
    assert run("models", "config", "--json", model_id, "served_model_name=codex-local") == 0
    assert run("profiles", "new", "default") == 0
    assert run("profiles", "set", "default", f"model={model_id}") == 0
    capsys.readouterr()

    from quantum_codex.config import load_profiles

    stored = load_profiles().resolve(None).default_model
    assert stored == model_id

    app = app_module.create_app(host="127.0.0.1", port=8123, preload=stored)
    with TestClient(app):
        pass

    assert engine.loaded_paths == [str(directory)]


def test_preload_still_accepts_a_path_and_a_served_name(capsys, model_dir, engine) -> None:
    """The other two selectors QCS itself uses, against the same weights."""
    directory = model_dir("my-own-gpt-oss")
    model_id = imported_id(capsys, directory)
    assert run("models", "config", "--json", model_id, "served_model_name=codex-local") == 0
    capsys.readouterr()

    for selector in (str(directory), "codex-local", model_id):
        engine.loaded_paths.clear()
        app = app_module.create_app(host="127.0.0.1", port=8123, preload=selector)
        with TestClient(app):
            pass
        assert engine.loaded_paths == [str(directory)], selector


def test_the_profile_form_names_models_without_storing_the_name(
    capsys, model_dir, engine
) -> None:
    """A stable id is what is stored; a human still has to read the list."""
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    assert run(
        "models",
        "config",
        "--json",
        model_id,
        "display_name=My Local Model",
        "served_model_name=codex-local",
    ) == 0
    capsys.readouterr()

    assert run("profiles", "schema") == 0
    field = next(
        item
        for item in json.loads(capsys.readouterr().out)["fields"]
        if item["name"] == "model"
    )

    assert field["choices"] == ["", model_id]
    assert field["choice_labels"][model_id] == "My Local Model — served as codex-local"


# -- names that may not be stored --------------------------------------------


def test_a_duplicate_served_name_is_refused_and_nothing_is_written(
    capsys, model_dir
) -> None:
    first = imported_id(capsys, model_dir("first-model"))
    second = imported_id(capsys, model_dir("second-model"))
    assert run("models", "config", "--json", first, "served_model_name=shared") == 0
    capsys.readouterr()
    before = json.loads(models_path().parent.joinpath("model-settings.json").read_text())

    assert run("models", "config", "--json", second, "served_model_name=shared") != 0

    assert "shared" in capsys.readouterr().err
    after = json.loads(models_path().parent.joinpath("model-settings.json").read_text())
    assert after == before
    assert second not in load_model_settings().overrides


def test_a_served_name_codex_could_not_ask_for_is_refused(capsys, model_dir) -> None:
    """It is interpolated into a generated `config.toml`, verbatim."""
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))

    assert run("models", "config", "--json", model_id, 'served_model_name=a" b') != 0

    assert "Served as" in capsys.readouterr().err
    assert load_model_settings().overrides == {}


def test_changing_the_display_name_changes_nothing_else(capsys, model_dir) -> None:
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    assert run("models", "config", "--json", model_id, "served_model_name=codex-local") == 0
    capsys.readouterr()

    assert run("models", "config", "--json", model_id, "display_name=Renamed") == 0
    capsys.readouterr()

    settings = load_model_settings().overrides[model_id]
    assert settings == {"served_model_name": "codex-local", "display_name": "Renamed"}
    assert [entry.entry.id for entry in load_registry().report()] == [model_id]


# -- identity that survives an unrelated change ------------------------------


def test_an_id_survives_forgetting_another_model(capsys, model_dir) -> None:
    """Identity may not depend on the membership of the file it was read from.

    Two directories with the same name get one derived id and one disambiguated
    id, and removing the first must not re-allocate the second's -- which would
    orphan the settings stored under it.

    This holds with or without the write-back in `load_registry`: forgetting a
    model saves the registry, and that save persists whatever ids were derived
    on the way in. It is kept as a witness for the invariant, not as evidence
    for that change.
    """
    first = model_dir("dup")
    second = model_dir("dup", under="elsewhere")

    first_id = imported_id(capsys, first)
    second_id = imported_id(capsys, second)
    assert first_id != second_id

    assert run("models", "config", "--json", second_id, "display_name=Second") == 0
    assert run("models", "forget", str(first)) == 0
    capsys.readouterr()

    surviving = load_registry().report()
    assert [entry.entry.id for entry in surviving] == [second_id]
    assert load_model_settings().overrides[second_id] == {"display_name": "Second"}


def test_a_version_one_library_keeps_its_ids_across_reads(capsys, model_dir) -> None:
    """The migration is written back, not re-derived on every read."""
    directory = model_dir("my-own-gpt-oss")
    from quantum_codex.config import write_json

    write_json(
        models_path(),
        {
            "version": 1,
            "roots": [],
            "models": [{"path": str(directory), "source": "imported"}],
        },
    )

    first = [entry.entry.id for entry in load_registry().report()]
    stored = json.loads(models_path().read_text())

    assert stored["version"] == 2
    assert [model["id"] for model in stored["models"]] == first


# -- a library that has become ambiguous -------------------------------------


def test_a_contested_name_does_not_stop_the_daemon_from_starting(
    capsys, model_dir, engine
) -> None:
    """Containment, at the boundary where the catalogue is actually built.

    Importing a second copy of a model is enough to make two entries claim one
    served name, and no configuration was involved. Building the catalogue by
    raising made that one ambiguous pair stop the whole daemon -- every other
    model with it, and with no running server left to fix it through.
    """
    first = model_dir("dup")
    model_dir("dup", under="elsewhere")
    other = model_dir("distinct-model")
    imported_id(capsys, first)
    imported_id(capsys, first.parent.parent / "elsewhere" / "dup")
    imported_id(capsys, other)
    engine.completion = completion_tokens("done")

    app = app_module.create_app(host="127.0.0.1", port=8123)
    with TestClient(app) as client:
        published = client.get("/v1/models").json()

        assert [m["slug"] for m in published["models"]] == ["distinct-model"]
        refused = client.post(
            "/v1/responses", json={"model": "dup", "input": "hi", "stream": False}
        )
        assert refused.status_code == 400
        answered = client.post(
            "/v1/responses",
            json={"model": "distinct-model", "input": "hi", "stream": False},
        )
        assert answered.status_code == 200, answered.text


# -- adapters ----------------------------------------------------------------
#
# An adapter path is the one model setting that names a place on disk, so the
# save boundary has to decide what it refuses. The rule is: refuse what is
# provably wrong, store and report what is merely currently unreachable.


@pytest.fixture
def adapter_dir(tmp_path):
    def build(name: str = "adapter", *, config: dict | None = None, names: list[str] | None = None):
        import struct

        directory = tmp_path / "adapters" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text(
            json.dumps(
                config
                if config is not None
                else {
                    "model": "/weights/my-own-gpt-oss",
                    "fine_tune_type": "lora",
                    "num_layers": 8,
                    "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
                }
            )
        )
        header = json.dumps(
            {
                name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]}
                for name in (names or ["model.layers.0.self_attn.q_proj.lora_a"])
            }
        ).encode()
        (directory / "adapters.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header
        )
        return directory

    return build


def test_an_adapter_is_stored_against_the_model_and_reaches_the_catalogue(
    capsys, model_dir, client, adapter_dir
) -> None:
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    adapter = adapter_dir()

    assert run("models", "config", "--json", model_id, f"adapter_path={adapter}") == 0
    capsys.readouterr()

    assert load_model_settings().overrides[model_id] == {"adapter_path": str(adapter)}
    app_module.refresh_registry(client.app.state.context)
    served = client.app.state.context.registry.by_library_id(model_id)
    assert served.adapter_path == str(adapter)


def test_an_unusable_adapter_is_refused_and_nothing_is_written(
    capsys, model_dir, tmp_path
) -> None:
    """A directory the user just chose, refused while they are looking at it."""
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    capsys.readouterr()
    empty = tmp_path / "not-an-adapter"
    empty.mkdir()

    assert run("models", "config", "--json", model_id, f"adapter_path={empty}") != 0

    assert "adapter_config.json" in capsys.readouterr().err
    assert load_model_settings().overrides == {}


def test_an_adapter_on_an_unmounted_volume_is_stored_and_not_refused(
    capsys, model_dir
) -> None:
    """The counter-test that stops the refusal over-reaching.

    An external drive being on the desk rather than plugged in is a normal
    situation, and refusing here would make the setting not only unstorable but
    unclearable — clearing arrives through this same path.
    """
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    capsys.readouterr()
    absent = "/Volumes/NotMounted/adapters/style-fr"

    assert run("models", "config", "--json", model_id, f"adapter_path={absent}") == 0

    assert load_model_settings().overrides[model_id] == {"adapter_path": absent}


def test_the_stored_adapter_is_reported_with_what_is_at_that_path(
    capsys, model_dir
) -> None:
    """So a form can say "set, but unreachable" without a second command."""
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    capsys.readouterr()
    absent = "/Volumes/NotMounted/adapters/style-fr"
    assert run("models", "config", "--json", model_id, f"adapter_path={absent}") == 0
    capsys.readouterr()

    assert run("models", "config", "--json", model_id) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["effective"]["adapter_path"] == absent
    assert payload["adapter"]["verdict"] == "UNUSABLE"
    assert "volume" in payload["adapter"]["reasons"][0]


def test_a_model_with_no_adapter_reports_none_rather_than_a_default(
    capsys, model_dir
) -> None:
    """`inherited` must not claim an adapter the model does not have."""
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    capsys.readouterr()

    assert run("models", "config", "--json", model_id) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter"] is None
    assert "adapter_path" not in payload["defaults"]
    assert "adapter_path" not in payload["inherited"]


def test_clearing_the_adapter_returns_the_model_to_its_base_weights(
    capsys, model_dir, client, adapter_dir
) -> None:
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    assert run("models", "config", "--json", model_id, f"adapter_path={adapter_dir()}") == 0
    capsys.readouterr()

    assert run("models", "config", "--json", model_id, "adapter_path=") == 0
    capsys.readouterr()

    # Cleared, not stored as null: a persisted null would be a value.
    assert load_model_settings().overrides.get(model_id, {}) == {}
    app_module.refresh_registry(client.app.state.context)
    assert client.app.state.context.registry.by_library_id(model_id).adapter_path is None


def test_an_adapter_naming_another_model_is_stored_with_a_note(
    capsys, model_dir, adapter_dir
) -> None:
    """A label, not a verdict.

    `mlx_lm.lora` records whatever was on its command line, so a mismatch here
    is weak evidence. The authority is the load, which compares tensor names
    against the weights themselves.
    """
    model_id = imported_id(capsys, model_dir("my-own-gpt-oss"))
    capsys.readouterr()
    adapter = adapter_dir(
        "elsewhere",
        config={
            "model": "somewhere/completely-different",
            "fine_tune_type": "lora",
            "num_layers": 8,
            "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
        },
    )

    assert run("models", "config", "--json", model_id, f"adapter_path={adapter}") == 0

    assert "completely-different" in capsys.readouterr().err
    assert load_model_settings().overrides[model_id] == {"adapter_path": str(adapter)}


def test_the_adapter_inspector_gates_a_script_on_its_exit_code(capsys, adapter_dir, tmp_path):
    assert run("models", "inspect-adapter", "--json", str(adapter_dir())) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "USABLE"

    assert run("models", "inspect-adapter", str(tmp_path / "nothing-here")) == 1
