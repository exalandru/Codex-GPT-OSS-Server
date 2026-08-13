"""Applying a LoRA adapter to the resident weights.

`mlx_lm`'s `load_adapters` finishes with `load_weights(..., strict=False)`,
which skips every key the model does not have and returns nothing. An adapter
trained against another model therefore loads without error and applies to
nothing: the configuration says one thing, the weights answering requests say
another, and no exception is raised anywhere.

These pin the witness that closes that gap, and the one property that makes it
worth having: what is reported is what was *applied*, never what was asked for.

The engine is real; the weights are not. Nothing here loads a model — the
witness reads the parameter tree, so a stub tree is exactly as discriminating as
sixty gigabytes of one.
"""

from __future__ import annotations

import asyncio
import json
import struct

import pytest

from quantum_codex.inference.engine import AdapterMismatchError, EngineState, MlxEngine

LORA_CONFIG = {
    "model": "/models/gpt-oss-20b-mxfp4-bf16",
    "fine_tune_type": "lora",
    "num_layers": 2,
    "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
}

# The names `mlx_lm.lora` writes: the base module path plus the two LoRA
# factors. Measured against a real 120B in the same shape as this.
ADAPTER_TENSORS = [
    "model.layers.0.self_attn.q_proj.lora_a",
    "model.layers.0.self_attn.q_proj.lora_b",
    "model.layers.1.mlp.experts.down_proj.lora_a",
    "model.layers.1.mlp.experts.down_proj.lora_b",
]


def write_adapter(directory, names=ADAPTER_TENSORS, config=LORA_CONFIG):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps(config))
    header = json.dumps(
        {name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]} for name in names}
    ).encode()
    (directory / "adapters.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    return directory


class StubModel:
    """Something with a parameter tree, which is all the witness reads."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def parameters(self):
        tree: dict = {}
        for name in self._names:
            node = tree
            *parents, leaf = name.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = 0.0
        return tree


@pytest.fixture
def engine():
    engine = MlxEngine()
    try:
        yield engine
    finally:
        engine.shutdown()


def witness(engine, adapter_path, model_names):
    engine._model = StubModel(model_names)  # noqa: SLF001
    engine._model_path = adapter_path.parent / "model"  # noqa: SLF001
    return engine._witness_adapter(adapter_path)  # noqa: SLF001


def test_an_adapter_that_matches_no_parameter_of_the_model_is_refused(engine, tmp_path) -> None:
    """The failure this whole mechanism exists for.

    Nothing raised inside MLX: the load succeeded, and every tensor was
    discarded. Without this the model answers from its base weights while the
    configuration, the status endpoint and the form all say an adapter is
    applied.
    """
    adapter = write_adapter(tmp_path / "adapter")

    with pytest.raises(AdapterMismatchError) as caught:
        witness(engine, adapter, ["model.layers.0.attention.query.weight"])

    message = str(caught.value)
    # Names one tensor from the adapter and the remedy, so the reader can tell
    # which of the two directories is the wrong one.
    assert "lora_a" in message
    assert "trained against a different model" in message
    assert "Clear the LoRA adapter" in message


def test_the_adapter_says_which_model_it_was_trained_against(engine, tmp_path) -> None:
    """A label, and the only clue available when the names disagree."""
    adapter = write_adapter(tmp_path / "adapter")

    with pytest.raises(AdapterMismatchError) as caught:
        witness(engine, adapter, ["something.else.weight"])

    assert "/models/gpt-oss-20b-mxfp4-bf16" in str(caught.value)


def test_a_fully_matching_adapter_reports_every_tensor_applied(engine, tmp_path) -> None:
    adapter = write_adapter(tmp_path / "adapter")

    applied = witness(engine, adapter, [*ADAPTER_TENSORS, "model.embed_tokens.weight"])

    assert applied.applied_tensors == 4
    assert applied.adapter_tensors == 4
    assert applied.fine_tune_type == "lora"
    assert applied.path == str(adapter)


def test_a_partly_matching_adapter_loads_and_says_how_much_applied(engine, tmp_path) -> None:
    """Legitimate, and not a refusal.

    An adapter trained over fewer blocks than this model has, or over an
    explicit subset of layer keys, lands here. Refusing would refuse correctly
    trained adapters; saying nothing would hide a partial application.
    """
    adapter = write_adapter(tmp_path / "adapter")

    applied = witness(engine, adapter, ADAPTER_TENSORS[:2])

    assert applied.applied_tensors == 2
    assert applied.adapter_tensors == 4


def test_the_witness_measures_the_weights_rather_than_echoing_the_setting(
    engine, tmp_path
) -> None:
    """The discriminating property of the reported figures.

    Both adapters below are configured identically. What separates them is only
    what reached the weights, so a report copied from the setting would be
    identical for the two and useless for telling them apart.
    """
    matching = write_adapter(tmp_path / "matching")
    partial = write_adapter(tmp_path / "partial")

    full = witness(engine, matching, ADAPTER_TENSORS)
    half = witness(engine, partial, ADAPTER_TENSORS[:2])

    assert (full.applied_tensors, half.applied_tensors) == (4, 2)


def test_a_full_fine_tune_is_reported_as_what_it_is(engine, tmp_path) -> None:
    adapter = write_adapter(
        tmp_path / "adapter", names=["model.embed_tokens.weight"], config={"fine_tune_type": "full"}
    )

    applied = witness(engine, adapter, ["model.embed_tokens.weight"])

    assert applied.fine_tune_type == "full"


# -- the load path -------------------------------------------------------------


def stub_mlx_load(monkeypatch, model_names, *, calls: list):
    """Replace the weight load, keeping everything the engine does around it."""
    import mlx_lm.utils

    def fake_load(path, adapter_path=None):  # noqa: ANN001
        calls.append((path, adapter_path))
        return StubModel(model_names), object()

    monkeypatch.setattr(mlx_lm.utils, "load", fake_load)
    monkeypatch.setattr(MlxEngine, "_warm_up", lambda self: None)


def model_dir(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    return directory


def test_the_configured_adapter_reaches_mlx(monkeypatch, engine, tmp_path) -> None:
    calls: list = []
    stub_mlx_load(monkeypatch, ADAPTER_TENSORS, calls=calls)
    adapter = write_adapter(tmp_path / "adapter")

    loaded = asyncio.run(
        engine.load(model_dir(tmp_path), "gpt-oss-20b", 131072, adapter_path=adapter)
    )

    assert calls == [(str(model_dir(tmp_path)), str(adapter))]
    assert loaded.adapter is not None
    assert loaded.adapter.applied_tensors == 4


def test_a_model_without_an_adapter_asks_mlx_for_none(monkeypatch, engine, tmp_path) -> None:
    calls: list = []
    stub_mlx_load(monkeypatch, ADAPTER_TENSORS, calls=calls)

    loaded = asyncio.run(engine.load(model_dir(tmp_path), "gpt-oss-20b", 131072))

    assert calls == [(str(model_dir(tmp_path)), None)]
    assert loaded.adapter is None


def test_a_refused_adapter_leaves_the_engine_holding_nothing(
    monkeypatch, engine, tmp_path
) -> None:
    """The base weights are already resident when the witness refuses.

    `load_adapters` runs after `load_model`, so the failure arrives with tens of
    gigabytes materialised and nothing assigned to hold them. The engine must
    end up in the same state as a load that never started, or the next load's
    release guard has nothing to release and the memory stays held.
    """
    stub_mlx_load(monkeypatch, ["model.layers.0.attention.query.weight"], calls=[])
    adapter = write_adapter(tmp_path / "adapter")

    with pytest.raises(AdapterMismatchError):
        asyncio.run(engine.load(model_dir(tmp_path), "gpt-oss-20b", 131072, adapter_path=adapter))

    assert engine.state is EngineState.FAILED
    assert engine.loaded_model is None
    assert engine._model is None  # noqa: SLF001
    assert engine._adapter_path is None  # noqa: SLF001


def test_the_resident_model_reports_the_adapter_it_is_serving_with(
    monkeypatch, engine, tmp_path
) -> None:
    """What `/internal/status` publishes as the model's shape."""
    stub_mlx_load(monkeypatch, ADAPTER_TENSORS, calls=[])
    adapter = write_adapter(tmp_path / "adapter")

    asyncio.run(engine.load(model_dir(tmp_path), "gpt-oss-20b", 131072, adapter_path=adapter))

    resident = engine.loaded_model
    assert resident.adapter.path == str(adapter)
    assert resident.adapter.adapter_tensors == 4


def test_releasing_the_model_forgets_the_adapter_too(monkeypatch, engine, tmp_path) -> None:
    stub_mlx_load(monkeypatch, ADAPTER_TENSORS, calls=[])
    adapter = write_adapter(tmp_path / "adapter")
    asyncio.run(engine.load(model_dir(tmp_path), "gpt-oss-20b", 131072, adapter_path=adapter))

    asyncio.run(engine.unload())

    assert engine._adapter_path is None  # noqa: SLF001
    assert engine.loaded_model is None
