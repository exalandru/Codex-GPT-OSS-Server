"""LoRA adapter inspection.

The inspector exists to refuse, clearly, rather than to let a malformed adapter
raise a KeyError on the MLX worker thread halfway through a load. So the
interesting assertions are the refusals and their reasons — and the two
deliberate non-refusals, which stop it from refusing setups that work.
"""

from __future__ import annotations

import json
import struct

import pytest

from quantum_codex.inspect_adapter import (
    AdapterVerdict,
    adapter_tensor_names,
    describes_the_same_model,
    inspect_adapter,
)

# What `mlx_lm.lora` writes: `vars(args)`, so the base model and the training
# knobs sit beside the two keys `load_adapters` actually reads.
LORA_CONFIG = {
    "model": "/Volumes/Weights/gpt-oss-120b-mxfp4-bf16",
    "fine_tune_type": "lora",
    "num_layers": 16,
    "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
    "iters": 200,
}


def safetensors(names: list[str]) -> bytes:
    """A safetensors file with a real header and no tensor data.

    The inspector reads names and never reads a tensor, so the payload can be
    empty; what has to be real is the 8-byte little-endian header length.
    """
    header = json.dumps(
        {name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]} for name in names}
    ).encode()
    return struct.pack("<Q", len(header)) + header


@pytest.fixture
def adapter_dir(tmp_path):
    def build(
        config: dict | None = LORA_CONFIG,
        *,
        weights: bytes | None = None,
        names: list[str] | None = None,
    ):
        directory = tmp_path / "adapter"
        directory.mkdir(exist_ok=True)
        if config is not None:
            (directory / "adapter_config.json").write_text(json.dumps(config))
        if weights is not None:
            (directory / "adapters.safetensors").write_bytes(weights)
        elif names is not False:
            (directory / "adapters.safetensors").write_bytes(
                safetensors(names or ["model.layers.20.self_attn.q_proj.lora_a"])
            )
        return directory

    return build


def test_a_directory_mlx_lm_lora_wrote_is_usable(adapter_dir) -> None:
    report = inspect_adapter(adapter_dir())

    assert report.verdict is AdapterVerdict.USABLE
    assert report.fine_tune_type == "lora"
    assert report.rank == 8
    assert report.num_layers == 16
    assert report.trained_against == "/Volumes/Weights/gpt-oss-120b-mxfp4-bf16"
    assert report.tensor_count == 1
    assert report.usable


def test_a_missing_adapter_directory_suggests_the_likely_cause(tmp_path) -> None:
    report = inspect_adapter(tmp_path / "not-there")

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "volume" in report.reasons[0]


def test_an_adapter_without_its_config_is_refused(adapter_dir) -> None:
    report = inspect_adapter(adapter_dir(None))

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "adapter_config.json" in report.reasons[0]


def test_an_unreadable_adapter_config_carries_the_parse_error(tmp_path) -> None:
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{not json")
    (directory / "adapters.safetensors").write_bytes(safetensors(["a.lora_a"]))

    report = inspect_adapter(directory)

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "not valid JSON" in report.reasons[0]


def test_an_adapter_without_its_weights_is_refused(tmp_path) -> None:
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text(json.dumps(LORA_CONFIG))

    report = inspect_adapter(directory)

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "adapters.safetensors" in report.reasons[0]


def test_an_empty_weights_file_is_refused(adapter_dir) -> None:
    report = inspect_adapter(adapter_dir(weights=b""))

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "empty" in report.reasons[0]


def test_a_fine_tune_kind_this_server_does_not_understand_is_refused(adapter_dir) -> None:
    # `load_adapters` treats every unrecognised kind as plain LoRA. Refusing is
    # the difference between "not loaded" and "loaded as something else".
    report = inspect_adapter(adapter_dir({**LORA_CONFIG, "fine_tune_type": "qlora"}))

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "qlora" in report.reasons[0]


def test_lora_parameters_missing_a_rank_are_refused_before_the_worker_sees_them(
    adapter_dir,
) -> None:
    # `linear_to_lora_layers` indexes rank/scale/dropout without checking, so
    # this would otherwise be a KeyError raised on the MLX worker thread.
    report = inspect_adapter(
        adapter_dir({**LORA_CONFIG, "lora_parameters": {"scale": 20.0, "dropout": 0.0}})
    )

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "rank" in report.reasons[0]


def test_an_adapter_without_num_layers_is_refused(adapter_dir) -> None:
    config = {key: value for key, value in LORA_CONFIG.items() if key != "num_layers"}

    report = inspect_adapter(adapter_dir(config))

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "num_layers" in report.reasons[0]


def test_a_full_fine_tune_needs_no_lora_parameters(adapter_dir) -> None:
    # It replaces weights rather than adding layers, so `load_adapters` skips
    # the layer surgery entirely and never reads lora_parameters.
    report = inspect_adapter(adapter_dir({"fine_tune_type": "full"}))

    assert report.verdict is AdapterVerdict.USABLE_WITH_WARNING
    assert report.usable
    assert "replaces weights" in report.reasons[0]


def test_an_adapter_missing_its_fine_tune_type_is_read_as_lora(adapter_dir) -> None:
    # The same default `load_adapters` applies, so the two agree about what an
    # older adapter directory means.
    config = {key: value for key, value in LORA_CONFIG.items() if key != "fine_tune_type"}

    report = inspect_adapter(adapter_dir(config))

    assert report.verdict is AdapterVerdict.USABLE
    assert report.fine_tune_type == "lora"


def test_weights_that_are_not_safetensors_are_refused(adapter_dir) -> None:
    report = inspect_adapter(adapter_dir(weights=b"not a safetensors file at all"))

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "safetensors" in report.reasons[0]


def test_a_header_holding_no_tensors_is_refused(adapter_dir) -> None:
    report = inspect_adapter(adapter_dir(weights=safetensors([])))

    assert report.verdict is AdapterVerdict.UNUSABLE
    assert "no tensors" in report.reasons[0]


def test_the_tensor_names_are_read_without_loading_the_weights(adapter_dir) -> None:
    # The witness in the engine compares against these, and the save boundary
    # reads them without importing MLX.
    names = ["model.layers.0.self_attn.q_proj.lora_a", "model.layers.0.self_attn.q_proj.lora_b"]

    assert set(adapter_tensor_names(adapter_dir(names=names))) == set(names)


def test_the_format_s_own_metadata_entry_is_not_a_tensor(adapter_dir) -> None:
    directory = adapter_dir()
    header = json.dumps(
        {
            "__metadata__": {"format": "mlx"},
            "model.layers.0.self_attn.q_proj.lora_a": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 0],
            },
        }
    ).encode()
    (directory / "adapters.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)

    assert adapter_tensor_names(directory) == ("model.layers.0.self_attn.q_proj.lora_a",)


class TestTheModelItNames:
    """`trained_against` is a label, and the code treats it as one.

    `mlx_lm.lora` records whatever was on its command line — usually an HF repo
    id, while the copy being served is a local directory — and model
    directories get renamed. Refusing on a string comparison would refuse
    working setups, so the mismatch is reported and nothing more.
    """

    def test_the_same_directory_name_matches_through_a_different_prefix(self) -> None:
        report = inspect_adapter("/nowhere")
        report.trained_against = "mlx-community/gpt-oss-120b-mxfp4-bf16"

        assert describes_the_same_model(report, "/Volumes/Weights/gpt-oss-120b-mxfp4-bf16")

    def test_a_different_model_is_reported_as_different(self) -> None:
        report = inspect_adapter("/nowhere")
        report.trained_against = "mlx-community/gpt-oss-20b-MXFP4-Q8"

        assert not describes_the_same_model(report, "/Volumes/Weights/gpt-oss-120b-mxfp4-bf16")

    def test_an_adapter_that_names_nothing_is_not_treated_as_a_mismatch(self) -> None:
        report = inspect_adapter("/nowhere")

        assert report.trained_against is None
        assert describes_the_same_model(report, "/Volumes/Weights/gpt-oss-120b-mxfp4-bf16")

    def test_naming_another_model_does_not_make_the_adapter_unusable(self, adapter_dir) -> None:
        report = inspect_adapter(adapter_dir({**LORA_CONFIG, "model": "somewhere/else-7b"}))

        assert report.verdict is AdapterVerdict.USABLE
