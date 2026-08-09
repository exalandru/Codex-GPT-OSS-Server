"""Model compatibility scanning.

The scanner exists to refuse, clearly, rather than to let an unsupported model
fail later somewhere less legible. So the interesting assertions are the
refusals and their reasons.
"""

from __future__ import annotations

import json

import pytest

from quantum_codex.inspect_model import Verdict, inspect_model

GPT_OSS_CONFIG = {
    "model_type": "gpt_oss",
    "architectures": ["GptOssForCausalLM"],
    "num_hidden_layers": 24,
    "num_local_experts": 32,
    "max_position_embeddings": 131072,
    "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
}


@pytest.fixture
def model_dir(tmp_path):
    def build(config: dict | None = GPT_OSS_CONFIG, *, shards: int = 1, tokenizer: bool = True):
        directory = tmp_path / "model"
        directory.mkdir(exist_ok=True)
        if config is not None:
            (directory / "config.json").write_text(json.dumps(config))
        for i in range(shards):
            (directory / f"model-{i:05d}.safetensors").write_bytes(b"x" * 1024)
        if shards > 1:
            (directory / "model.safetensors.index.json").write_text("{}")
        if tokenizer:
            (directory / "tokenizer.json").write_text("{}")
        return directory

    return build


def test_a_real_gpt_oss_directory_is_supported(model_dir) -> None:
    report = inspect_model(model_dir())

    assert report.verdict is Verdict.SUPPORTED
    assert report.model_type == "gpt_oss"
    assert report.quantization == "mxfp4-4bit"
    assert report.context_length == 131072
    assert report.experts == 32
    assert report.disk_bytes == 1024


def test_a_missing_path_suggests_the_likely_cause(tmp_path) -> None:
    # An unmounted external volume is the common case, and it is a normal
    # situation rather than a broken model (cahier 43).
    report = inspect_model(tmp_path / "not-there")

    assert report.verdict is Verdict.UNSUPPORTED
    assert "volume" in report.reasons[0]


def test_another_model_family_is_refused_by_design(model_dir) -> None:
    report = inspect_model(
        model_dir({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )

    assert report.verdict is Verdict.UNSUPPORTED
    # The reason must say this is a design choice, not a gap: the server renders
    # Harmony prompts other families do not understand.
    assert "by design" in report.reasons[0]


def test_a_directory_without_weights_is_refused(model_dir) -> None:
    report = inspect_model(model_dir(shards=0))

    assert report.verdict is Verdict.UNSUPPORTED
    assert "safetensors" in report.reasons[0]


def test_a_directory_without_a_tokenizer_is_refused(model_dir) -> None:
    report = inspect_model(model_dir(tokenizer=False))

    assert report.verdict is Verdict.UNSUPPORTED
    assert "tokenis" in report.reasons[0].lower() or "tokenizer" in report.reasons[0]


def test_sharded_weights_need_their_index(model_dir) -> None:
    directory = model_dir(shards=3)
    (directory / "model.safetensors.index.json").unlink()

    report = inspect_model(directory)

    assert report.verdict is Verdict.UNSUPPORTED
    assert "index" in report.reasons[0]


def test_a_missing_config_is_refused(model_dir) -> None:
    report = inspect_model(model_dir(config=None))

    assert report.verdict is Verdict.UNSUPPORTED
    assert "config.json" in report.reasons[0]


def test_unquantized_weights_warn_rather_than_refuse(model_dir) -> None:
    """The distinction the three-valued verdict exists for.

    It will load; it will just want far more memory than the operator expects.
    Refusing would be wrong and staying silent would be worse.
    """
    config = {k: v for k, v in GPT_OSS_CONFIG.items() if k != "quantization"}
    report = inspect_model(model_dir(config))

    assert report.verdict is Verdict.SUPPORTED_WITH_WARNING
    assert report.usable is True
    assert "memory" in report.reasons[0]


def test_a_small_context_window_warns(model_dir) -> None:
    config = {**GPT_OSS_CONFIG, "max_position_embeddings": 8192}
    report = inspect_model(model_dir(config))

    assert report.verdict is Verdict.SUPPORTED_WITH_WARNING
    assert "Codex session" in report.reasons[0]
