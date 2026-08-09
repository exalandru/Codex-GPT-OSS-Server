"""Model compatibility scanning (cahier 25).

This project is GPT-OSS-only by design, so the scanner recognises and validates
GPT-OSS variants rather than discovering model families. A model it cannot vouch
for is refused with a reason, not loaded hopefully and allowed to fail somewhere
less legible.

The verdict is deliberately three-valued. Collapsing ``SUPPORTED_WITH_WARNING``
into either neighbour would lose the only interesting case: a model that will
work but not the way the operator expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# The architecture this server exists to serve. Anything else is out of scope by
# design, not by omission.
GPT_OSS_MODEL_TYPE = "gpt_oss"
GPT_OSS_ARCHITECTURE = "GptOssForCausalLM"

# Weight formats MLX can load directly.
WEIGHT_SUFFIXES = (".safetensors",)


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_WARNING = "SUPPORTED_WITH_WARNING"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ModelReport:
    """What inspection established about a directory."""

    path: str
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    model_type: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    layers: int | None = None
    experts: int | None = None
    disk_bytes: int = 0
    shards: int = 0

    @property
    def usable(self) -> bool:
        return self.verdict is not Verdict.UNSUPPORTED

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "verdict": self.verdict.value,
            "reasons": self.reasons,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "layers": self.layers,
            "experts": self.experts,
            "disk_bytes": self.disk_bytes,
            "shards": self.shards,
        }


def _directory_bytes(path: Path) -> tuple[int, int]:
    total = 0
    shards = 0
    for entry in path.iterdir():
        if entry.is_file() and entry.suffix in WEIGHT_SUFFIXES:
            total += entry.stat().st_size
            shards += 1
    return total, shards


def inspect_model(path: str | Path) -> ModelReport:
    """Decide whether this directory is a GPT-OSS model this server can run."""
    directory = Path(path).expanduser()
    report = ModelReport(path=str(directory), verdict=Verdict.UNSUPPORTED)

    if not directory.exists():
        # An unmounted external volume looks exactly like this, and it is a
        # normal situation rather than a broken model (cahier 43).
        report.reasons.append(
            "Path does not exist. If it lives on an external volume, the volume may not be mounted."
        )
        return report

    if not directory.is_dir():
        report.reasons.append("Not a directory. A model is a directory of weights and config.")
        return report

    config_file = directory / "config.json"
    if not config_file.is_file():
        report.reasons.append("No config.json, so the architecture cannot be established.")
        return report

    try:
        config = json.loads(config_file.read_text())
    except json.JSONDecodeError as exc:
        report.reasons.append(f"config.json is not valid JSON: {exc}")
        return report

    report.model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    report.architecture = architectures[0] if architectures else None
    report.context_length = config.get("max_position_embeddings")
    report.layers = config.get("num_hidden_layers")
    report.experts = config.get("num_local_experts")
    report.disk_bytes, report.shards = _directory_bytes(directory)

    quantization = config.get("quantization")
    if isinstance(quantization, dict):
        mode = quantization.get("mode", "quantized")
        bits = quantization.get("bits", "?")
        report.quantization = f"{mode}-{bits}bit"

    # -- refusals ------------------------------------------------------------

    if report.model_type != GPT_OSS_MODEL_TYPE:
        report.reasons.append(
            f"model_type is {report.model_type!r}, not {GPT_OSS_MODEL_TYPE!r}. "
            "This server serves GPT-OSS only, by design — it renders Harmony prompts "
            "that other families do not understand."
        )
        return report

    if report.architecture and report.architecture != GPT_OSS_ARCHITECTURE:
        report.reasons.append(
            f"architecture is {report.architecture!r}, not {GPT_OSS_ARCHITECTURE!r}."
        )
        return report

    if report.shards == 0:
        report.reasons.append("No .safetensors weights found; MLX has nothing to load.")
        return report

    if not (directory / "tokenizer.json").is_file():
        report.reasons.append("No tokenizer.json, so prompts cannot be tokenised.")
        return report

    index = directory / "model.safetensors.index.json"
    if report.shards > 1 and not index.is_file():
        report.reasons.append(
            f"{report.shards} weight shards but no model.safetensors.index.json to map them."
        )
        return report

    # -- warnings ------------------------------------------------------------

    warnings: list[str] = []
    if report.quantization is None:
        warnings.append(
            "Unquantized weights. This loads, but needs far more memory than the "
            "mxfp4 builds this server is normally run with."
        )
    if report.context_length and report.context_length < 32768:
        warnings.append(
            f"Context window is only {report.context_length} tokens, which a Codex "
            "session will exhaust quickly."
        )

    if warnings:
        report.verdict = Verdict.SUPPORTED_WITH_WARNING
        report.reasons = warnings
        return report

    report.verdict = Verdict.SUPPORTED
    report.reasons.append(
        f"GPT-OSS, {report.quantization}, {report.layers} layers, "
        f"{report.context_length} token context."
    )
    return report
