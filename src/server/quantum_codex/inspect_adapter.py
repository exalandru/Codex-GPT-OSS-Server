"""LoRA adapter inspection.

The counterpart of :mod:`quantum_codex.inspect_model`, for the other kind of
directory a user can point this server at: an adapter produced by
``mlx_lm.lora``. Same three-valued verdict and same refuse-with-a-reason
discipline, for the same purpose — a directory that cannot work is refused while
the user is still looking at the form, not thirty seconds into a model load.

Two properties shape everything here.

``mlx_lm.tuner.utils.load_adapters`` indexes ``config["rank"]``, ``["scale"]``
and ``["dropout"]`` unconditionally, so a malformed ``adapter_config.json``
surfaces as a ``KeyError`` raised on the MLX worker thread — an unreadable
failure at the worst moment. Checking the shape here turns that into a sentence.

It also finishes with ``model.load_weights(..., strict=False)``, which skips
every key the model does not have. An adapter trained against another model
therefore loads without error and applies to nothing. That one cannot be caught
from the filesystem — only by comparing the two — so it is caught at load time
by the engine's witness, and this module exists to give that witness the names
to compare (:func:`adapter_tensor_names`).

Stdlib only, deliberately: the CLI's save boundary validates an adapter path on
every write, and importing MLX to read a JSON file and a file header would put a
GPU framework on that path.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

#: What `mlx_lm.lora` writes, and what `load_adapters` reads back.
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapters.safetensors"

#: Fine-tune kinds `load_adapters` implements distinct behaviour for. Anything
#: else it silently treats as plain LoRA, which is why an unknown value is
#: refused here rather than passed through: a future kind would otherwise be
#: applied as something it is not.
FINE_TUNE_TYPES = ("lora", "dora", "full")

#: Keys `linear_to_lora_layers` reads from `lora_parameters` without checking.
LORA_PARAMETERS = ("rank", "scale", "dropout")


class AdapterVerdict(StrEnum):
    USABLE = "USABLE"
    USABLE_WITH_WARNING = "USABLE_WITH_WARNING"
    UNUSABLE = "UNUSABLE"


@dataclass
class AdapterReport:
    """What inspection established about an adapter directory."""

    path: str
    verdict: AdapterVerdict
    reasons: list[str] = field(default_factory=list)
    fine_tune_type: str | None = None
    num_layers: int | None = None
    rank: int | None = None
    #: What ``mlx_lm.lora`` recorded as the base model at training time. It
    #: saves ``vars(args)``, so this is whatever was on the command line — an
    #: HF repo id as often as a local path. Reported, never enforced: the
    #: authority on whether an adapter fits is the engine's witness, which
    #: compares weights rather than a label.
    trained_against: str | None = None
    tensor_count: int = 0
    weight_bytes: int = 0

    @property
    def usable(self) -> bool:
        return self.verdict is not AdapterVerdict.UNUSABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "verdict": self.verdict.value,
            "reasons": self.reasons,
            "fine_tune_type": self.fine_tune_type,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "trained_against": self.trained_against,
            "tensor_count": self.tensor_count,
            "weight_bytes": self.weight_bytes,
        }


def adapter_tensor_names(path: str | Path) -> tuple[str, ...]:
    """The tensor names in ``adapters.safetensors``, without loading it.

    The safetensors header is an 8-byte little-endian length followed by that
    many bytes of JSON whose keys are the tensor names. Reading it directly is
    what lets the save boundary check an adapter without MLX, and what lets the
    engine's witness compare against the same source the loader used.

    Raises whatever the filesystem or the parse raises; callers that need a
    verdict rather than an exception use :func:`inspect_adapter`.
    """
    weights = Path(path).expanduser() / ADAPTER_WEIGHTS
    size = weights.stat().st_size
    with weights.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        # The header lives inside the file, so the declared length is bounded by
        # it. Checked rather than trusted: the first eight bytes of any file
        # decode as *some* 64-bit number, and reading that many bytes from a
        # file that is not safetensors asks for an allocation the machine
        # cannot serve. A user pointing at the wrong directory must get a
        # sentence, not a MemoryError inside the daemon.
        if length > size - 8:
            raise ValueError(
                f"declares a {length}-byte header inside a {size}-byte file, "
                "so it is not safetensors"
            )
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError("the safetensors header is not a JSON object")
    # `__metadata__` is the format's own reserved entry, not a tensor.
    return tuple(name for name in header if name != "__metadata__")


def inspect_adapter(path: str | Path) -> AdapterReport:
    """Decide whether this directory is an adapter MLX can apply."""
    directory = Path(path).expanduser()
    report = AdapterReport(path=str(directory), verdict=AdapterVerdict.UNUSABLE)

    if not directory.exists():
        # Same wording as `inspect_model`, for the same reason: an unmounted
        # external volume looks exactly like this and is a normal situation.
        report.reasons.append(
            "Path does not exist. If it lives on an external volume, the volume may not "
            "be mounted."
        )
        return report

    if not directory.is_dir():
        report.reasons.append(
            "Not a directory. An adapter is a directory holding "
            f"{ADAPTER_CONFIG} and {ADAPTER_WEIGHTS}."
        )
        return report

    config_file = directory / ADAPTER_CONFIG
    if not config_file.is_file():
        report.reasons.append(
            f"No {ADAPTER_CONFIG}, so the adapter's shape cannot be established. "
            "A directory written by `mlx_lm.lora --adapter-path` has one."
        )
        return report

    try:
        config = json.loads(config_file.read_text())
    except json.JSONDecodeError as exc:
        report.reasons.append(f"{ADAPTER_CONFIG} is not valid JSON: {exc}")
        return report

    if not isinstance(config, dict):
        report.reasons.append(f"{ADAPTER_CONFIG} is not a JSON object.")
        return report

    weights_file = directory / ADAPTER_WEIGHTS
    report.fine_tune_type = config.get("fine_tune_type") or "lora"
    report.trained_against = config.get("model") if isinstance(config.get("model"), str) else None
    parameters = config.get("lora_parameters")
    if isinstance(parameters, dict) and isinstance(parameters.get("rank"), int):
        report.rank = parameters["rank"]
    if isinstance(config.get("num_layers"), int):
        report.num_layers = config["num_layers"]

    # -- refusals ------------------------------------------------------------

    if not weights_file.is_file():
        report.reasons.append(
            f"No {ADAPTER_WEIGHTS}; there are no adapter weights to apply."
        )
        return report

    report.weight_bytes = weights_file.stat().st_size
    if report.weight_bytes == 0:
        report.reasons.append(f"{ADAPTER_WEIGHTS} is empty.")
        return report

    if report.fine_tune_type not in FINE_TUNE_TYPES:
        # `load_adapters` branches on "full" and "dora" and treats everything
        # else as LoRA. An unrecognised kind would therefore be applied as
        # something it is not, silently, which is worse than not loading.
        report.reasons.append(
            f"fine_tune_type is {report.fine_tune_type!r}, which this server does not "
            f"understand. Known kinds: {', '.join(FINE_TUNE_TYPES)}."
        )
        return report

    if report.fine_tune_type != "full":
        # Exactly what `linear_to_lora_layers` indexes without checking. Read
        # here so a missing key is a sentence rather than a KeyError raised on
        # the MLX worker thread halfway through a load.
        if not isinstance(config.get("num_layers"), int):
            report.reasons.append(
                f"{ADAPTER_CONFIG} has no integer num_layers, which decides how many "
                "blocks the adapter applies to."
            )
            return report
        if not isinstance(parameters, dict):
            report.reasons.append(f"{ADAPTER_CONFIG} has no lora_parameters object.")
            return report
        missing = [key for key in LORA_PARAMETERS if key not in parameters]
        if missing:
            report.reasons.append(
                f"lora_parameters is missing {', '.join(missing)}, which MLX reads "
                "while building the adapter layers."
            )
            return report

    try:
        names = adapter_tensor_names(directory)
    except (OSError, ValueError, struct.error, UnicodeDecodeError) as exc:
        report.reasons.append(f"{ADAPTER_WEIGHTS} is not a readable safetensors file: {exc}")
        return report

    report.tensor_count = len(names)
    if report.tensor_count == 0:
        report.reasons.append(f"{ADAPTER_WEIGHTS} holds no tensors.")
        return report

    # -- warnings ------------------------------------------------------------

    warnings: list[str] = []
    if report.fine_tune_type == "full":
        warnings.append(
            "A full fine-tune, not a LoRA: it replaces weights rather than adding to "
            "them, and only the parameters it names are replaced."
        )

    if warnings:
        report.verdict = AdapterVerdict.USABLE_WITH_WARNING
        report.reasons = warnings
        return report

    report.verdict = AdapterVerdict.USABLE
    detail = f"{report.fine_tune_type}, {report.tensor_count} tensors"
    if report.rank is not None:
        detail += f", rank {report.rank}"
    if report.num_layers is not None:
        detail += f", {report.num_layers} layers"
    report.reasons.append(detail + ".")
    return report


def describes_the_same_model(report: AdapterReport, model_path: str | Path) -> bool:
    """Whether the adapter names the model it is about to be applied to.

    A weak signal, and treated as one everywhere it is used. ``mlx_lm.lora``
    records the string that was on its command line, which is frequently an HF
    repo id while the served copy is a local directory, and users rename model
    directories. It is worth showing next to an adapter and never worth refusing
    one over — an adapter that truly does not fit is caught at load, by
    comparing tensor names against the weights themselves.
    """
    if not report.trained_against:
        return True
    return Path(report.trained_against).name == Path(model_path).name
