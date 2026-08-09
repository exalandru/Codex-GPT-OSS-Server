"""Served-model capability definition (D5).

One internal definition per served model, in this project's own vocabulary. It
is the single source of truth for three consumers:

1. ``GET /v1/models``, rendered in the schema Codex expects;
2. request validation, which refuses what the model cannot do;
3. the desktop control plane, later.

Keeping the definition semantic rather than wire-shaped is what lets the same
facts be presented in a client's schema without that schema becoming the way the
server thinks about its own capabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonical import ReasoningEffort
from .library.catalog import defaults_for

logger = logging.getLogger(__name__)

# GPT-OSS has exactly these reasoning levels. Levels belonging to other model
# families do not exist here and must never be mapped onto one of these.
GPT_OSS_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    ReasoningEffort.LOW,
    ReasoningEffort.MEDIUM,
    ReasoningEffort.HIGH,
)


@dataclass(frozen=True)
class ServedModel:
    """What one served model can actually do."""

    slug: str
    display_name: str
    context_window: int

    default_reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    reasoning_efforts: tuple[ReasoningEffort, ...] = GPT_OSS_REASONING_EFFORTS

    # Whether this server can route a tool call back to the client at all.
    # If it cannot, no tool may be advertised: a model that calls a tool nothing
    # can execute produces a turn that cannot be completed.
    #
    # Note this governs what *this server* advertises and accepts. It does not
    # govern Codex's own tool surface, which Codex decides from its feature
    # flags
    supports_tools: bool = True

    # Harmony's `<|call|>` ends the assistant turn so the tool result can come
    # back. One call per turn is correct Harmony semantics, not a limitation to
    # be worked around, and advertising otherwise would be a false claim.
    supports_parallel_tool_calls: bool = False

    # No provider-side search executor exists here. The client may still run its
    # own tools; this is only about tools this server would have to execute.
    supports_hosted_search: bool = False

    input_modalities: tuple[str, ...] = ("text",)

    # Percentage of the context window advertised as usable. 100 means the
    # reported window matches the real KV limit; a lower value hands the client
    # headroom for its own compaction.
    effective_context_percent: int = 100

    # System instructions handed to the client when it has none of its own.
    # Filled in Slice 2, where the Codex-native metadata endpoint lands.
    base_instructions: str | None = None

    # Per-model generation defaults, when the user set one. `None` means
    # "inherit the server default" -- storing a resolved value here would freeze
    # it, so a later change to the default would never reach this model.
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    quantization: str | None = None
    path: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def effective_context_window(self) -> int:
        return self.context_window * self.effective_context_percent // 100

    @property
    def codex_shell_type(self) -> str:
        """How Codex should treat the shell tool for this model.

        ``disabled`` makes Codex send no shell tool at all. That is the honest
        setting while this server cannot route a tool call back: advertising a
        shell the model may call, and then refusing the request that carries it,
        would break every turn (cahier 5.5).
        """
        return "default" if self.supports_tools else "disabled"

    def supports_effort(self, effort: ReasoningEffort) -> bool:
        return effort in self.reasoning_efforts


class ModelRegistry:
    """Every model this server *could* serve.

    Not the loaded one. The daemon advertises what is installed and usable, and
    a client picks from that list with its normal ``model`` field; whichever it
    names becomes resident on demand. One model is resident at a time (cahier
    7) — unified memory makes a second a real cost, not a convenience — but
    that is a property of the engine, not of what may be offered.
    """

    def __init__(self) -> None:
        self._models: dict[str, ServedModel] = {}

    def register(self, model: ServedModel) -> None:
        self._models[model.slug] = model

    def replace_all(self, models: Iterable[ServedModel]) -> None:
        """Swap the whole catalogue, atomically from a reader's point of view.

        Rebuilding in place would let a concurrent request see an empty
        registry and be told, wrongly, that its model is not served.
        """
        self._models = {model.slug: model for model in models}

    def clear(self) -> None:
        self._models.clear()

    def get(self, slug: str) -> ServedModel | None:
        return self._models.get(slug)

    def all(self) -> tuple[ServedModel, ...]:
        return tuple(self._models.values())


def slug_for(name: str) -> str:
    """A stable client-facing id for a model directory.

    Derived from the directory name with the quantisation suffix removed, so a
    checkout of ``gpt-oss-20b-mxfp4-bf16`` is addressed as ``gpt-oss-20b`` --
    which is what a user types after ``--model`` and what Codex shows.
    """
    slug = name.strip()
    for suffix in ("-mxfp4-bf16", "-MXFP4-Q8", "-mxfp4", "-MXFP4", "-bf16", "-8bit", "-4bit"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return slug.lower()


def served_models_from_library(
    reports: Iterable[Any],
    *,
    default_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[ServedModel, ...]:
    """Build the served catalogue from the model library's own reports.

    Only ``READY`` entries are advertised. A model whose volume is unplugged, or
    whose files are incomplete, is present in the library and deliberately
    absent here: offering it would produce a load failure at the exact moment a
    user was trying to start work.

    Slug collisions keep the first entry and log the loser, because two models
    answering to one name would make which weights served a request
    unknowable.
    """
    models: dict[str, ServedModel] = {}
    for report in reports:
        if not report.state.usable:
            continue
        slug = slug_for(report.entry.name)
        if slug in models:
            logger.warning(
                "Two installed models resolve to the slug %r; serving %s and ignoring %s",
                slug,
                models[slug].path,
                report.entry.path,
            )
            continue
        # Precedence, in one place:
        #
        #   what the model actually is (discovered from disk)
        #     -> the server's default
        #       -> the catalogue's default for this model, if it is a preset
        #         -> this model's persisted override
        #
        # A request may still narrow any of it later; that is the last level and
        # it belongs to the request path, not here.
        #
        # The catalogue sits above disk metadata on purpose for the presets:
        # `gpt-oss-20b-mxfp4-bf16` is where the weights live, and `gpt-oss-20b`
        # is what the model is called.
        chosen = {**defaults_for(slug), **(overrides or {}).get(slug, {})}
        effort = chosen.get("reasoning_effort")
        models[slug] = ServedModel(
            slug=slug,
            # The alias a client sees. A model's own property: two models cannot
            # share one, and it is not a fact about the daemon.
            display_name=chosen.get("served_model_name") or report.entry.name,
            context_window=int(chosen.get("context_length") or report.context_length or 131072),
            default_reasoning_effort=(
                ReasoningEffort(effort) if effort else default_effort
            ),
            quantization=report.quantization,
            path=report.entry.path,
            max_output_tokens=_optional_int(chosen.get("max_output_tokens")),
            temperature=_optional_float(chosen.get("temperature")),
            top_p=_optional_float(chosen.get("top_p")),
        )
    return tuple(models.values())


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
