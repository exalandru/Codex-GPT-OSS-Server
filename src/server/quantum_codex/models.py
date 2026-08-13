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
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonical import ReasoningEffort
from .config import ConfigError
from .library.catalog import defaults_for, display_name_for

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
    # Immutable library/settings identity. ``slug`` is the mutable name exposed
    # to Codex and accepted on the wire.
    library_id: str | None = None

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
    # An optional LoRA adapter applied on top of the weights at `path`. It sits
    # here, beside the path, rather than with the sampling defaults above: it is
    # part of *which weights these are*, not of how a request generates from
    # them. `load_identity` below is what makes that structural.
    adapter_path: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def effective_context_window(self) -> int:
        return self.context_window * self.effective_context_percent // 100

    @property
    def load_identity(self) -> tuple[str | None, str | None, int, str | None]:
        """What must match for resident weights to be the ones this model names.

        Deliberately not the slug. The slug is the name a request asks for; this
        is the set of facts that decide which weights answer it. Two models
        sharing a slug but not an adapter are different weights, and answering a
        request for one with the other is exactly the failure a lease exists to
        prevent — it simply could not be expressed before, because the name and
        the weights were the same question.

        Kept as one expression, in the type that owns the facts, so no consumer
        can build a second version of it and disagree.
        """
        return (self.library_id, self.path, self.context_window, self.adapter_path)

    @property
    def id(self) -> str:
        return self.library_id or self.slug

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
        """Wire routing: the served name, and only the served name.

        Deliberately not the library id. The id is QCS's internal identity; a
        client that could also address a model by it would be using a second
        Codex-facing name that nothing validates for uniqueness and that
        ``/v1/models`` never published.
        """
        return self._models.get(slug)

    def by_library_id(self, model_id: str) -> ServedModel | None:
        return next((model for model in self._models.values() if model.id == model_id), None)

    def select(self, value: str) -> ServedModel | None:
        """Resolve a *QCS-internal* selector: id, then served name, then path.

        What a profile stores, what the desktop selects with and what ``--model``
        accepts all arrive here. The stable id is tried first because that is
        what QCS persists: a served name that happens to equal another model's
        id must not shadow it, or renaming one model would silently redirect a
        profile pointing at another.
        """
        return (
            self.by_library_id(value)
            or self._models.get(value)
            or next((model for model in self._models.values() if model.path == value), None)
        )

    def all(self) -> tuple[ServedModel, ...]:
        return tuple(self._models.values())


def slug_for(name: str) -> str:
    """The legacy/catalogue slug derived from a model directory.

    Derived from the directory name with the quantisation suffix removed.
    Existing version-1 records used this as identity. New code keeps it only as
    a deterministic default; the library id and mutable served name are
    separate values.
    """
    slug = name.strip()
    for suffix in ("-mxfp4-bf16", "-MXFP4-Q8", "-mxfp4", "-MXFP4", "-bf16", "-8bit", "-4bit"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return slug.lower()


#: What a served name may be.
#:
#: The served name is not only a routing key. It is interpolated verbatim into
#: the `codex …` command and the `config.toml` fragment QCS generates, and it is
#: what a client sends on the wire. Constraining it here -- at the boundary that
#: owns the name -- is what keeps every consumer from needing its own escaping
#: rule, and what stops a quote or a newline from producing a Codex
#: configuration that says something the user did not write.
SERVED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

SERVED_NAME_HELP = (
    "1-64 characters: letters, digits, dot, dash, underscore or colon, "
    "starting with a letter or digit"
)


def served_name_problem(name: str) -> str | None:
    """Why this served name is unusable, or ``None`` when it is fine."""
    if not name.strip():
        return "Served as cannot be empty"
    if not SERVED_NAME.match(name.strip()):
        return f"Served as must be {SERVED_NAME_HELP}"
    return None


@dataclass(frozen=True)
class ResolvedModelNames:
    """The three deliberately separate identities of one library entry."""

    library_id: str
    display_name: str
    served_name: str
    catalog_slug: str


def resolved_model_names(
    report: Any,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> ResolvedModelNames:
    """Resolve immutable identity, UI name and Codex-facing name once.

    Overrides are read by library id and by nothing else. Falling back to the
    derived catalogue slug looks harmless -- version-1 settings were keyed by it
    -- but it can only ever fire for an entry whose id had to be disambiguated
    from an identically named directory, which is exactly the case where the
    slug belongs to a *different* physical model. The registry migration keeps
    the first entry's id equal to that slug, so genuine version-1 settings are
    matched by the id lookup alone.
    """
    catalog_slug = slug_for(report.entry.name)
    library_id = getattr(report.entry, "id", None) or catalog_slug
    stored = (overrides or {}).get(library_id, {})
    chosen = {**defaults_for(catalog_slug), **stored}

    served_name = str(chosen.get("served_model_name") or catalog_slug).strip()
    if not served_name:
        raise ConfigError(f"model {library_id!r} needs a non-empty served name")
    display_name = str(
        chosen.get("display_name") or display_name_for(catalog_slug) or report.entry.name
    ).strip()
    if not display_name:
        raise ConfigError(f"model {library_id!r} needs a non-empty display name")
    return ResolvedModelNames(
        library_id=library_id,
        display_name=display_name,
        served_name=served_name,
        catalog_slug=catalog_slug,
    )


@dataclass(frozen=True)
class ServedNameProblem:
    """One served name that no model may answer to.

    ``library_ids`` names every entry that wanted it, so a message can say which
    models are affected without a caller re-deriving the grouping.
    """

    served_name: str
    library_ids: tuple[str, ...]
    reason: str  # "duplicate" | "invalid"
    message: str


@dataclass(frozen=True)
class ServedCatalogue:
    """What may be served, and what may not.

    The two are separated rather than collapsed into an exception because the
    two boundaries that ask this question want different things from the answer.
    Configuration must refuse a save outright; a running daemon must keep
    serving every unaffected model while refusing to answer to an ambiguous
    name. Both enforce the same invariant -- one served name resolves to one set
    of weights -- and both read it from here.
    """

    models: tuple[ServedModel, ...] = ()
    problems: tuple[ServedNameProblem, ...] = ()


def resolve_served_catalogue(
    reports: Iterable[Any],
    *,
    default_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> ServedCatalogue:
    """Build the served catalogue, keeping unusable names out of it.

    Only ``READY`` entries are advertised. A model whose volume is unplugged, or
    whose files are incomplete, is present in the library and deliberately
    absent here: offering it would produce a load failure at the exact moment a
    user was trying to start work.

    A served name claimed by two models is served by neither. Keeping one would
    make a save look successful while other weights continued answering to the
    name the user chose, and choosing between them would be arbitrary -- the
    server cannot know which of the two the client meant.
    """
    candidates: dict[str, list[ServedModel]] = {}
    problems: list[ServedNameProblem] = []
    for report in reports:
        if not report.state.usable:
            continue
        names = resolved_model_names(report, overrides=overrides)
        slug = names.served_name
        invalid = served_name_problem(slug)
        if invalid is not None:
            problems.append(
                ServedNameProblem(
                    served_name=slug,
                    library_ids=(names.library_id,),
                    reason="invalid",
                    message=(
                        f"model {names.library_id!r} has an unusable served name "
                        f"{slug!r}: {invalid.lower()}"
                    ),
                )
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
        stored = (overrides or {}).get(names.library_id, {})
        chosen = {**defaults_for(names.catalog_slug), **stored}
        effort = chosen.get("reasoning_effort")
        candidates.setdefault(slug, []).append(
            ServedModel(
                slug=slug,
                display_name=names.display_name,
                context_window=int(
                    chosen.get("context_length") or report.context_length or 131072
                ),
                library_id=names.library_id,
                default_reasoning_effort=(
                    ReasoningEffort(effort) if effort else default_effort
                ),
                quantization=report.quantization,
                path=report.entry.path,
                adapter_path=_optional_path(chosen.get("adapter_path")),
                max_output_tokens=_optional_int(chosen.get("max_output_tokens")),
                temperature=_optional_float(chosen.get("temperature")),
                top_p=_optional_float(chosen.get("top_p")),
            )
        )

    models: list[ServedModel] = []
    for slug, claimants in candidates.items():
        if len(claimants) == 1:
            models.append(claimants[0])
            continue
        problems.append(
            ServedNameProblem(
                served_name=slug,
                library_ids=tuple(model.id for model in claimants),
                reason="duplicate",
                message=(
                    f"served name {slug!r} is claimed by "
                    + " and ".join(repr(model.id) for model in claimants)
                    + "; choose a different Served as value for one of them"
                ),
            )
        )
    return ServedCatalogue(models=tuple(models), problems=tuple(problems))


def served_models_from_library(
    reports: Iterable[Any],
    *,
    default_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[ServedModel, ...]:
    """The served catalogue, refusing outright if any name is unusable.

    The strict reading, for the boundary that is deciding whether to *store* a
    configuration. A save that produced an ambiguous name must fail while the
    user is still looking at the form; discovering it later, as a model that
    quietly stopped being served, is the failure this prevents.
    """
    catalogue = resolve_served_catalogue(
        reports, default_effort=default_effort, overrides=overrides
    )
    if catalogue.problems:
        raise ConfigError("; ".join(problem.message for problem in catalogue.problems))
    return catalogue.models


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_path(value: Any) -> str | None:
    """A blank is an absence, not a value.

    ``Path("")`` is the current directory, so an empty string surviving into
    the engine would ask MLX to load the daemon's working directory as an
    adapter. A hand-edited settings file is enough to produce one.
    """
    text = str(value).strip() if value is not None else ""
    return text or None
