"""What a server profile contains, described once.

The desktop configuration form is generated from this. It exists so that adding
a setting means editing one file rather than three: the dataclass, a Rust
struct, and a TypeScript form. A form that encoded its own idea of which values
are valid would disagree with the server the first time a bound changed, and the
disagreement would surface as a request the UI allowed and the server refused.

Fields are grouped the way the cahier (29) asks, and the grouping is a claim
about *who should be changing them*:

``basic``     what a normal setup needs
``advanced``  worth tuning once the basics work
``expert``    changes behaviour in ways that need understanding

Nothing is listed here that the server does not actually honour. A setting shown
in a form and ignored at runtime is worse than an absent one: the user believes
they configured something.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical import ReasoningEffort
from .config import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HOST,
    DEFAULT_IDLE_TIMEOUT_MINUTES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_PORT,
    MAX_IDLE_TIMEOUT_MINUTES,
)
from .inference.prompt_cache import DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES
from .models import SERVED_NAME, SERVED_NAME_HELP

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Field:
    """One setting, and everything a form needs to render and explain it."""

    name: str
    label: str
    #: ``string`` | ``integer`` | ``number`` | ``choice`` | ``path``
    kind: str
    group: str
    help: str
    default: Any = None
    choices: list[str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    #: Editable only when the server is stopped, because it decides what is
    #: loaded or where it listens.
    restart_required: bool = False
    #: Wants a warning label rather than a plain input (cahier 29).
    caution: str | None = None
    required: bool = False
    #: ``None`` means "inherit", which is not the same as a value.
    nullable: bool = False
    #: A regular expression a string value must match, with the message to show
    #: when it does not. Declared beside the field rather than checked by each
    #: caller: a name that leaves this module has already been through it.
    pattern: str | None = None
    pattern_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


FIELDS: list[Field] = [
    # -- basic ---------------------------------------------------------------
    Field(
        name="model",
        label="Default model",
        # A choice over installed models, not a path. A filesystem path is the
        # model library's business: it changes when a volume is remounted or a
        # directory is renamed, and a profile that stored one would silently
        # stop matching. The identity a profile records is the stable slug.
        kind="choice",
        group="basic",
        choices=[],  # filled from the library; see `schema()`
        help=(
            "Optional. Loaded once the server is answering, so a session need not wait "
            "for it. Leave empty to load nothing: Codex picks a model per session and "
            "it is loaded on demand."
        ),
        required=False,
        nullable=True,
        restart_required=True,
    ),
    Field(
        name="port",
        label="Port",
        kind="integer",
        group="basic",
        help="Where the server listens on this machine.",
        default=DEFAULT_PORT,
        minimum=1,
        maximum=65535,
        restart_required=True,
    ),
    # -- advanced ------------------------------------------------------------
    Field(
        name="model_idle_timeout_minutes",
        label="Unload idle model after",
        kind="integer",
        group="advanced",
        # Says what it releases and what it does not: the complaint this answers
        # is a user seeing "no model loaded" and reading it as the server having
        # died.
        help=(
            "Unload the model after this many minutes without inference activity. "
            "0 disables automatic unload. The server remains running, and the next "
            "request loads the model again."
        ),
        default=DEFAULT_IDLE_TIMEOUT_MINUTES,
        minimum=0,
        maximum=MAX_IDLE_TIMEOUT_MINUTES,
        unit="minutes",
        # The supervisor reads this once, when it is constructed. Reloading it
        # into a running daemon would need a configuration-watching mechanism
        # that does not exist, and claiming a live effect it does not have is
        # the failure mode this flag exists to prevent.
        restart_required=True,
    ),
    Field(
        name="cache_max_entries",
        label="Prompt cache sessions",
        kind="integer",
        group="advanced",
        help=(
            "How many conversations keep a reusable prefix. Zero disables reuse, "
            "which makes every turn re-evaluate the whole conversation."
        ),
        default=DEFAULT_MAX_ENTRIES,
        minimum=0,
        maximum=64,
        restart_required=True,
    ),
    Field(
        name="cache_max_bytes",
        label="Prompt cache budget",
        kind="integer",
        group="advanced",
        help=(
            "Memory the prompt cache may hold. It shares unified memory with the "
            "model, so a large budget costs what the model would have used."
        ),
        default=DEFAULT_MAX_BYTES,
        minimum=0,
        unit="bytes",
        restart_required=True,
    ),
    Field(
        name="host",
        label="Bind address",
        kind="string",
        group="advanced",
        help="Which interface to listen on.",
        default=DEFAULT_HOST,
        restart_required=True,
        caution=(
            "Anything other than 127.0.0.1 exposes this server to your network. "
            "It has no authentication of its own."
        ),
    ),
    # -- expert --------------------------------------------------------------
    Field(
        name="log_level",
        label="Log level",
        kind="choice",
        group="expert",
        help="How much the server writes. DEBUG includes cache and tool decisions.",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        restart_required=True,
    ),
]

#: Settings that belong to a *model* rather than to the daemon.
#:
#: Each of these can legitimately differ between the 20B and the 120B: an alias
#: names one set of weights, and a reasoning effort or output budget that suits
#: one does not suit the other. Held here so the model form and the profile form
#: are generated from one description, and stored per model by
#: `quantum_codex.model_settings`.
MODEL_FIELDS: list[Field] = [
    Field(
        name="display_name",
        label="Display name",
        kind="string",
        group="basic",
        help="The name shown in QCS. It does not change model identity or what Codex requests.",
        nullable=True,
    ),
    Field(
        name="served_model_name",
        label="Served as",
        kind="string",
        group="basic",
        help=(
            "The model name exposed to Codex. It is separate from the immutable library id."
        ),
        nullable=True,
        pattern=SERVED_NAME.pattern,
        pattern_message=f"Served as must be {SERVED_NAME_HELP}",
        restart_required=True,
    ),
    Field(
        name="reasoning_effort",
        label="Reasoning effort",
        kind="choice",
        group="basic",
        help=(
            "How much the model thinks before answering, when a request does not "
            "ask for something else. GPT-OSS has exactly these three levels."
        ),
        default=ReasoningEffort.MEDIUM.value,
        choices=[effort.value for effort in ReasoningEffort],
    ),
    Field(
        name="context_length",
        label="Context length",
        kind="integer",
        group="basic",
        help=(
            "Maximum tokens the KV cache will hold. A prompt that would exceed it "
            "is refused rather than silently truncated."
        ),
        default=DEFAULT_CONTEXT_LENGTH,
        minimum=2048,
        maximum=131072,
        unit="tokens",
        restart_required=True,
    ),
    Field(
        name="max_output_tokens",
        label="Maximum output",
        kind="integer",
        group="advanced",
        help=(
            "Cap on a single answer when the request does not set one. The real "
            "limit is whatever remains of the context window."
        ),
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        minimum=1,
        unit="tokens",
    ),
    Field(
        name="temperature",
        label="Temperature",
        kind="number",
        group="expert",
        help=(
            "Sampling temperature used when a request does not set one. Leave "
            "empty to inherit the model's own default."
        ),
        minimum=0.0,
        maximum=2.0,
        nullable=True,
        caution="GPT-OSS reasoning is sensitive to sampling changes.",
    ),
    Field(
        name="top_p",
        label="Top-p",
        kind="number",
        group="expert",
        help="Nucleus sampling threshold used when a request does not set one.",
        minimum=0.0,
        maximum=1.0,
        nullable=True,
        caution="GPT-OSS reasoning is sensitive to sampling changes.",
    ),
]

GROUPS: list[dict[str, str]] = [
    {
        "id": "basic",
        "label": "Basic",
        "help": "What a working setup needs.",
    },
    {
        "id": "advanced",
        "label": "Advanced",
        "help": "Worth tuning once the basics work.",
    },
    {
        "id": "expert",
        "label": "Expert",
        "help": "Changes behaviour in ways that need understanding first.",
    },
]


#: What the model field offers when nothing is chosen. Empty string rather than
#: null, because a form's select needs a value it can round-trip.
NO_DEFAULT_MODEL = ""


def schema(
    installed: Sequence[str] | None = None,
    *,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The description a form is generated from.

    ``installed`` is the set of stable library ids the library currently holds.
    It is passed in rather than read here so this module stays a leaf: it
    describes settings, and the library decides what exists.

    ``labels`` names those ids for a human. The *value* stays the id -- that is
    what the profile stores and what must survive a rename -- while the label is
    free to change with the model's display and served names. A form showing the
    raw id would be showing an internal identifier to someone who chose a name.
    """
    fields = []
    for field in FIELDS:
        described = field.as_dict()
        if field.name == "model":
            # Rebuilt on every call: a model imported a moment ago must appear
            # in the form without restarting anything.
            described["choices"] = [NO_DEFAULT_MODEL, *(installed or ())]
            described["choice_labels"] = {
                NO_DEFAULT_MODEL: "None — load on demand",
                **{
                    model_id: label
                    for model_id, label in (labels or {}).items()
                    if model_id in (installed or ())
                },
            }
        fields.append(described)
    return {"version": SCHEMA_VERSION, "groups": GROUPS, "fields": fields}


def field_names() -> set[str]:
    return {item.name for item in FIELDS}


@dataclass
class ValidationProblem:
    """A rejected value, attributed to the field that caused it."""

    field: str
    message: str


def coerce(name: str, raw: Any) -> Any:
    """Turn a form value into the type the profile stores.

    Text inputs deliver strings for everything. Doing this here rather than in
    the form is the same rule as the rest of this file: the server decides what
    its own settings mean.
    """
    return coerce_in(FIELDS, name, raw)


def coerce_in(fields: Sequence[Field], name: str, raw: Any) -> Any:
    """``coerce``, against a named field set.

    The profile form and the model form use the same rules; only the set of
    settings differs.
    """
    definition = next((item for item in fields if item.name == name), None)
    if definition is None:
        raise ValueError(f"unknown setting: {name}")

    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        if definition.nullable:
            return None
        raise ValueError(f"{definition.label} cannot be empty")

    if definition.kind == "integer":
        return int(raw)
    if definition.kind == "number":
        return float(raw)
    return str(raw).strip()


def validate(values: dict[str, Any]) -> list[ValidationProblem]:
    """Check values against the declared bounds.

    Returns every problem rather than the first, so a form can mark all the
    offending fields at once instead of making the user resubmit repeatedly.
    """
    return validate_in(FIELDS, values)


def validate_in(fields: Sequence[Field], values: dict[str, Any]) -> list[ValidationProblem]:
    """``validate``, against a named field set."""
    problems: list[ValidationProblem] = []
    for definition in fields:
        if definition.name not in values:
            continue
        value = values[definition.name]

        if value is None:
            if not definition.nullable:
                problems.append(
                    ValidationProblem(definition.name, f"{definition.label} is required")
                )
            continue

        if definition.choices and value not in definition.choices:
            problems.append(
                ValidationProblem(
                    definition.name,
                    f"{definition.label} must be one of: {', '.join(definition.choices)}",
                )
            )
            continue

        if definition.pattern is not None and not re.match(definition.pattern, str(value)):
            problems.append(
                ValidationProblem(
                    definition.name,
                    definition.pattern_message
                    or f"{definition.label} has an unusable value",
                )
            )
            continue

        if definition.kind in ("integer", "number"):
            if definition.minimum is not None and value < definition.minimum:
                problems.append(
                    ValidationProblem(
                        definition.name,
                        f"{definition.label} must be at least {definition.minimum:g}",
                    )
                )
            if definition.maximum is not None and value > definition.maximum:
                problems.append(
                    ValidationProblem(
                        definition.name,
                        f"{definition.label} must be at most {definition.maximum:g}",
                    )
                )

    return problems


# -- per-model settings -------------------------------------------------------


def model_schema() -> dict[str, Any]:
    """The description the per-model form is generated from.

    Same shape as :func:`schema`, so one form component renders both. Groups are
    reused rather than reinvented: "basic" means the same thing in either.
    """
    return {
        "version": SCHEMA_VERSION,
        "groups": GROUPS,
        "fields": [field.as_dict() for field in MODEL_FIELDS],
    }


def model_field_names() -> set[str]:
    return {item.name for item in MODEL_FIELDS}


def coerce_model(name: str, raw: Any) -> Any:
    return coerce_in(MODEL_FIELDS, name, raw)


def validate_model(values: dict[str, Any]) -> list[ValidationProblem]:
    return validate_in(MODEL_FIELDS, values)
