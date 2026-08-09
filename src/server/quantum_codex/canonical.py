"""The canonical intermediate representation (D2).

This is the only currency that crosses the boundary between the Responses
protocol and Harmony/MLX:

    ResponsesRequest -> CanonicalTurn -> HarmonyConversation -> TokenizedPrompt
    HarmonyGeneration -> CanonicalTurnResult -> ResponsesEvent / ResponsesObject

No raw Responses dictionary reaches ``harmony`` or ``inference``. That rule is
what keeps protocol quirks — Codex dialect, version differences, field shapes —
inside ``api`` and ``codex``, instead of leaking into prompt rendering and
inference where they would be impossible to reason about later.

These types are deliberately about *meaning*, not about wire shape. If a future
Codex version renames a field, this file should not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Who authored a turn, in canonical terms."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"


class ReasoningEffort(StrEnum):
    """The reasoning levels GPT-OSS actually has.

    Not an open vocabulary: Harmony accepts exactly these three. Levels from
    other model families (``xhigh``, ``max``, …) do not exist here and must be
    rejected rather than silently mapped onto one of these.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CanonicalMessage:
    """One conversational turn, already normalised out of the wire format."""

    role: Role
    text: str


@dataclass(frozen=True)
class ReasoningTrace:
    """Chain-of-thought from an earlier assistant turn, replayed back to us.

    This is a first-class conversation item, not decoration. Dropping it between
    a tool call and its result is what breaks continuity: the model loses the
    thinking that led to the call and starts the task over.
    """

    text: str


@dataclass(frozen=True)
class ToolCall:
    """A function call the model made.

    ``arguments`` stays a raw JSON *string*, exactly as the Responses API
    carries it. Parsing and re-serialising here would silently reformat what the
    model produced, and the client is the one that has to interpret it.

    ``namespace`` is ``None`` for an ordinary top-level function. It is not the
    same as the function name, and inventing one would produce a call the client
    cannot route.
    """

    call_id: str
    name: str
    arguments: str
    namespace: str | None = None


@dataclass(frozen=True)
class ToolOutput:
    """The client's result for a previous :class:`ToolCall`.

    ``name`` and ``namespace`` are resolved from the matching call rather than
    carried on the wire: the Responses API identifies an output only by
    ``call_id``, but Harmony attributes a tool message to the tool that produced
    it, which means it needs both.

    Attributing a namespaced result to ``functions`` instead would show the model
    an answer from a tool it never called, which is exactly the kind of
    incoherence that makes the next turn restart the task.
    """

    call_id: str
    output: str
    name: str | None = None
    namespace: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """A function tool the client declared and can execute."""

    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None
    strict: bool = False


@dataclass(frozen=True)
class ToolNamespace:
    """A named group of tools the client declared and routes itself.

    A namespace is a group with its own identity, not a prefix on each member:
    it carries a description of its own, and the client dispatches on the
    namespace and the function name as *separate* values. Flattening it into
    ``namespace_name`` per tool would lose the description and produce calls the
    client rejects -- observed against Codex 0.147, which answers a flattened
    ``multi_agent_v1.close_agent`` with ``unsupported call``.
    """

    name: str
    tools: tuple[ToolDefinition, ...] = ()
    description: str | None = None


# One conversation item. The order of these in a turn is the conversation, and
# it is preserved end to end.
CanonicalItem = CanonicalMessage | ReasoningTrace | ToolCall | ToolOutput


@dataclass(frozen=True)
class CanonicalTurn:
    """Everything needed to render a prompt and run one generation.

    ``instructions`` is the developer-level instruction block. ``items`` is the
    conversation in order, including replayed reasoning and tool traffic.
    Sampling and limits live here rather than being read from request
    dictionaries deeper in the stack.

    ``tools`` holds top-level functions, which Harmony renders into its
    ``functions`` namespace. ``tool_namespaces`` holds client-routed groups and
    is kept separate rather than folded into one list, so "a top-level function
    has no namespace" is structural instead of a rule someone has to remember.
    """

    items: tuple[CanonicalItem, ...]
    instructions: str | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_namespaces: tuple[ToolNamespace, ...] = ()
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class Usage:
    """Token accounting.

    Every number here comes from real token counts — the tokenizer for input,
    the generation loop for output. No character-count estimation anywhere
    (cahier 21).

    ``reasoning_tokens`` is a subset of ``output_tokens``, matching how the
    Responses API reports it.
    """

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class GenerationTiming:
    """Measured, not estimated.

    ``prefill_seconds`` is time until the first generated token, which includes
    prompt evaluation. It is an upper bound on pure prefill, not a pure
    measurement of it — named so that nothing downstream mistakes it for one.
    """

    prefill_seconds: float
    decode_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds

    def decode_tokens_per_second(self, output_tokens: int) -> float:
        if self.decode_seconds <= 0 or output_tokens <= 0:
            return 0.0
        return output_tokens / self.decode_seconds

    def prefill_tokens_per_second(self, input_tokens: int) -> float:
        if self.prefill_seconds <= 0 or input_tokens <= 0:
            return 0.0
        return input_tokens / self.prefill_seconds


@dataclass(frozen=True)
class CanonicalTurnResult:
    """What one generation produced, before it becomes a Responses object.

    ``reasoning`` holds the Harmony ``analysis`` channel content. It is kept
    separate from ``text`` (the ``final`` channel) all the way out, because
    conflating them is how chain-of-thought ends up leaking into user-visible
    output or being lost across tool turns.
    """

    text: str
    reasoning: tuple[str, ...] = field(default_factory=tuple)
    # At most one: Harmony's `<|call|>` ends the assistant turn. A tuple rather
    # than a single value so the shape does not have to change if a future
    # Harmony release ever emits more, but the capability stays declared as one.
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))
    finish_reason: FinishReason = FinishReason.STOP
    timing: GenerationTiming | None = None
