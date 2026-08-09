"""What this server can actually do on ``/v1/responses`` (D4).

This module answers exactly one question: *can this server really do X?* The
request schema answers a different one: *is this payload well formed?* Keeping
them apart matters, because a payload can be perfectly valid and still ask for
something that does not exist here.

Three categories, and they are not interchangeable:

``SUPPORTED``
    Really implemented. The model sees it and the server can honour it.
``ACCEPTED``
    Received, checked, then deliberately ignored. Accepting a field is not a
    claim that its behaviour exists.
``OPTIONAL_UNSUPPORTED``
    Declared by the client but executed by the provider. This server cannot run
    them, so they are removed before the model could call them.

Anything in none of those buckets is unknown and fails closed with a 400, so a
client evolution surfaces as a clear error instead of silently changed
behaviour.

This module is a leaf: it never raises, and returns :class:`CompatProblem` for
the API layer to turn into an error response.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompatProblem:
    """A capability refusal, ready to become a 400."""

    message: str
    param: str | None = None


class ToolSupport(StrEnum):
    SUPPORTED = "supported"
    OPTIONAL_UNSUPPORTED = "optional_unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DroppedTool:
    """A client tool removed before the prompt was rendered."""

    index: int
    type: str
    name: str | None = None

    @property
    def identity(self) -> tuple[str, str | None]:
        return (self.type, self.name)

    def __str__(self) -> str:
        return f"type={self.type} name={self.name}"


@dataclass(frozen=True)
class ToolPlan:
    """The outcome of bucketing a request's ``tools``.

    ``kept`` carries each tool's *original* index. Re-indexing after a drop
    would make every later error ``param`` point at the wrong tool.
    """

    kept: tuple[tuple[int, dict[str, Any]], ...] = ()
    dropped: tuple[DroppedTool, ...] = ()
    problem: CompatProblem | None = None


# Codex asks for encrypted reasoning on every turn. This server is fully local
# and produces no `encrypted_content`, so the value is accepted to keep the
# client working, and then ignored. Accepting it is not supporting it.
ACCEPTED_INCLUDE_VALUES = frozenset({"reasoning.encrypted_content"})

# `namespace` is client-executed and Codex routes it. Harmony declares it as a
# real namespace in the developer block, and a call comes back with `namespace`
# and `name` as separate fields -- verified against Codex 0.147, whose router
# dispatches that shape and answers a flattened `ns.name` with `unsupported
# call`.
SUPPORTED_TOOL_TYPES = frozenset({"function", "namespace"})

# Known, deliberately not exposed to the model in this build.
#
# `web_search*` is provider-executed. This server has no search executor, and
# the client never expected it to run one.
#
# `local_shell` is deliberately absent: Codex executes it itself, so dropping it
# would remove a real capability while reporting nothing.
OPTIONAL_TOOL_TYPES = frozenset(
    {
        "web_search",
        "web_search_preview",
        "web_search_preview_2025_03_11",
    }
)

# Harmony renders top-level functions into a namespace of this name. A client
# namespace claiming it would collide with them in Harmony's namespace map --
# one silently replacing the other, so tools the client declared would vanish
# from the prompt while still being routable. Refused rather than merged.
RESERVED_NAMESPACE = "functions"

SUPPORTED_INPUT_ITEM_TYPES = frozenset(
    {"message", "reasoning", "function_call", "function_call_output"}
)


@dataclass(frozen=True)
class ResponsesCapabilities:
    """The capability surface of ``/v1/responses``."""

    # Harmony's `<|call|>` is an assistant stop token: the turn ends so the tool
    # result can come back. One call per turn is correct Harmony semantics, not
    # a limitation to work around.
    parallel_tool_calls: bool = False

    # Chain-of-thought is exposed as plain `reasoning` items and replayed into
    # the Harmony `analysis` channel. OpenAI's encrypted form is not produced,
    # and an encrypted-only item coming back cannot be interpreted.
    plaintext_reasoning: bool = True
    encrypted_reasoning: bool = False

    function_tools: bool = True
    namespace_tools: bool = True
    prompt_cache: bool = False
    structured_output: bool = False
    forced_tool_choice: bool = False

    supported_tool_types: frozenset[str] = SUPPORTED_TOOL_TYPES
    optional_tool_types: frozenset[str] = OPTIONAL_TOOL_TYPES
    supported_input_item_types: frozenset[str] = SUPPORTED_INPUT_ITEM_TYPES
    supported_tool_choices: frozenset[str] = frozenset({"auto", "none"})
    supported_text_formats: frozenset[str] = frozenset({"text"})
    accepted_include_values: frozenset[str] = ACCEPTED_INCLUDE_VALUES

    # -- resolution ---------------------------------------------------------

    def resolve_parallel_tool_calls(self, requested: bool | None) -> bool:
        """Report the capability delivered, not the one requested.

        ``false`` matches reality exactly. ``true`` means "you *may* parallelise",
        not "you must", so it is honoured sequentially rather than rejected --
        and the response says what actually happened.
        """
        return bool(requested) and self.parallel_tool_calls

    # -- request fields -----------------------------------------------------

    def check_include(self, include: Sequence[str] | None) -> CompatProblem | None:
        if not include:
            return None
        for index, value in enumerate(include):
            if value not in self.accepted_include_values:
                accepted = ", ".join(sorted(self.accepted_include_values))
                return CompatProblem(
                    f"Unsupported `include` value: {value}. Accepted values: {accepted}.",
                    param=f"include[{index}]",
                )
        return None

    def check_tool_choice(self, tool_choice: Any) -> CompatProblem | None:
        # A named/forced choice arrives as an object, which is unhashable: test
        # the type before the membership check.
        if tool_choice is None or (
            isinstance(tool_choice, str) and tool_choice in self.supported_tool_choices
        ):
            return None
        supported = ", ".join(f"`{choice}`" for choice in sorted(self.supported_tool_choices))
        return CompatProblem(
            f"Only `tool_choice` values {supported} are supported.", param="tool_choice"
        )

    def check_text_format(self, text_config: Mapping[str, Any] | None) -> CompatProblem | None:
        if not text_config:
            return None
        text_format = text_config.get("format")
        if text_format is None:
            return None
        format_type = text_format.get("type") if isinstance(text_format, Mapping) else None
        if format_type in self.supported_text_formats:
            return None
        return CompatProblem("Only plain text responses are supported.", param="text.format")

    # -- tools --------------------------------------------------------------

    def classify_tool_type(self, tool_type: str | None) -> ToolSupport:
        if tool_type in self.supported_tool_types:
            return ToolSupport.SUPPORTED
        if tool_type in self.optional_tool_types:
            return ToolSupport.OPTIONAL_UNSUPPORTED
        return ToolSupport.UNKNOWN

    def check_namespace_tool(
        self, tool: Mapping[str, Any], *, index: int, claimed: Sequence[str]
    ) -> CompatProblem | None:
        """Structural checks a namespace must pass before the model sees it.

        Every failure here would otherwise reach the model as a namespace it can
        address but nothing can route, so each one fails closed.
        """
        param = f"tools[{index}]"
        if not self.namespace_tools:
            return CompatProblem(
                "This build cannot expose namespaced tools to the model.", param=param
            )

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            return CompatProblem(
                "A `namespace` tool requires a non-empty string `name`.", param=f"{param}.name"
            )
        if name == RESERVED_NAMESPACE:
            return CompatProblem(
                f"`{RESERVED_NAMESPACE}` is reserved for top-level function tools and cannot "
                f"be used as a namespace name.",
                param=f"{param}.name",
            )
        if name in claimed:
            return CompatProblem(
                f"Namespace `{name}` is declared more than once.", param=f"{param}.name"
            )

        inner = tool.get("tools")
        if not isinstance(inner, list):
            return CompatProblem(
                "A `namespace` tool requires a `tools` array.", param=f"{param}.tools"
            )

        for position, member in enumerate(inner):
            member_param = f"{param}.tools[{position}]"
            if not isinstance(member, dict):
                return CompatProblem("Each namespaced tool must be an object.", param=member_param)
            member_type = member.get("type", "function")
            if member_type != "function":
                # A nested namespace, or a provider-executed tool inside a
                # client namespace, is not something this server can render or
                # route. Refusing names the exact position rather than dropping
                # a member out of a group the client believes is intact.
                return CompatProblem(
                    f"Unsupported tool type inside namespace `{name}`: {member_type}",
                    param=f"{member_param}.type",
                )
            member_name = member.get("name")
            if not isinstance(member_name, str) or not member_name:
                return CompatProblem(
                    "Each namespaced tool requires a non-empty string `name`.",
                    param=f"{member_param}.name",
                )
        return None

    def plan_tools(self, tools: Sequence[Any]) -> ToolPlan:
        """Bucket a request's tools, preserving original indices."""
        kept: list[tuple[int, dict[str, Any]]] = []
        dropped: list[DroppedTool] = []
        namespaces: list[str] = []

        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                return ToolPlan(
                    problem=CompatProblem("Each tool must be an object.", param=f"tools[{index}]")
                )

            raw_type = tool.get("type")
            tool_type = raw_type if isinstance(raw_type, str) else None
            support = self.classify_tool_type(tool_type)

            if support is ToolSupport.SUPPORTED and tool_type == "namespace":
                problem = self.check_namespace_tool(tool, index=index, claimed=namespaces)
                if problem is not None:
                    return ToolPlan(problem=problem)
                namespaces.append(str(tool.get("name")))
                kept.append((index, tool))
            elif support is ToolSupport.SUPPORTED:
                if not self.function_tools:
                    # Name the tool: "tools are unsupported" alone leaves the
                    # operator guessing which of a dozen declared tools tripped
                    # it, and which client setting would stop it being sent.
                    name = tool.get("name")
                    described = f"`{name}` ({tool_type})" if name else f"`{tool_type}`"
                    return ToolPlan(
                        problem=CompatProblem(
                            f"Tool {described} was declared, but this build cannot route tool "
                            f"calls back to the client, so no tool may be exposed to the model.",
                            param=f"tools[{index}]",
                        )
                    )
                kept.append((index, tool))
            elif support is ToolSupport.OPTIONAL_UNSUPPORTED:
                name = tool.get("name")
                dropped.append(
                    DroppedTool(
                        index=index,
                        type=str(tool_type),
                        name=name if isinstance(name, str) and name else None,
                    )
                )
            else:
                return ToolPlan(
                    problem=CompatProblem(
                        f"Unsupported tool type: {raw_type}", param=f"tools[{index}].type"
                    )
                )

        return ToolPlan(kept=tuple(kept), dropped=tuple(dropped))


CAPABILITIES = ResponsesCapabilities()


# -- dropped-tool logging ---------------------------------------------------

_MAX_TRACKED_DROPPED_TOOLS = 256
_seen_dropped_tools: set[tuple[str, str | None]] = set()


def log_dropped_tools(dropped: Sequence[DroppedTool], *, request_id: str | None = None) -> None:
    """Announce each distinct dropped tool once at INFO; repeats go to DEBUG.

    Codex resends the same tool list every turn, so unconditional per-request
    logging would be pure noise and would bury everything else.
    """
    if not dropped:
        return

    newly_seen = False
    for tool in dropped:
        if tool.identity in _seen_dropped_tools:
            continue
        if len(_seen_dropped_tools) >= _MAX_TRACKED_DROPPED_TOOLS:
            continue  # tracking table full: stay quiet rather than spam
        _seen_dropped_tools.add(tool.identity)
        newly_seen = True

    emit = logger.info if newly_seen else logger.debug
    emit(
        "Dropping provider-executed tool(s) this server cannot run rid=%s: %s",
        request_id or "-",
        "; ".join(sorted({str(tool) for tool in dropped})),
    )
