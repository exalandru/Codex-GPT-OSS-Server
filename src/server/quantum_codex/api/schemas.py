"""Responses API request shapes, and their normalisation into the IR.

This module knows about the wire format. It is where protocol shape stops: what
leaves here is a :class:`CanonicalTurn` (D2).

Unknown fields are collected rather than ignored, so that the compatibility
layer can fail closed on them. Pydantic's own ``extra="forbid"`` is not used
because it would produce Pydantic's error shape instead of the OpenAI envelope
clients expect.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..canonical import (
    CanonicalItem,
    CanonicalMessage,
    CanonicalTurn,
    ReasoningEffort,
    ReasoningTrace,
    Role,
    ToolCall,
    ToolDefinition,
    ToolNamespace,
    ToolOutput,
)
from .errors import invalid_request

logger = logging.getLogger(__name__)

# Content part types. Input and output parts are named differently by the
# Responses API even when both are plain text; a replayed assistant turn uses
# `output_text` where a user turn uses `input_text`.
_TEXT_PART_TYPES = frozenset({"input_text", "output_text", "text"})

_ROLES: dict[str, Role] = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "system": Role.SYSTEM,
    "developer": Role.DEVELOPER,
}


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    effort: Literal["low", "medium", "high"] | None = None
    summary: str | None = None
    context: Any | None = None


class ResponsesRequest(BaseModel):
    """A ``POST /v1/responses`` body.

    The declared set covers what Codex 0.147 actually sends
    (``codex-rs/codex-api/src/common.rs``, ``ResponsesApiRequest``) plus the
    common SDK fields. Declaring a field here is not a claim that its behaviour
    exists -- that judgement belongs to the capability layer. It only means the
    field is known, so it does not trip the fail-closed check on unknown fields.

    Anything genuinely undeclared lands in ``model_extra`` and becomes a 400.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    stream: bool = False
    reasoning: ReasoningConfig | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    store: bool = True

    # Sent by Codex on every turn. Each is checked by the capability layer, not
    # here: `tools` gets bucketed, `include` and `tool_choice` and `text` are
    # validated against what this server can really do.
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    include: list[str] | None = None
    text: dict[str, Any] | None = None

    # Accepted and inert. Accepting them is not a claim that they do anything:
    # `prompt_cache_key` in particular is a hint with no cache behind it yet,
    # and reporting a cache hit for it would be a fabricated number.
    stream_options: dict[str, Any] | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    client_metadata: dict[str, Any] | None = None
    previous_response_id: str | None = None

    @property
    def unknown_fields(self) -> dict[str, Any]:
        return self.model_extra or {}


def _param_path(location: tuple[Any, ...]) -> str | None:
    """Render a Pydantic error location as a Responses API `param`."""
    if not location:
        return None
    path = str(location[0])
    for part in location[1:]:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def parse_request(body: dict[str, Any]) -> ResponsesRequest:
    """Validate a request body into the request model.

    Pydantic's ``ValidationError`` is translated here rather than allowed to
    escape. Left alone it becomes a 500, which tells a client nothing about what
    it sent wrong -- a schema mismatch is the client's error and must come back
    as one, in the envelope clients actually parse.
    """
    try:
        return ResponsesRequest.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise invalid_request(first.get("msg", "Invalid request."), param=_param_path(first["loc"])) from exc


def _part_text(part: Any, *, param: str) -> str:
    """Extract text from one content part, refusing anything non-textual."""
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        raise invalid_request("Content parts must be objects.", param=param)

    part_type = part.get("type")
    if part_type not in _TEXT_PART_TYPES:
        # Images, audio and files are not supported. Skipping them silently
        # would drop what the user actually sent, so this fails closed.
        raise invalid_request(
            f"Unsupported content part type: {part_type}. Only text is supported.",
            param=f"{param}.type",
        )

    text = part.get("text")
    if not isinstance(text, str):
        raise invalid_request("Text content parts require a string `text`.", param=f"{param}.text")
    return text


def _item_text(content: Any, *, param: str) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            _part_text(part, param=f"{param}[{index}]") for index, part in enumerate(content)
        )
    raise invalid_request("`content` must be a string or a list of content parts.", param=param)


def _reasoning_text(item: dict[str, Any], *, param: str) -> str:
    """Pull replayed chain-of-thought out of a reasoning item.

    Codex serialises `reasoning_text` content back to us (its
    `should_serialize_reasoning_content` keeps the field precisely when a
    `ReasoningText` part is present), which is what makes continuity across tool
    turns possible without encryption.

    An item carrying only `encrypted_content` cannot be interpreted here. It is
    skipped rather than rejected -- refusing it would break the turn outright,
    whereas skipping degrades continuity and says so.
    """
    parts = item.get("content")
    if not isinstance(parts, list):
        return ""

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("reasoning_text", "text"):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)

    if not texts and item.get("encrypted_content"):
        logger.warning(
            "Reasoning item at %s carries only `encrypted_content`; this server cannot "
            "decrypt it, so the chain of thought for that turn is lost.",
            param,
        )
    return "".join(texts)


def _tool_output_text(output: Any, *, param: str) -> str:
    """A function call result, as text.

    The Responses API allows either a plain string or a list of content items.
    Both reach the model as text, because Harmony tool results are text.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        texts: list[str] = []
        for part in output:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "".join(texts)
    if isinstance(output, dict):
        # Some clients wrap the payload; take its text rather than stringifying
        # a dict into the prompt.
        text = output.get("output") or output.get("text")
        if isinstance(text, str):
            return text
    raise invalid_request(
        "`function_call_output.output` must be a string or a list of content parts.",
        param=param,
    )


def _items_from_input(request: ResponsesRequest) -> tuple[CanonicalItem, ...]:
    if request.input is None:
        raise invalid_request("`input` is required.", param="input")

    if isinstance(request.input, str):
        return (CanonicalMessage(role=Role.USER, text=request.input),)

    items: list[CanonicalItem] = []
    # The wire identifies an output only by `call_id`, but Harmony attributes a
    # tool message to the tool that produced it -- which means both the function
    # name and its namespace. Resolve them from the call that came earlier in the
    # same conversation.
    call_targets: dict[str, tuple[str, str | None]] = {}

    for index, item in enumerate(request.input):
        param = f"input[{index}]"
        if not isinstance(item, dict):
            raise invalid_request("Each input item must be an object.", param=param)

        # `type` is optional for message items: Codex and the SDKs both send
        # bare `{role, content}` objects.
        item_type = item.get("type", "message")

        if item_type == "message":
            role_name = item.get("role")
            role = _ROLES.get(role_name) if isinstance(role_name, str) else None
            if role is None:
                raise invalid_request(f"Unsupported role: {role_name}", param=f"{param}.role")
            items.append(
                CanonicalMessage(
                    role=role, text=_item_text(item.get("content"), param=f"{param}.content")
                )
            )

        elif item_type == "reasoning":
            text = _reasoning_text(item, param=param)
            if text:
                items.append(ReasoningTrace(text=text))

        elif item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise invalid_request(
                    "`function_call` requires string `call_id` and `name`.", param=param
                )
            arguments = item.get("arguments")
            raw_namespace = item.get("namespace")
            namespace = raw_namespace if isinstance(raw_namespace, str) and raw_namespace else None
            call_targets[call_id] = (name, namespace)
            items.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments if isinstance(arguments, str) else "{}",
                    namespace=namespace,
                )
            )

        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise invalid_request(
                    "`function_call_output` requires a string `call_id`.", param=param
                )
            name, namespace = call_targets.get(call_id, (None, None))
            items.append(
                ToolOutput(
                    call_id=call_id,
                    output=_tool_output_text(item.get("output"), param=f"{param}.output"),
                    name=name,
                    namespace=namespace,
                )
            )

        else:
            raise invalid_request(
                f"Unsupported input item type: {item_type}", param=f"{param}.type"
            )

    if not items:
        raise invalid_request("`input` must contain at least one item.", param="input")

    return tuple(items)


def _definition_from_spec(spec: Mapping[str, Any], *, param: str) -> ToolDefinition:
    """One function tool, from either the flat or nested wire shape.

    Both the flat Responses shape (`{type, name, parameters}`) and the nested
    Chat Completions shape (`{type, function: {...}}`) are accepted, because
    SDKs differ and the difference is not semantic.
    """
    inner = spec.get("function")
    body: Mapping[str, Any] = inner if isinstance(inner, dict) else spec

    name = body.get("name")
    if not isinstance(name, str) or not name:
        raise invalid_request("Each function tool requires a `name`.", param=param)

    description = body.get("description")
    parameters = body.get("parameters")
    return ToolDefinition(
        name=name,
        description=description if isinstance(description, str) else "",
        parameters=parameters if isinstance(parameters, dict) else None,
        strict=bool(body.get("strict")),
    )


def _tools_from_request(
    request: ResponsesRequest,
) -> tuple[tuple[ToolDefinition, ...], tuple[ToolNamespace, ...]]:
    """Normalise declared tools into top-level functions and namespaces.

    Order is preserved within each group. The capability layer has already
    refused anything malformed, so this reads what it validated rather than
    re-deciding it.
    """
    if not request.tools:
        return ((), ())

    definitions: list[ToolDefinition] = []
    namespaces: list[ToolNamespace] = []

    for index, tool in enumerate(request.tools):
        if not isinstance(tool, dict):
            continue  # already bucketed by the capability layer

        tool_type = tool.get("type")
        if tool_type == "function":
            definitions.append(_definition_from_spec(tool, param=f"tools[{index}]"))
        elif tool_type == "namespace":
            members = tool.get("tools")
            description = tool.get("description")
            namespaces.append(
                ToolNamespace(
                    name=str(tool.get("name")),
                    tools=tuple(
                        _definition_from_spec(member, param=f"tools[{index}].tools[{position}]")
                        for position, member in enumerate(members)
                        if isinstance(member, dict)
                    ),
                    description=description if isinstance(description, str) else None,
                )
            )

    return tuple(definitions), tuple(namespaces)


def to_canonical_turn(
    request: ResponsesRequest, *, default_effort: ReasoningEffort
) -> CanonicalTurn:
    """Normalise a validated request into the IR.

    After this call, nothing downstream needs the request object.
    """
    effort = default_effort
    if request.reasoning is not None and request.reasoning.effort is not None:
        effort = ReasoningEffort(request.reasoning.effort)

    tools, namespaces = _tools_from_request(request)
    return CanonicalTurn(
        items=_items_from_input(request),
        instructions=request.instructions,
        tools=tools,
        tool_namespaces=namespaces,
        reasoning_effort=effort,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
