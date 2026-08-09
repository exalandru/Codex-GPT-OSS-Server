"""Building Responses objects out of the IR.

The Responses API is a state machine over output items, not a single blob of
text. Reasoning and the final answer are *separate items*: keeping them apart
here is what lets a client replay chain-of-thought across tool turns instead of
receiving one flattened string.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from ..canonical import CanonicalTurnResult, FinishReason
from .sse import function_call_item, message_item, reasoning_item


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def build_output_items(result: CanonicalTurnResult) -> list[dict[str, Any]]:
    """Reasoning first, then any tool call or message. Order is the turn's order."""
    items: list[dict[str, Any]] = []

    for reasoning in result.reasoning:
        items.append(reasoning_item(_item_id("rs"), reasoning))

    for call in result.tool_calls:
        items.append(
            function_call_item(
                _item_id("fc"),
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                namespace=call.namespace,
            )
        )

    if result.text:
        items.append(message_item(_item_id("msg"), result.text))

    return items


def build_usage(result: CanonicalTurnResult) -> dict[str, Any]:
    usage = result.usage
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {"cached_tokens": usage.cached_tokens},
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        "total_tokens": usage.total_tokens,
    }


def build_response(
    *,
    response_id: str,
    model: str,
    result: CanonicalTurnResult,
    instructions: str | None,
    tools: list[dict[str, Any]] | None = None,
    created_at: int | None = None,
) -> dict[str, Any]:
    """A completed ``response`` object.

    ``tools`` echoes only what was actually forwarded to the model, so a client
    can see when something it declared was dropped instead of having to guess.
    """
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at if created_at is not None else int(time.time()),
        "status": _status(result.finish_reason),
        "error": None,
        "incomplete_details": _incomplete_details(result.finish_reason),
        "instructions": instructions,
        "model": model,
        "output": build_output_items(result),
        # Reported as the capability actually delivered, not as the value that
        # was requested: Harmony ends the turn at the first call.
        "parallel_tool_calls": False,
        "tools": tools or [],
        "tool_choice": "auto",
        "usage": build_usage(result),
        "metadata": {},
    }


def _status(finish_reason: FinishReason) -> str:
    if finish_reason is FinishReason.LENGTH:
        return "incomplete"
    if finish_reason is FinishReason.CANCELLED:
        return "cancelled"
    # A turn that ended at a tool call is `completed`, not pending: the model
    # finished its turn and handed control to the client, which is exactly what
    # Harmony's `<|call|>` means.
    return "completed"


def _incomplete_details(finish_reason: FinishReason) -> dict[str, Any] | None:
    if finish_reason is FinishReason.LENGTH:
        return {"reason": "max_output_tokens"}
    return None
