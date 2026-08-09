"""Responses API server-sent events.

Event names and payload requirements come from the Codex 0.147.0 source,
``codex-rs/codex-api/src/sse/responses.rs`` (``process_responses_event``).
Codex acts on exactly these:

===============================  ====================================
event                            what Codex needs on it
===============================  ====================================
``response.created``             a ``response`` object (presence only)
``response.output_item.added``   ``item``
``response.reasoning_text.delta``  ``delta`` **and** ``content_index``
``response.output_text.delta``   ``delta``
``response.output_item.done``    ``item``
``response.completed``           ``response`` with ``id`` and ``usage``
``response.failed``              ``response`` with ``error``
===============================  ====================================

Anything else is traced and discarded, so emitting more is harmless and
emitting less is not.

Two failure modes here are silent, which is why the ordering below is not
cosmetic:

1. ``output_item.done`` carries the item that actually enters Codex's
   conversation. ``output_text.delta`` is for live display only -- a stream that
   emits deltas but no ``done`` renders on screen and then vanishes from
   history.
2. Codex parses each item with ``serde_json::from_value::<ResponseItem>`` and
   merely logs a debug line on failure. A reasoning item missing its
   ``encrypted_content`` key is dropped without any error surfacing, because
   ``Option<String>`` without ``#[serde(default)]`` is still a required key.
"""

from __future__ import annotations

import json
from typing import Any


def encode_event(event_type: str, payload: dict[str, Any]) -> str:
    """One SSE frame.

    The ``event:`` line is written as well as the ``type`` field: Codex reads
    the JSON, but keeping both makes the stream legible to ordinary SSE tools
    when debugging.
    """
    body = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"


def reasoning_item(item_id: str, text: str) -> dict[str, Any]:
    """A reasoning output item.

    ``encrypted_content`` must be present even though it is always null here:
    Codex's deserializer requires the key, and a missing one makes the whole
    item disappear silently.
    """
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": None,
    }


def message_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call_item(
    item_id: str, *, call_id: str, name: str, arguments: str, namespace: str | None = None
) -> dict[str, Any]:
    """A function call output item.

    ``arguments`` is a JSON *string*, not an object -- that is the Responses API
    shape, and Codex parses it itself.

    ``namespace`` is emitted only when there really is one. An ordinary
    top-level function has none, and sending ``"functions"`` (Harmony's internal
    namespace for them) would describe a routing target the client does not
    have.
    """
    item: dict[str, Any] = {
        "id": item_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    if namespace:
        item["namespace"] = namespace
    return item


class ResponseStream:
    """Builds a deterministic event sequence for one response.

    The order is fixed and testable::

        response.created
        response.in_progress
        [ per reasoning item ]
            response.output_item.added
            response.reasoning_text.delta      (xN)
            response.output_item.done
        [ per message ]
            response.output_item.added
            response.output_text.delta         (xN)
            response.output_item.done
        response.completed | response.failed
    """

    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        created_at: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.created_at = created_at
        # Echoes only the tools actually forwarded to the model, so a client can
        # see what was dropped rather than infer it.
        self.tools = tools or []
        self._sequence = 0
        self._output_index = 0

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence += 1
        return value

    def _envelope(self, *, status: str, output: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output if output is not None else [],
            "parallel_tool_calls": False,
            "tools": self.tools,
            "tool_choice": "auto",
            "metadata": {},
        }

    def created(self) -> str:
        return encode_event(
            "response.created",
            {"sequence_number": self._next_sequence(), "response": self._envelope(status="in_progress")},
        )

    def in_progress(self) -> str:
        return encode_event(
            "response.in_progress",
            {"sequence_number": self._next_sequence(), "response": self._envelope(status="in_progress")},
        )

    def item_added(self, item: dict[str, Any]) -> str:
        return encode_event(
            "response.output_item.added",
            {
                "sequence_number": self._next_sequence(),
                "output_index": self._output_index,
                "item": item,
            },
        )

    def reasoning_delta(self, item_id: str, delta: str) -> str:
        # `content_index` is required: without it Codex drops the delta.
        return encode_event(
            "response.reasoning_text.delta",
            {
                "sequence_number": self._next_sequence(),
                "item_id": item_id,
                "output_index": self._output_index,
                "content_index": 0,
                "delta": delta,
            },
        )

    def function_arguments_delta(self, item_id: str, call_id: str, delta: str) -> str:
        """Stream a tool call's arguments as they are generated.

        Codex handles `response.function_call_arguments.delta` for live display.
        The call only becomes real for it on `output_item.done`, so this event
        is progress reporting, never the payload.
        """
        return encode_event(
            "response.function_call_arguments.delta",
            {
                "sequence_number": self._next_sequence(),
                "item_id": item_id,
                "call_id": call_id,
                "output_index": self._output_index,
                "delta": delta,
            },
        )

    def text_delta(self, item_id: str, delta: str) -> str:
        return encode_event(
            "response.output_text.delta",
            {
                "sequence_number": self._next_sequence(),
                "item_id": item_id,
                "output_index": self._output_index,
                "content_index": 0,
                "delta": delta,
            },
        )

    def item_done(self, item: dict[str, Any]) -> str:
        """Close an item and advance the output index.

        This is the event that puts the item into Codex's conversation. A stream
        without it displays text and then forgets it.
        """
        frame = encode_event(
            "response.output_item.done",
            {
                "sequence_number": self._next_sequence(),
                "output_index": self._output_index,
                "item": item,
            },
        )
        self._output_index += 1
        return frame

    def completed(self, *, output: list[dict[str, Any]], usage: dict[str, Any]) -> str:
        response = self._envelope(status="completed", output=output)
        response["usage"] = usage
        response["error"] = None
        response["incomplete_details"] = None
        return encode_event(
            "response.completed", {"sequence_number": self._next_sequence(), "response": response}
        )

    def failed(self, *, message: str, error_type: str = "server_error", code: str | None = None) -> str:
        response = self._envelope(status="failed")
        response["error"] = {"type": error_type, "message": message, "code": code}
        return encode_event(
            "response.failed", {"sequence_number": self._next_sequence(), "response": response}
        )

    def heartbeat(self, processed: int, total: int) -> str:
        """Keep a stream visibly alive during a long prefill.

        An SSE comment rather than an event: comments are part of the protocol
        and every client ignores them, so this cannot be mistaken for output.
        Inventing a progress *event* would put an item into the response that
        the Responses API does not define.

        Without it, a multi-second prefill is indistinguishable from a dead
        connection, and a client with an idle timeout gives up on a request that
        is progressing normally (cahier 22).
        """
        return f": prefill {processed}/{total}\n\n"

    def done(self) -> str:
        """The terminator SDKs expect after the last event."""
        return "data: [DONE]\n\n"

