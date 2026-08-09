"""SSE event shape and ordering.

Checked against Codex 0.147.0's ``process_responses_event``
(``codex-rs/codex-api/src/sse/responses.rs``), which decides what Codex acts on
and silently ignores everything else.
"""

from __future__ import annotations

import json

from quantum_codex.api.sse import ResponseStream, message_item, reasoning_item


def parse(frame: str) -> dict:
    for line in frame.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"no data line in frame: {frame!r}")


def stream() -> ResponseStream:
    return ResponseStream(response_id="resp_test", model="gpt-oss-20b", created_at=1)


def test_created_carries_a_response_object() -> None:
    # Codex only emits its Created event `if event.response.is_some()`.
    event = parse(stream().created())

    assert event["type"] == "response.created"
    assert event["response"]["id"] == "resp_test"


def test_reasoning_delta_carries_content_index() -> None:
    # `response.reasoning_text.delta` needs both `delta` and `content_index`;
    # without the index Codex drops the delta on the floor.
    event = parse(stream().reasoning_delta("rs_1", "thinking"))

    assert event["type"] == "response.reasoning_text.delta"
    assert event["delta"] == "thinking"
    assert event["content_index"] == 0


def test_text_delta_carries_a_delta() -> None:
    event = parse(stream().text_delta("msg_1", "hello"))

    assert event["type"] == "response.output_text.delta"
    assert event["delta"] == "hello"


def test_reasoning_item_includes_encrypted_content_key() -> None:
    """Regression guard for a silent drop.

    `ResponseItem::Reasoning.encrypted_content` is `Option<String>` with no
    `#[serde(default)]`, so the key is required. Codex parses items with
    `serde_json::from_value` and only logs a debug line on failure -- a missing
    key makes the reasoning item vanish with no error anywhere.
    """
    item = reasoning_item("rs_1", "some reasoning")

    assert "encrypted_content" in item
    assert item["encrypted_content"] is None
    assert item["content"] == [{"type": "reasoning_text", "text": "some reasoning"}]
    assert item["summary"] == []


def test_message_item_uses_output_text_parts() -> None:
    item = message_item("msg_1", "answer")

    assert item["role"] == "assistant"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == "answer"


def test_output_index_advances_only_on_done() -> None:
    """Deltas belong to the item that is still open."""
    events = stream()
    first = reasoning_item("rs_1", "r")
    second = message_item("msg_1", "m")

    assert parse(events.item_added(first))["output_index"] == 0
    assert parse(events.reasoning_delta("rs_1", "r"))["output_index"] == 0
    assert parse(events.item_done(first))["output_index"] == 0

    assert parse(events.item_added(second))["output_index"] == 1
    assert parse(events.item_done(second))["output_index"] == 1


def test_completed_carries_id_and_usage() -> None:
    # Codex parses `ResponseCompleted { id, usage, end_turn }`; a decode failure
    # here becomes a hard stream error, not a silent drop.
    usage = {
        "input_tokens": 70,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 61,
        "output_tokens_details": {"reasoning_tokens": 50},
        "total_tokens": 131,
    }
    event = parse(stream().completed(output=[], usage=usage))

    assert event["type"] == "response.completed"
    assert event["response"]["id"] == "resp_test"
    assert event["response"]["usage"]["input_tokens"] == 70
    assert event["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 50


def test_failed_carries_an_error_object() -> None:
    event = parse(stream().failed(message="engine died"))

    assert event["type"] == "response.failed"
    assert event["response"]["error"]["message"] == "engine died"


def test_sequence_numbers_are_monotonic() -> None:
    events = stream()
    frames = [
        events.created(),
        events.in_progress(),
        events.item_added(message_item("msg_1", "")),
        events.text_delta("msg_1", "a"),
        events.item_done(message_item("msg_1", "a")),
    ]

    numbers = [parse(frame)["sequence_number"] for frame in frames]
    assert numbers == sorted(numbers)
    assert len(set(numbers)) == len(numbers)


def test_frames_are_well_formed_sse() -> None:
    frame = stream().created()

    assert frame.startswith("event: response.created\n")
    assert frame.endswith("\n\n")
    assert stream().done() == "data: [DONE]\n\n"
