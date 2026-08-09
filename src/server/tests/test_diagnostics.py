"""Diagnostics: what is recorded, what is deliberately not, and the arithmetic.

The privacy tests come first because they guard the property most easily lost by
accident. Every other field here is a number; adding "and the prompt, for
context" would be a one-line change that turns a diagnostics buffer into the
most sensitive thing the application holds.
"""

from __future__ import annotations

import json

from quantum_codex.diagnostics import Diagnostics, Outcome, RequestRecord, ToolCallRecord


def finished(**fields) -> RequestRecord:
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="req_1", model="gpt-oss-20b")
    for key, value in fields.items():
        setattr(record, key, value)
    diagnostics.finish(record, Outcome.COMPLETED)
    return record


# -- privacy -----------------------------------------------------------------


def test_a_record_carries_no_conversation_content() -> None:
    """The rule this module is built around."""
    stored = set(RequestRecord.__dataclass_fields__)

    for forbidden in ("prompt", "input", "text", "reasoning", "arguments", "output", "messages"):
        assert forbidden not in stored, f"{forbidden} would make this a transcript store"


def test_a_tool_call_records_identity_and_nothing_else() -> None:
    # Knowing *that* exec_command was called explains the turn's shape. Knowing
    # what it was asked to run is conversation content.
    stored = set(ToolCallRecord.__dataclass_fields__)

    assert stored == {"name", "namespace"}


def test_the_serialised_form_is_numbers_names_and_outcomes() -> None:
    record = finished(
        input_tokens=3348,
        cached_tokens=3349,
        output_tokens=180,
        tool_calls=[ToolCallRecord(name="exec_command")],
    )
    payload = json.dumps(record.as_dict())

    # A cheap but real check: nothing free-text beyond our own error strings.
    assert "exec_command" in payload
    for forbidden in ("prompt", "reasoning_text", "arguments"):
        assert forbidden not in payload


# -- honest arithmetic -------------------------------------------------------


def test_prefill_throughput_counts_evaluated_tokens_only() -> None:
    """Otherwise the number rises with cache reuse and measures the cache."""
    record = finished(input_tokens=10_000, cached_tokens=9_000, prefill_seconds=1.0)

    assert record.prefill_tokens_per_second == 1_000


def test_a_fully_cached_prompt_reports_no_prefill_throughput() -> None:
    # Zero tokens evaluated is not "infinitely fast"; it is not a measurement.
    record = finished(input_tokens=5_000, cached_tokens=5_000, prefill_seconds=0.1)

    assert record.prefill_tokens_per_second is None


def test_time_to_first_token_includes_the_queue() -> None:
    """What the client actually waited, which is not what the model spent."""
    record = finished(queue_wait_seconds=2.0, prefill_seconds=1.5)

    assert record.time_to_first_token_seconds == 3.5
    # Reported apart, because they answer different questions.
    assert record.prefill_seconds == 1.5


def test_missing_timings_report_nothing_rather_than_zero() -> None:
    record = finished()

    assert record.prefill_tokens_per_second is None
    assert record.decode_tokens_per_second is None
    assert record.time_to_first_token_seconds is None


def test_a_cache_hit_is_derived_from_what_was_reused() -> None:
    assert finished(cached_tokens=3_349).cache_hit is True
    assert finished(cached_tokens=0).cache_hit is False


# -- the store ---------------------------------------------------------------


def test_an_in_flight_request_is_visible_before_it_finishes() -> None:
    # So "what is it doing right now" is answerable, and a crash still leaves
    # evidence that the request started.
    diagnostics = Diagnostics()
    diagnostics.begin(request_id="req_live", model="gpt-oss-20b")

    recent = diagnostics.recent()
    assert len(recent) == 1
    assert recent[0].outcome is None


def test_history_is_bounded_and_recent_first() -> None:
    diagnostics = Diagnostics(history=3)
    for index in range(10):
        record = diagnostics.begin(request_id=f"req_{index}", model="m")
        diagnostics.finish(record, Outcome.COMPLETED)

    recent = diagnostics.recent()
    assert [r.request_id for r in recent] == ["req_9", "req_8", "req_7"]


def test_lifetime_counts_survive_the_ring_buffer() -> None:
    """The window forgets; how many requests were served should not."""
    diagnostics = Diagnostics(history=2)
    for index in range(10):
        record = diagnostics.begin(request_id=f"req_{index}", model="m")
        diagnostics.finish(record, Outcome.COMPLETED)

    aggregates = diagnostics.aggregates()
    assert aggregates["lifetime"]["requests"] == 10
    assert aggregates["lifetime"]["completed"] == 10
    assert aggregates["window"]["size"] == 2


def test_outcomes_are_counted_apart() -> None:
    diagnostics = Diagnostics()
    for outcome in (Outcome.COMPLETED, Outcome.CANCELLED, Outcome.FAILED, Outcome.INCOMPLETE):
        record = diagnostics.begin(request_id=outcome.value, model="m")
        diagnostics.finish(record, outcome)

    lifetime = diagnostics.aggregates()["lifetime"]
    assert lifetime["completed"] == 1
    assert lifetime["cancelled"] == 1
    assert lifetime["failed"] == 1
    assert lifetime["incomplete"] == 1


def test_medians_ignore_requests_that_produced_no_measurement() -> None:
    diagnostics = Diagnostics()
    for evaluated, seconds in ((1000, 1.0), (3000, 1.0), (0, 0.0)):
        record = diagnostics.begin(request_id=f"r{evaluated}", model="m")
        record.input_tokens = evaluated
        record.prefill_seconds = seconds
        diagnostics.finish(record, Outcome.COMPLETED)

    window = diagnostics.aggregates()["window"]
    # 1000 and 3000 tok/s; the third contributed nothing measurable.
    assert window["median_prefill_tokens_per_second"] == 2000


def test_an_empty_window_reports_no_ratio_rather_than_zero() -> None:
    # 0% would claim every request missed the cache; there were no requests.
    window = Diagnostics().aggregates()["window"]

    assert window["cache_hit_ratio"] is None
    assert window["median_decode_tokens_per_second"] is None


def test_the_cache_hit_ratio_counts_requests() -> None:
    diagnostics = Diagnostics()
    for cached in (0, 100, 200, 0):
        record = diagnostics.begin(request_id=str(cached), model="m")
        record.cached_tokens = cached
        record.input_tokens = 1000
        diagnostics.finish(record, Outcome.COMPLETED)

    window = diagnostics.aggregates()["window"]
    assert window["cache_hit_ratio"] == 0.5
    assert window["tokens_reused"] == 300
    assert window["tokens_evaluated"] == 3700


# -- why a turn ended --------------------------------------------------------
#
# The question a terminated session leaves behind. It has to be answerable from
# the record, because the alternative is reading raw logs.


def test_a_normal_answer_says_so() -> None:
    record = finished(finish_reason="stop")

    assert "final answer" in record.termination


def test_a_tool_call_names_the_tool() -> None:
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="r", model="m")
    record.tool_calls = [ToolCallRecord(name="exec_command")]
    diagnostics.finish(record, Outcome.COMPLETED, finish_reason="tool_call")

    assert "exec_command" in record.termination


def test_hitting_the_output_limit_is_not_reported_as_a_normal_answer() -> None:
    """The distinction the token count alone cannot make."""
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="r", model="m")
    record.output_tokens = 32768
    diagnostics.finish(record, Outcome.INCOMPLETE, finish_reason="length")

    assert "output limit" in record.termination
    assert "max_output_tokens" in record.termination


def test_a_disconnect_is_distinguishable_from_a_completed_turn() -> None:
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="r", model="m")
    diagnostics.finish(record, Outcome.CANCELLED, finish_reason="cancelled")

    assert "disconnected" in record.termination


def test_a_failure_reports_our_own_error_text() -> None:
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="r", model="m")
    diagnostics.finish(record, Outcome.FAILED, error="weights unreadable")

    assert record.termination == "weights unreadable"


def test_an_in_flight_request_has_no_termination_yet() -> None:
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="r", model="m")

    assert record.termination is None


def test_the_termination_summary_carries_no_conversation_content() -> None:
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="r", model="m")
    record.tool_calls = [ToolCallRecord(name="exec_command")]
    diagnostics.finish(record, Outcome.COMPLETED, finish_reason="tool_call")

    # Tool *names* are execution shape; arguments and outputs are conversation.
    payload = json.dumps(record.as_dict())
    for forbidden in ("prompt", "reasoning_text", "arguments", "output_text"):
        assert forbidden not in payload
