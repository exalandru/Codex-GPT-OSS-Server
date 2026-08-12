"""Per-request diagnostics and aggregate history.

## What this is, and is not

It records *how a request executed*: timings, token counts, cache reuse, which
tools were called, how it ended. It is deliberately not a transcript store.

Nothing here holds a prompt, a reasoning trace, a tool argument or a tool
result. That is a design rule rather than a default to be relaxed later: a
diagnostics buffer that accumulated conversation content would become the most
sensitive file the application writes, sitting in memory and in every support
bundle, for no diagnostic gain — a slow prefill is diagnosed by its token count
and duration, never by its wording.

Tool *names* are kept. They come from the client's declared tool surface, not
from the user, and a tool sequence is often the whole explanation for a turn's
shape.

## Bounded, and in memory

A ring buffer of recent requests plus a few lifetime counters. No database: a
Codex session produces a request every few seconds, so a few hundred records
covers any question worth asking about "what just happened", and the counters
cover the rest. Introducing storage would mean schema migrations and a file to
corrupt, for a problem nobody has demonstrated.

The consequence is stated rather than hidden: history does not survive a
restart, and percentiles describe the window, not all time.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .canonical import FinishReason

#: Recent requests kept. At roughly one request every few seconds, this is a
#: couple of hours of a working session.
DEFAULT_HISTORY = 200


class Outcome(StrEnum):
    COMPLETED = "completed"
    #: Stopped at the output limit, with whatever had been produced.
    INCOMPLETE = "incomplete"
    #: The client went away or asked to stop.
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCallRecord:
    """A call the model made — its identity only.

    Arguments are excluded on purpose: they are the most content-bearing part of
    a turn, and knowing *that* `exec_command` was called is what explains the
    turn's shape.
    """

    name: str
    namespace: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "namespace": self.namespace}


@dataclass
class RequestRecord:
    """One request's execution, start to finish."""

    request_id: str
    model: str
    started_at: float
    streamed: bool = False
    reasoning_effort: str | None = None

    ended_at: float | None = None
    outcome: Outcome | None = None
    #: Our own error text. Never a prompt, never model output.
    error: str | None = None

    #: Why generation stopped, in the model's own terms: which Harmony stop
    #: token ended the turn, or that the output limit was reached, or that the
    #: client went away. This is the field that answers "why did it end?"
    #: without anyone opening a log — the question a terminated session leaves
    #: behind. It names a mechanism; it never carries model text.
    finish_reason: str | None = None
    #: The last channel the model was producing on when the turn ended. An
    #: answer that ended on `final` is a completed thought; one that ended on
    #: `commentary` was mid tool call.
    last_channel: str | None = None
    #: Content-free semantic shape and terminal mechanism. These distinguish a
    #: real final answer, a tool handoff and a reasoning-only terminal turn.
    terminal_token_class: str | None = None
    had_reasoning: bool = False
    had_tool_call: bool = False
    had_final_output: bool = False
    empty_completion_detected: bool = False
    #: Reserved diagnostics for an explicitly designed recovery mechanism. QCS
    #: currently fails closed instead of silently continuing a model turn.
    recovery_attempted: bool = False
    recovery_outcome: str | None = None

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    #: Time between accepting the request and the worker starting on it.
    queue_wait_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    #: How many tools were declared, and how many survived the capability filter.
    tools_declared: int = 0
    tools_forwarded: int = 0

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    @property
    def time_to_first_token_seconds(self) -> float | None:
        """What the client actually waited before anything arrived.

        Queue wait plus prefill. Reported separately from ``prefill_seconds``
        because they answer different questions: one is how busy the server was,
        the other is how expensive the prompt was.
        """
        if self.prefill_seconds <= 0:
            return None
        return self.queue_wait_seconds + self.prefill_seconds

    @property
    def prefill_tokens_per_second(self) -> float | None:
        """Evaluated tokens per second — cached ones were not evaluated.

        Dividing the whole prompt by prefill time would report a throughput that
        rises with cache reuse, which measures the cache rather than the model.
        """
        evaluated = self.input_tokens - self.cached_tokens
        if self.prefill_seconds <= 0 or evaluated <= 0:
            return None
        return evaluated / self.prefill_seconds

    @property
    def decode_tokens_per_second(self) -> float | None:
        if self.decode_seconds <= 0 or self.output_tokens <= 0:
            return None
        return self.output_tokens / self.decode_seconds

    @property
    def cache_hit(self) -> bool:
        return self.cached_tokens > 0

    @property
    def termination(self) -> str | None:
        """One sentence naming what ended the turn, for someone not reading logs.

        Built from the mechanism, not from model output. "Answered normally" and
        "reached the output limit" look identical in a token count and mean
        completely different things to whoever is debugging a short session.
        """
        if self.outcome is None:
            return None
        if self.outcome is Outcome.CANCELLED:
            return "The client disconnected or cancelled before the turn finished."
        if self.outcome is Outcome.FAILED:
            return self.error or "The request failed before producing a result."
        if self.outcome is Outcome.INCOMPLETE:
            return (
                f"Generation reached the {self.output_tokens}-token output limit before the "
                f"model finished. Raise `max_output_tokens` for this profile."
            )
        if self.finish_reason == FinishReason.TOOL_CALL.value:
            names = ", ".join(call.name for call in self.tool_calls) or "a tool"
            return f"The model called {names} and handed control back to the client."
        return "The model produced a final answer and ended its turn."

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "streamed": self.streamed,
            "reasoning_effort": self.reasoning_effort,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "outcome": self.outcome.value if self.outcome else None,
            "finish_reason": self.finish_reason,
            "last_channel": self.last_channel,
            "terminal_token_class": self.terminal_token_class,
            "had_reasoning": self.had_reasoning,
            "had_tool_call": self.had_tool_call,
            "had_final_output": self.had_final_output,
            "empty_completion_detected": self.empty_completion_detected,
            "recovery_attempted": self.recovery_attempted,
            "recovery_outcome": self.recovery_outcome,
            "termination": self.termination,
            "error": self.error,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_hit": self.cache_hit,
            "queue_wait_seconds": self.queue_wait_seconds,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "time_to_first_token_seconds": self.time_to_first_token_seconds,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "tools_declared": self.tools_declared,
            "tools_forwarded": self.tools_forwarded,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
        }


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


class Diagnostics:
    """The bounded record of what this server has done."""

    def __init__(self, history: int = DEFAULT_HISTORY) -> None:
        self._lock = threading.Lock()
        self._records: deque[RequestRecord] = deque(maxlen=history)
        self._history = history
        # Lifetime counters, kept separately: the ring buffer forgets, and
        # "how many requests has this server served" should not.
        self._counts: dict[str, int] = {outcome.value: 0 for outcome in Outcome}
        self._total = 0

    def begin(self, *, request_id: str, model: str, **fields: Any) -> RequestRecord:
        """Open a record. It is stored immediately so an in-flight request is
        visible, and a crash still leaves evidence that it started."""
        record = RequestRecord(
            request_id=request_id, model=model, started_at=time.time(), **fields
        )
        with self._lock:
            self._records.append(record)
            self._total += 1
        return record

    def finish(
        self,
        record: RequestRecord,
        outcome: Outcome,
        *,
        error: str | None = None,
        finish_reason: str | None = None,
        last_channel: str | None = None,
    ) -> None:
        with self._lock:
            record.ended_at = time.time()
            record.outcome = outcome
            record.error = error
            if finish_reason is not None:
                record.finish_reason = finish_reason
            if last_channel is not None:
                record.last_channel = last_channel
            self._counts[outcome.value] += 1

    # -- reading -------------------------------------------------------------

    def recent(self, limit: int = 50) -> list[RequestRecord]:
        with self._lock:
            return list(self._records)[-limit:][::-1]

    def aggregates(self) -> dict[str, Any]:
        """Lifetime counts, and percentiles over the retained window.

        The two are labelled apart because they answer different questions and
        blending them would misrepresent both.
        """
        with self._lock:
            records = [r for r in self._records if r.ended_at is not None]
            counts = dict(self._counts)
            total = self._total

        prefill = [r.prefill_tokens_per_second for r in records]
        decode = [r.decode_tokens_per_second for r in records]
        ttft = [r.time_to_first_token_seconds for r in records]
        queue = [r.queue_wait_seconds for r in records]

        cache_hits = sum(1 for r in records if r.cache_hit)
        reused = sum(r.cached_tokens for r in records)
        evaluated = sum(r.input_tokens - r.cached_tokens for r in records)

        return {
            "lifetime": {
                "requests": total,
                **counts,
            },
            "window": {
                "size": len(records),
                "capacity": self._history,
                "median_prefill_tokens_per_second": _median([v for v in prefill if v]),
                "median_decode_tokens_per_second": _median([v for v in decode if v]),
                "median_time_to_first_token_seconds": _median([v for v in ttft if v]),
                "median_queue_wait_seconds": _median(queue),
                "cache_hit_ratio": (cache_hits / len(records)) if records else None,
                "tokens_reused": reused,
                "tokens_evaluated": evaluated,
            },
        }

    def as_dict(self, limit: int = 50) -> dict[str, Any]:
        return {
            **self.aggregates(),
            "requests": [record.as_dict() for record in self.recent(limit)],
        }
