"""Prompt cache authority.

For a Codex session this is not an optimisation detail. Every turn replays the
whole conversation, so the prompt grows monotonically and each turn would
otherwise re-evaluate everything the previous one already did. Reusing that work
is the difference between a usable local agent and an unusable one.

## Why this is not a wrapper around ``mlx_lm.LRUPromptCache``

That was the plan, and measurement falsified it. ``LRUPromptCache`` keeps KV
caches in a trie and reuses them in three ways: an exact hit, a stored key that
is a *prefix* of the query, or a stored key the query diverges from -- the last
one by trimming the cache back to the common prefix.

Two facts kill all three for GPT-OSS on the Codex path:

1. **Trimming is unavailable.** GPT-OSS uses sliding-window attention on half its
   layers (12 ``RotatingKVCache`` and 12 ``KVCache`` on the 20B), and
   ``RotatingKVCache.is_trimmable()`` is ``offset < max_size`` with
   ``max_size = 128``. Any real prompt is past the window, so
   ``can_trim_prompt_cache`` is ``False`` and the divergent-branch path is
   skipped entirely.

2. **The prefix path never matches.** An entry can only be keyed by what the KV
   actually covers, which after a turn is *prompt + generated*. The next turn's
   prompt is not an extension of that: Harmony replays a finished assistant
   message with ``<|end|>`` where generation emitted ``<|return|>``, and renders
   a tool call as ``assistant to=X<|channel|>commentary`` where generation
   emitted ``assistant<|channel|>commentary to=X``. The sequences diverge, and
   without trimming a divergence is fatal.

Observed: two requests sharing a 6661-token prefix produced
``shorter=None longer=6682 common_prefix=6661`` and zero reuse.

## What this does instead

A Codex conversation is a single, linearly growing token sequence. So the cache
keeps *live sessions* rather than a trie of copies: each session remembers the
exact tokens its KV covers, and a new prompt that strictly extends those tokens
resumes it in place.

That needs no trimming, which is what makes it work at all here, and no
deep copy, which makes it cheap: ``fetch_nearest_cache`` copies a multi-gigabyte
cache on every read, and this does not.

The cost is that reuse is exact-prefix only. A prompt that diverges from every
session starts cold, because resuming from a diverged state would mean running
the model against attention state that does not match the prompt. Slower is
acceptable; wrong is not.

## On ``prompt_cache_key``

Codex sends one. It is deliberately **not** used for lookup.

A key is a claim by the client that two requests share a prefix. The tokens
establish that exactly and for free. Trusting the key instead would mean serving
state never verified against the prompt -- a silent correctness hazard in
exchange for nothing. The key is accepted, logged at debug, and ignored.

## Threading

Sessions hold live MLX arrays and are mutated in place by generation, so this
object belongs to the inference worker thread and to nothing else (D3). It
carries no lock because it is never touched from anywhere else.

## Reporting

Counting bytes means asking MLX arrays their size, which is the worker's job.
So the worker computes a :class:`CacheStats` snapshot and everyone else reads
that plain value -- no lock, no round trip, and status stays answerable while a
model is loading.

The invariant that keeps it honest: **every operation that changes cache state
republishes before returning.** That is ``fetch`` (counters), ``store``
(sessions and counters, after eviction) and ``clear``, plus the engine's own
load. Publishing anywhere else, or forgetting one of these, produces exactly the
bug this rule was written for -- a daemon holding a live session while
``/health`` reported none, correcting itself only when ``/internal/cache``
happened to be called.

The cost is bounded and lands on the worker, not on a status poll: a snapshot
walks at most ``max_entries`` sessions, once per operation that already involved
a generation.
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 4
DEFAULT_MAX_BYTES = 8 * 1024**3  # 8 GiB


@dataclass(frozen=True)
class ModelIdentity:
    """What makes two cache entries interchangeable.

    Equality *is* the compatibility rule, so this is a correctness boundary
    rather than a label: serving one model's attention state to another produces
    a plausible answer computed from the wrong weights, which no timing check
    would catch.

    ``generation`` increments on every load, so a reloaded model never inherits
    state built against the previous weights -- the served name alone cannot
    distinguish them.

    Sampling parameters are deliberately absent: temperature and top-p decide
    which token is chosen, not the attention state that produced the logits. A
    setting that *does* change the KV layout (quantised KV, a bounded window)
    would belong here, and omitting it would be a correctness bug rather than a
    missed optimisation.
    """

    served_name: str
    path: str
    generation: int

    def __str__(self) -> str:
        return f"{self.served_name}@{self.generation}"


@dataclass
class _Session:
    """One live KV cache and the exact tokens it covers."""

    identity: ModelIdentity
    tokens: list[int]
    cache: list[Any]

    @property
    def nbytes(self) -> int:
        return sum(getattr(layer, "nbytes", 0) for layer in self.cache)

    def extends(self, tokens: list[int]) -> bool:
        """True when ``tokens`` continues this session exactly.

        Strictly shorter, because a prompt entirely covered by the session would
        leave the generation loop with nothing to evaluate, and giving a token
        back would need the trimming that is unavailable here.
        """
        if len(self.tokens) >= len(tokens):
            return False
        return tokens[: len(self.tokens)] == self.tokens


@dataclass(frozen=True)
class CacheLookup:
    """The outcome of a lookup.

    ``prompt_cache`` is a *copy* of the session's state, never the session's own
    object. Generation mutates whatever it is handed, so lending out the stored
    cache would mean a request that dies mid-prefill leaves a session whose
    recorded tokens no longer describe its contents -- reusable-looking state
    that is silently wrong. The copy is nearly free because MLX arrays are
    immutable.
    """

    prompt_cache: list[Any] | None
    tokens_to_evaluate: list[int]
    cached_tokens: int

    @property
    def hit(self) -> bool:
        return self.cached_tokens > 0


@dataclass
class CacheStats:
    """Operational counters. Every number here is measured, not estimated."""

    entries: int = 0
    bytes: int = 0
    max_entries: int = 0
    max_bytes: int = 0
    hits: int = 0
    misses: int = 0
    cached_tokens_total: int = 0
    evaluated_tokens_total: int = 0
    evictions: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def hit_ratio(self) -> float:
        """Share of *requests* that reused anything.

        Deliberately not the share of tokens saved -- that is
        ``cached_tokens_total`` against the sum of both token counters.
        Conflating them would let a handful of enormous hits read as a warm
        cache.
        """
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "bytes": self.bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(self.hit_ratio, 4),
            "cached_tokens_total": self.cached_tokens_total,
            "evaluated_tokens_total": self.evaluated_tokens_total,
            "evictions": self.evictions,
            "by_model": self.by_model,
        }


class PromptCache:
    """Owns prefix reuse for one running server."""

    def __init__(
        self, *, max_entries: int = DEFAULT_MAX_ENTRIES, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes

        # Most-recently-used last.
        self._sessions: OrderedDict[int, _Session] = OrderedDict()
        self._next_id = 0
        # The session the current generation is running against, so `store` can
        # attribute the result without the caller having to carry a handle.
        self._active: int | None = None

        self._hits = 0
        self._misses = 0
        self._cached_tokens_total = 0
        self._evaluated_tokens_total = 0
        self._evictions = 0

        # Published by the worker, read by everyone else. Starts empty so a
        # status request before the first generation reports zeros rather than
        # waiting for a worker that may be loading weights.
        self._snapshot = CacheStats(max_entries=max_entries, max_bytes=max_bytes)

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0 and self._max_bytes > 0

    # -- lookup --------------------------------------------------------------

    def fetch(
        self, identity: ModelIdentity, tokens: list[int], *, hint: str | None = None
    ) -> CacheLookup:
        """Find a session whose tokens this prompt continues.

        ``hint`` is the client's ``prompt_cache_key``. It is recorded and
        otherwise unused -- see the module docstring.

        Publishes on the way out, like every mutating operation here: this one
        moves the hit and miss counters, and a snapshot that did not follow them
        would let ``/health`` and ``/internal/cache`` report different totals for
        the same cache.
        """
        try:
            return self._fetch(identity, tokens, hint=hint)
        finally:
            self.publish()

    def _fetch(
        self, identity: ModelIdentity, tokens: list[int], *, hint: str | None = None
    ) -> CacheLookup:
        if hint:
            logger.debug("prompt_cache_key=%s (recorded, not used for lookup)", hint)

        self._active = None
        if not self.enabled or not tokens:
            return self._miss(tokens)

        # Longest match wins: with several sessions on the same conversation,
        # the most advanced one saves the most work.
        best_id: int | None = None
        best_length = 0
        for session_id, session in self._sessions.items():
            if session.identity != identity:
                continue
            if session.extends(tokens) and len(session.tokens) > best_length:
                best_id, best_length = session_id, len(session.tokens)

        if best_id is None:
            return self._miss(tokens)

        self._sessions.move_to_end(best_id)
        self._active = best_id
        session = self._sessions[best_id]

        remaining = tokens[best_length:]
        self._hits += 1
        self._cached_tokens_total += best_length
        self._evaluated_tokens_total += len(remaining)
        logger.debug(
            "prompt cache resumed session=%d covered=%d evaluating=%d",
            best_id,
            best_length,
            len(remaining),
        )
        # Hand out a copy. The stored session stays exactly as recorded, so a
        # request that is cancelled, disconnected or fails leaves it intact and
        # still reusable.
        return CacheLookup(copy.deepcopy(session.cache), remaining, best_length)

    def _miss(self, tokens: list[int]) -> CacheLookup:
        self._misses += 1
        self._evaluated_tokens_total += len(tokens)
        return CacheLookup(None, tokens, 0)

    # -- insertion -----------------------------------------------------------

    def store(self, identity: ModelIdentity, tokens: list[int], prompt_cache: list[Any]) -> None:
        """Record the state generation has just produced.

        ``tokens`` must be the whole sequence the KV now covers -- prompt plus
        everything generated. Recording anything shorter would claim less
        history than the cache holds, and the next turn would resume from the
        wrong position.

        This is the operation that changes what is *resident*, so it is the one
        that most needs to republish. Without it the daemon held a live session
        while ``/health`` reported none, and the only way to see the truth was to
        call ``/internal/cache``, which republished as a side effect of being
        asked.
        """
        if not self.enabled or not tokens:
            return

        if self._active is not None and self._active in self._sessions:
            session = self._sessions[self._active]
            session.tokens = list(tokens)
            session.cache = prompt_cache
            self._sessions.move_to_end(self._active)
        else:
            session_id = self._next_id
            self._next_id += 1
            self._sessions[session_id] = _Session(
                identity=identity, tokens=list(tokens), cache=prompt_cache
            )

        self._active = None
        self._evict()
        # After eviction, never before: publishing first would advertise entries
        # the budget was about to drop.
        self.publish()

    def _evict(self) -> None:
        """Enforce the entry and byte budgets, oldest first."""
        while len(self._sessions) > self._max_entries:
            self._sessions.popitem(last=False)
            self._evictions += 1

        while self._sessions and self._total_bytes() > self._max_bytes:
            self._sessions.popitem(last=False)
            self._evictions += 1

    def _total_bytes(self) -> int:
        return sum(session.nbytes for session in self._sessions.values())

    # -- administration ------------------------------------------------------

    def clear(self) -> None:
        """Drop every session, keeping the counters as a record of what happened."""
        self._sessions.clear()
        self._active = None
        self.publish()
        logger.info("prompt cache cleared")

    def publish(self) -> CacheStats:
        """Recompute the snapshot readable from outside the worker thread.

        Counting bytes means asking MLX arrays their size, which belongs to the
        worker like everything else here (D3). So the worker computes, and
        anyone else reads :attr:`snapshot` -- a plain frozen value, no lock and
        no round trip.

        Without this, reporting cache stats meant submitting work to the worker,
        and a status request during a 30-second model load would wait behind it.
        Management has to stay answerable exactly when something slow is
        happening.
        """
        self._snapshot = self.stats()
        return self._snapshot

    @property
    def snapshot(self) -> CacheStats:
        """The last published stats. Safe to read from the event loop."""
        return self._snapshot

    def stats(self) -> CacheStats:
        by_model: dict[str, dict[str, int]] = {}
        for session in self._sessions.values():
            key = str(session.identity)
            entry = by_model.setdefault(key, {"sessions": 0, "tokens": 0, "bytes": 0})
            entry["sessions"] += 1
            entry["tokens"] += len(session.tokens)
            entry["bytes"] += session.nbytes

        return CacheStats(
            entries=len(self._sessions),
            bytes=self._total_bytes(),
            max_entries=self._max_entries,
            max_bytes=self._max_bytes,
            hits=self._hits,
            misses=self._misses,
            cached_tokens_total=self._cached_tokens_total,
            evaluated_tokens_total=self._evaluated_tokens_total,
            evictions=self._evictions,
            by_model=by_model,
        )
