"""The MLX engine and the single thread that owns it (D3).

Every MLX operation that can depend on thread-local state runs on one
long-lived worker thread: load, warm-up, generation, cache work, unload. The
HTTP layer never touches MLX directly — it submits work and awaits a future.

Why one thread rather than "load here, generate there": MLX materialises its
per-thread stream state lazily. Loading a model on one thread and generating on
another leaves the generating thread without a stream, which surfaces as
``There is no Stream(gpu, N) in current thread`` on the first request. Keeping
every call on one thread removes the condition instead of papering over it.

Serialisation is a consequence of the same choice: a single-worker executor runs
one generation at a time, with the rest queued behind it. That is the intended
first-version behaviour (cahier 7). Batching is a later question that needs
evidence, not an assumption.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..canonical import FinishReason, GenerationTiming
from ..logs import get_request_id, set_request_id
from .prompt_cache import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ENTRIES,
    CacheStats,
    ModelIdentity,
    PromptCache,
)

logger = logging.getLogger(__name__)

# One token, generated at startup, purely to move kernel compilation off the
# first real request's latency. It is not what makes the threading correct --
# see the module docstring.
_WARMUP_TOKENS = 1


class EngineState(StrEnum):
    """What the worker is doing with weights right now.

    ``WARMING_UP`` is separate from ``LOADING`` because they fail differently
    and take very different times: loading reads tens of gigabytes from disk,
    warming up compiles kernels for one token. Collapsing them would leave a
    user watching "loading" through two unrelated waits.
    """

    UNLOADED = "unloaded"
    LOADING = "loading"
    WARMING_UP = "warming_up"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class LoadedModel:
    """Identity and shape of the model currently held by the worker."""

    served_name: str
    path: str
    context_length: int
    quantization: str | None
    num_hidden_layers: int | None


@dataclass(frozen=True)
class GenerationOutcome:
    """Raw generation result, still in tokens.

    Returning tokens rather than text is deliberate: the Harmony parser needs
    the ids, and decoding here would force a re-encode later.
    """

    tokens: list[int]
    input_tokens: int
    finish_reason: FinishReason
    timing: GenerationTiming
    # How much of the prompt came from the cache instead of being evaluated.
    # Taken from what the cache actually returned, never from a request hint.
    cached_tokens: int = 0
    # Time spent waiting for the worker, measured rather than inferred. One
    # generation runs at a time, so this is where a busy server shows up.
    queue_wait_seconds: float = 0.0
    # Exact token that matched the configured stop set. Kept as an id so this
    # layer stays independent of Harmony; the protocol layer classifies it.
    stop_token_id: int | None = None


@dataclass(frozen=True)
class PrefillProgress:
    """How far prompt evaluation has got, reported from the worker."""

    processed: int
    total: int


class ModelNotLoadedError(RuntimeError):
    """Raised when work is submitted before the model is ready."""


class Cancellation:
    """A one-way flag the generation loop polls.

    MLX cannot be interrupted from outside and the worker thread cannot be
    killed, so cancellation lands at the next token rather than immediately.
    That is the honest granularity: a request cancelled during a long prefill
    stops when prefill ends, not before.

    One object per generation, so a cancelled request can never stop the one
    that follows it.
    """

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        # A plain bool, not an Event: it is written from the event loop and read
        # from the worker, and CPython's GIL makes that assignment atomic. There
        # is no wakeup to coordinate -- the loop polls it between tokens.
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class MlxEngine:
    """Owns the model, the tokenizer and every MLX call.

    All public methods are safe to call from the event loop: each one submits to
    the worker thread and awaits the result.
    """

    def __init__(
        self,
        *,
        cache_max_entries: int = DEFAULT_MAX_ENTRIES,
        cache_max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        # No model. The engine is constructed before anything is chosen, so the
        # daemon can serve `/v1/models` and its management plane with nothing
        # resident. Which model to hold is decided per request, by the slug the
        # client asked for.
        self._model_path: Path | None = None
        self._served_name: str | None = None
        self._context_length = 0

        # When the current load began, for reporting real elapsed time. MLX
        # exposes no progress for a weight load, so elapsed seconds is the only
        # honest thing to show -- a percentage would be invented.
        self._load_started_at: float | None = None

        # Holds MLX arrays, so it belongs to the worker thread like everything
        # else here (D3). Constructed eagerly because it holds no MLX state
        # until something is inserted.
        self._prompt_cache = PromptCache(
            max_entries=cache_max_entries, max_bytes=cache_max_bytes
        )
        # Bumped on every load so a reloaded model never inherits entries built
        # against the previous weights.
        self._load_generation = 0

        # max_workers=1 is the whole mechanism: one thread, created once,
        # reused for every MLX call for the process lifetime.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")

        self._state = EngineState.UNLOADED
        self._loaded: LoadedModel | None = None
        self._model: Any = None
        self._tokenizer: Any = None

        # Queue visibility for /health. Incremented on submit rather than on
        # execution, so a request waiting for the worker is counted as queued
        # and not as invisible.
        self._counter_lock = threading.Lock()
        self._active = 0
        self._queued = 0

    # -- state ---------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def loaded_model(self) -> LoadedModel | None:
        return self._loaded

    @property
    def queue_depth(self) -> tuple[int, int]:
        """``(active, queued)`` right now."""
        with self._counter_lock:
            return self._active, self._queued

    # -- submission ----------------------------------------------------------

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        with self._counter_lock:
            self._queued += 1
        # Captured on the event loop: a ContextVar set there is invisible to the
        # worker thread, so it has to be carried across explicitly.
        request_id = get_request_id()

        def tracked() -> Any:
            with self._counter_lock:
                self._queued -= 1
                self._active += 1
            set_request_id(request_id)
            try:
                return fn(*args)
            finally:
                set_request_id(None)
                with self._counter_lock:
                    self._active -= 1

        return await asyncio.wrap_future(self._executor.submit(tracked))

    # -- lifecycle -----------------------------------------------------------

    async def load(
        self, model_path: str | Path, served_name: str, context_length: int
    ) -> LoadedModel:
        """Make one model resident. Replaces whatever was loaded before.

        The caller decides *when* switching is safe; this only performs it. See
        :mod:`quantum_codex.lifecycle`, which owns that judgement because it is
        the thing that knows about in-flight requests.
        """
        return await self._run(self._load_on_worker, Path(model_path), served_name, context_length)

    @property
    def load_elapsed_seconds(self) -> float | None:
        """How long the current load has been running, or ``None`` when idle.

        Real elapsed time, not an estimate of what remains: mlx-lm exposes no
        progress for a weight load, so any completion figure would be fiction.
        """
        if self._load_started_at is None:
            return None
        return time.perf_counter() - self._load_started_at

    async def unload(self) -> None:
        await self._run(self._unload_on_worker)

    def shutdown(self) -> None:
        """Stop the worker. Called on process shutdown, not per request."""
        self._executor.shutdown(wait=True)

    # -- prompt cache ---------------------------------------------------------

    async def cache_stats(self) -> CacheStats:
        """Read cache counters.

        Goes through the worker like everything else: the entries hold MLX
        arrays, and reading their byte totals from another thread would be
        touching MLX from outside its owner (D3).
        """
        return await self._run(self._prompt_cache.publish)

    async def clear_cache(self) -> None:
        await self._run(self._prompt_cache.clear)

    @property
    def cache_snapshot(self) -> CacheStats:
        """Last published cache counters, readable without the worker.

        What ``/health`` and the management plane report. Asking the worker
        would make status wait behind a model load or a generation, which is
        precisely when someone is watching it.
        """
        return self._prompt_cache.snapshot

    def _load_on_worker(
        self, model_path: Path, served_name: str, context_length: int
    ) -> LoadedModel:
        import json

        import mlx.core as mx
        from mlx_lm.utils import load as mlx_load

        # Anything previously resident goes first. Loading a second set of
        # weights before releasing the first would need both in unified memory
        # at once, which for two GPT-OSS models does not fit.
        if self._model is not None:
            self._unload_on_worker()

        self._model_path = model_path
        self._served_name = served_name
        self._context_length = context_length

        self._state = EngineState.LOADING
        started = time.perf_counter()
        self._load_started_at = started
        logger.info("Loading model %s from %s", served_name, model_path)

        try:
            self._model, self._tokenizer = mlx_load(str(self._model_path))
        except Exception:
            self._state = EngineState.FAILED
            self._load_started_at = None
            raise

        config: dict[str, Any] = {}
        config_path = self._model_path / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())

        quantization = None
        if isinstance(config.get("quantization"), dict):
            quant = config["quantization"]
            quantization = f"{quant.get('mode', 'quantized')}-{quant.get('bits', '?')}bit"

        self._loaded = LoadedModel(
            served_name=self._served_name,
            path=str(self._model_path),
            context_length=self._context_length,
            quantization=quantization,
            num_hidden_layers=config.get("num_hidden_layers"),
        )

        self._prompt_cache.publish()
        self._load_generation += 1
        self._state = EngineState.WARMING_UP
        self._warm_up()
        mx.clear_cache()

        self._state = EngineState.READY
        self._load_started_at = None
        logger.info(
            "Model %s ready in %.1fs (%s)",
            self._served_name,
            time.perf_counter() - started,
            quantization or "unquantized",
        )
        return self._loaded

    def _warm_up(self) -> None:
        """Generate one token so kernel compilation is not billed to request 1."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        prompt = mx.array(self._tokenizer.encode("hi"))
        for _ in generate_step(prompt=prompt, model=self._model, max_tokens=_WARMUP_TOKENS):
            break

    def _unload_on_worker(self) -> None:
        import gc

        import mlx.core as mx

        # Entries hold KV state for weights that are about to disappear.
        self._prompt_cache.clear()
        self._model = None
        self._tokenizer = None
        self._loaded = None
        self._model_path = None
        self._served_name = None
        self._context_length = 0
        self._state = EngineState.UNLOADED
        self._load_started_at = None
        gc.collect()
        mx.clear_cache()

    # -- generation ----------------------------------------------------------

    async def generate(
        self,
        prompt_tokens: list[int],
        *,
        stop_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        cache_hint: str | None = None,
        cancellation: Cancellation | None = None,
    ) -> GenerationOutcome:
        if self._state is not EngineState.READY:
            raise ModelNotLoadedError(f"engine is {self._state.value}, not ready")
        submitted_at = time.perf_counter()
        outcome = await self._run(
            self._generate_on_worker,
            prompt_tokens,
            stop_tokens,
            max_tokens,
            temperature,
            top_p,
            None,
            cache_hint,
            cancellation,
            None,
        )
        # Includes the await, which is exactly the wait a client experienced.
        return replace(
            outcome,
            queue_wait_seconds=max(
                0.0, (time.perf_counter() - submitted_at) - outcome.timing.total_seconds
            ),
        )

    async def generate_stream(
        self,
        prompt_tokens: list[int],
        *,
        stop_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        cache_hint: str | None = None,
    ) -> AsyncIterator[int | GenerationOutcome]:
        """Yield each token as it is produced, then the final outcome.

        Generation still happens entirely on the worker thread; only the tokens
        cross over. The worker hands each one to the event loop with
        ``call_soon_threadsafe``, which is the one direction that is safe -- the
        reverse, touching MLX from the loop, is what D3 forbids.

        The outcome is yielded last so a consumer that streams deltas can still
        report exact usage and timing at the end.
        """
        if self._state is not EngineState.READY:
            raise ModelNotLoadedError(f"engine is {self._state.value}, not ready")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        finished = object()
        cancellation = Cancellation()

        def on_token(token_id: int) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, token_id)

        def on_prefill(processed: int, total: int) -> None:
            # Prefill can run for many seconds before the first token, which
            # looks identical to a dead connection from the client's side.
            # Reporting progress from the worker is what lets the caller keep
            # the stream visibly alive (cahier 22).
            loop.call_soon_threadsafe(queue.put_nowait, PrefillProgress(processed, total))

        with self._counter_lock:
            self._queued += 1
        request_id = get_request_id()
        submitted_at = time.perf_counter()

        def tracked() -> GenerationOutcome:
            with self._counter_lock:
                self._queued -= 1
                self._active += 1
            set_request_id(request_id)
            queue_wait = time.perf_counter() - submitted_at
            try:
                outcome = self._generate_on_worker(
                    prompt_tokens,
                    stop_tokens,
                    max_tokens,
                    temperature,
                    top_p,
                    on_token,
                    cache_hint,
                    cancellation,
                    on_prefill,
                )
                return replace(outcome, queue_wait_seconds=queue_wait)
            finally:
                set_request_id(None)
                with self._counter_lock:
                    self._active -= 1

        future = self._executor.submit(tracked)
        # Fires whether the worker returned or raised, so a failure cannot leave
        # the consumer waiting on a queue nothing will ever fill.
        future.add_done_callback(lambda _: loop.call_soon_threadsafe(queue.put_nowait, finished))

        try:
            while True:
                item = await queue.get()
                if item is finished:
                    break
                yield item

            # Raises here if the worker failed, after the loop rather than
            # inside it, so the error is not mistaken for end-of-stream.
            yield await asyncio.wrap_future(future)
        finally:
            # Reached when the consumer stops early: a disconnected client, a
            # cancelled task, or an exception upstream. Without this the worker
            # would keep generating to `max_tokens` against nobody, holding the
            # queue closed to every request behind it.
            if not future.done():
                cancellation.cancel()
                logger.info("generation cancelled by the client; draining the worker")
                # Wait for the worker to notice. Returning while it still runs
                # would let the next request start against a busy engine and a
                # cache mid-write.
                await asyncio.shield(asyncio.wrap_future(future))

    def _generate_on_worker(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        on_token: Callable[[int], None] | None = None,
        cache_hint: str | None = None,
        cancellation: Cancellation | None = None,
        on_prefill: Callable[[int, int], None] | None = None,
    ) -> GenerationOutcome:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        stop = set(stop_tokens)
        sampler = make_sampler(temp=temperature, top_p=top_p)

        identity = ModelIdentity(
            served_name=self._served_name,
            path=str(self._model_path),
            generation=self._load_generation,
        )
        lookup = self._prompt_cache.fetch(identity, prompt_tokens, hint=cache_hint)
        kv_cache = lookup.prompt_cache
        if kv_cache is None:
            kv_cache = make_prompt_cache(self._model)

        prompt = mx.array(lookup.tokens_to_evaluate)

        generated: list[int] = []
        finish_reason = FinishReason.LENGTH
        stop_token_id: int | None = None
        started = time.perf_counter()
        first_token_at: float | None = None

        # No `max_kv_size`: that would install a rotating cache which silently
        # discards the oldest tokens once the window fills. For a long Codex
        # session that means quietly losing the start of the conversation. The
        # context limit is enforced up front instead, so overflow is a clear
        # error rather than invisible truncation.
        snapshot_taken = False

        for token, _logprobs in generate_step(
            prompt=prompt,
            model=self._model,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=kv_cache,
            prompt_progress_callback=on_prefill,
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()

            if not snapshot_taken:
                snapshot_taken = True
                # Snapshot at the prompt boundary, not at the end of the turn.
                #
                # Generation and replay serialise an assistant turn differently:
                # Harmony emits `assistant<|channel|>commentary to=X` while
                # replaying the same call renders `assistant to=X<|channel|>
                # commentary`. A session recorded after a tool call therefore
                # diverges from the next turn's prompt and can never be resumed
                # -- measured: zero reuse across a real three-turn Codex session.
                # The prompt itself always renders identically, so the boundary
                # right after prefill is the last point both agree on.
                #
                # The copy is cheap: MLX arrays are immutable, so this shares
                # buffers and the live cache's later writes do not reach it
                # (verified: the snapshot's offset and contents are unchanged
                # after 20 more tokens).
                self._prompt_cache.store(
                    identity, prompt_tokens + [int(token)], copy.deepcopy(kv_cache)
                )

            token_id = int(token)
            # The stop token is kept: Harmony's parser uses it to close the
            # final message, and the model did generate it, so usage should
            # count it.
            generated.append(token_id)
            if on_token is not None:
                on_token(token_id)

            if token_id in stop:
                finish_reason = FinishReason.STOP
                stop_token_id = token_id
                break

            # Polled after the token is recorded, so a cancelled turn keeps
            # everything the model actually produced rather than discarding a
            # partial answer the client may already have displayed.
            if cancellation is not None and cancellation.cancelled:
                finish_reason = FinishReason.CANCELLED
                break

        ended = time.perf_counter()
        if first_token_at is None:
            first_token_at = ended

        timing = GenerationTiming(
            prefill_seconds=first_token_at - started,
            decode_seconds=ended - first_token_at,
        )

        return GenerationOutcome(
            tokens=generated,
            input_tokens=len(prompt_tokens),
            finish_reason=finish_reason,
            timing=timing,
            cached_tokens=lookup.cached_tokens,
            stop_token_id=stop_token_id,
        )
