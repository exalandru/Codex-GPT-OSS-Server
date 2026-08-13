"""Daemon and model lifecycle.

The daemon and the weights it may be holding have **separate** lifetimes. The
daemon starts, serves ``/v1/models`` and its management plane, and answers
status — all with nothing resident. A model becomes resident because a request
named it, not because the process started.

That separation is why this module exists. The engine knows how to load and
unload; it does not know whether doing so right now would pull the weights out
from under a request that is mid-generation. This does, because it is the thing
that counts requests.

Reporting is deliberately cheap: :meth:`ModelSupervisor.snapshot` reads plain
attributes and takes no lock, so status stays answerable during a load that is
occupying the worker thread for half a minute.

## Idle residency

Weights that nothing is using are tens of gigabytes of unified memory held for
no reason, so a resident model is released after a configured period of
inference inactivity and reloaded on demand. Three properties make that safe,
and all three are enforced here rather than by a caller:

**Only inference counts as activity.** The lease is the single activity signal.
Everything that genuinely needs the model — an in-flight request, a request
queued behind the worker, a load performed for a request, a generation, the
completion of one — passes through :meth:`lease`, and nothing else does.
``/health``, ``/internal/status``, ``/v1/models``, diagnostics, library scans
and downloads never take a lease, so they cannot keep weights alive by being
polled. The idle clock therefore starts when the last lease is released and
stops when the next one is taken, with no separate accounting to disagree with.

**One timer, cancelled rather than checked.** At most one idle task exists, and
it is armed exactly when a model is resident with nothing using it. Taking a
lease disarms it; dropping the last one arms a fresh one. A stale timer is
normally destroyed rather than left to notice it is stale.

The disarm happens before the lease is known to succeed, which means the paths
that *fail* have to put the policy back: a refused switch, a load that failed, a
load whose awaiter was cancelled. :meth:`_restore_idle_policy` is that path, and
it restores the deadline rather than a fresh period — a request that did no
inference must not be able to extend residency, or anything able to produce a
failing request could hold the weights indefinitely.

**The final decision happens under the same gate as loading.** The timer sleeps
outside the lock and then re-acquires it to decide, so the fired-versus-arrived
race has exactly two outcomes: the request took the gate first and the unload is
refused, or the unload took it first and the request loads the model again on
its normal path. A request is never refused because a timer happened to fire,
and weights are never freed under a generation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .config import DEFAULT_IDLE_TIMEOUT_MINUTES
from .inference.engine import EngineState, MlxEngine
from .logs import set_request_id
from .models import ServedModel

logger = logging.getLogger(__name__)


class LifecycleState(StrEnum):
    """What the daemon is doing, as a user would describe it.

    ``DAEMON_STARTING`` is never reported by the server — by definition the
    process cannot answer while it is still starting. It is the *client's* state
    for "spawned, not yet answering", and it is named here so both sides use one
    vocabulary.
    """

    DAEMON_STARTING = "daemon_starting"
    IDLE = "idle"
    MODEL_LOADING = "model_loading"
    MODEL_WARMING_UP = "model_warming_up"
    READY = "ready"
    MODEL_UNLOADING = "model_unloading"
    STOPPING = "stopping"
    ERROR = "error"


class UnloadReason(StrEnum):
    """Why the last unload happened.

    Recorded because "no model is loaded" is the same observation whether the
    user asked for it, the idle timer fired, or a different model displaced it,
    and those are not the same event to someone reading status.
    """

    MANUAL = "manual"
    IDLE_TIMEOUT = "idle_timeout"
    MODEL_SWITCH = "model_switch"

    # There is deliberately no ``SHUTDOWN``. Shutdown releases weights through
    # the lifespan's own ``engine.unload()``, not through this supervisor, and
    # even if it did the reason would be unreadable: every surface that reports
    # ``unload_reason`` -- ``/health``, ``/internal/status`` -- needs a daemon
    # alive to answer, and by then there is none. ``STOPPING`` already says what
    # is happening while it happens. A member nothing can produce and nothing
    # can observe is a label, not a state.


_FROM_ENGINE: dict[EngineState, LifecycleState] = {
    EngineState.UNLOADED: LifecycleState.IDLE,
    EngineState.LOADING: LifecycleState.MODEL_LOADING,
    EngineState.WARMING_UP: LifecycleState.MODEL_WARMING_UP,
    EngineState.READY: LifecycleState.READY,
    EngineState.FAILED: LifecycleState.ERROR,
}

class ModelBusyError(RuntimeError):
    """A different model is in use and cannot be swapped out yet."""

    def __init__(
        self, requested: str, current: str, in_flight: int, detail: str | None = None
    ) -> None:
        # `detail` exists because two models can now differ while sharing a
        # name: the same slug with a different adapter is different weights.
        # Without it the message reads "`gpt-oss-120b` is serving 1 request(s);
        # `gpt-oss-120b` cannot be loaded", which explains nothing.
        super().__init__(
            f"Model `{current}` is serving {in_flight} request(s); `{requested}`"
            + (f" {detail}" if detail else "")
            + " cannot be loaded until they finish. Retry shortly, or use one "
            "model per session."
        )
        self.requested = requested
        self.current = current
        self.in_flight = in_flight
        self.detail = detail


class ModelInUseError(RuntimeError):
    """Unload was asked for while the model is still serving something.

    Distinct from :class:`ModelBusyError`, which is about a *swap*. Nothing is
    being requested here, so there is no other model to name, and the honest
    answer is that the weights are in use rather than that some other model
    cannot be loaded.
    """

    def __init__(self, current: str | None, in_flight: int) -> None:
        super().__init__(
            f"Model is currently in use: `{current or 'unknown'}` is serving "
            f"{in_flight} request(s). Unloading now would free weights a generation "
            f"is still reading."
        )
        self.current = current
        self.in_flight = in_flight


def _differs_by(current: ServedModel, requested: ServedModel) -> str | None:
    """Why these two are not the same weights, in words, or ``None``.

    Only reached when the load identities already differ, so this names the
    difference rather than deciding whether there is one. A plain model switch
    needs no explanation — the two slugs are right there in the message — and
    returns ``None``; the cases that share a name are the ones a user cannot
    otherwise account for.
    """
    if current.slug != requested.slug or current.library_id != requested.library_id:
        return None
    if current.adapter_path != requested.adapter_path:
        return "with a different LoRA adapter"
    if current.context_window != requested.context_window:
        return "with a different context length"
    if current.path != requested.path:
        return "from a different directory"
    return None


@dataclass(frozen=True)
class LifecycleSnapshot:
    """What the daemon would say about itself right now."""

    state: LifecycleState
    model: str | None
    display_name: str | None
    elapsed_seconds: float | None
    in_flight: int
    error: str | None
    #: The inactivity period actually enforced, in seconds. ``0`` means never.
    #:
    #: Reported in the unit the supervisor enforces rather than the minutes a
    #: user configures: rounding to minutes here would report a 30-second
    #: setting as "never", which is the opposite of what it does.
    idle_timeout_seconds: float = 0.0
    #: Seconds since the last inference finished, or ``None`` while one is in
    #: flight or when no model is resident. Never a countdown: the remaining
    #: time is derivable and inventing it would imply a precision the timer
    #: does not promise across a suspended machine.
    idle_seconds: float | None = None
    #: Whether an automatic unload is actually scheduled right now.
    auto_unload_armed: bool = False
    #: Why the model was released the last time one was. ``None`` until the
    #: first unload of this process.
    unload_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "model": self.model,
            "display_name": self.display_name,
            # Real elapsed seconds in the current state. Never a prediction:
            # MLX reports no progress for a weight load.
            "elapsed_seconds": (
                round(self.elapsed_seconds, 1) if self.elapsed_seconds is not None else None
            ),
            "in_flight": self.in_flight,
            "error": self.error,
            "idle_timeout_seconds": round(self.idle_timeout_seconds, 3),
            "idle_seconds": (
                round(self.idle_seconds, 1) if self.idle_seconds is not None else None
            ),
            "auto_unload_armed": self.auto_unload_armed,
            "unload_reason": self.unload_reason,
        }


class ModelSupervisor:
    """Decides which model is resident, and when it is safe to change."""

    def __init__(
        self,
        engine: MlxEngine,
        *,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_MINUTES * 60,
    ) -> None:
        self._engine = engine
        # Serialises the *decision* to load, switch or release. Generation
        # itself is already serialised by the single worker thread.
        self._gate = asyncio.Lock()
        self._current: ServedModel | None = None
        self._in_flight = 0
        self._error: str | None = None
        self._stopping = False
        self._state_since = time.perf_counter()

        # Seconds rather than minutes: the unit a user configures belongs to the
        # form, and a lifecycle mechanism that knew about it would have to be
        # changed to express anything finer.
        self._idle_timeout = max(0.0, idle_timeout_seconds)
        self._idle_task: asyncio.Task[None] | None = None
        self._idle_since: float | None = None

        # Incremented on every load and every release, so a timer armed for one
        # residency can prove it is talking about that residency and not the one
        # that replaced it.
        #
        # It is not the whole safety story and is not claimed to be. Cancellation
        # is what stops a timer acting at all; the epoch is what stops one that
        # somehow survived from releasing a *different* model. Within one
        # residency the two are indistinguishable to the epoch, so the in-flight
        # check below carries that case.
        self._epoch = 0

        self._unloading = False
        self._unload_reason: UnloadReason | None = None

    # -- reporting ------------------------------------------------------------

    @property
    def current(self) -> ServedModel | None:
        return self._current

    @property
    def idle_timeout_seconds(self) -> float:
        return self._idle_timeout

    def snapshot(self) -> LifecycleSnapshot:
        """Cheap and lock-free, so status answers during a long load."""
        if self._stopping:
            state = LifecycleState.STOPPING
        elif self._unloading:
            # Above the engine's own state on purpose. The engine still reports
            # READY until the release completes, and reporting READY through an
            # unload would make the model look available at the moment it is
            # being taken away.
            state = LifecycleState.MODEL_UNLOADING
        else:
            state = _FROM_ENGINE.get(self._engine.state, LifecycleState.IDLE)
            if state is LifecycleState.READY and self._current is None:
                # READY means "a model is resident and can serve", and this is
                # the thing that decides which model is resident. The engine
                # holding weights it was never given a lease for is not a model
                # anyone may use, and `state=ready` beside `model=null` is a
                # pair no reader can act on -- the dashboard offered *Unload
                # model* for it, and the unload then reported nothing to
                # release. One authority, one answer.
                state = LifecycleState.IDLE

        # While loading, elapsed time comes from the engine, which knows when
        # the weights started arriving. Otherwise it is time in this state.
        elapsed = self._engine.load_elapsed_seconds
        if elapsed is None and state in (LifecycleState.IDLE, LifecycleState.READY):
            elapsed = time.perf_counter() - self._state_since

        idle_seconds = (
            time.perf_counter() - self._idle_since if self._idle_since is not None else None
        )

        return LifecycleSnapshot(
            state=state,
            model=self._current.slug if self._current else None,
            display_name=self._current.display_name if self._current else None,
            elapsed_seconds=elapsed,
            in_flight=self._in_flight,
            error=self._error,
            idle_timeout_seconds=self._idle_timeout,
            idle_seconds=idle_seconds,
            auto_unload_armed=self._idle_task is not None and not self._idle_task.done(),
            unload_reason=self._unload_reason.value if self._unload_reason else None,
        )

    def begin_stopping(self) -> None:
        self._stopping = True
        # A pending unload has nothing left to release that shutdown will not
        # release anyway, and leaving it armed would have it wake into a closing
        # event loop.
        self._disarm_idle_timer()

    # -- leases ---------------------------------------------------------------

    @asynccontextmanager
    async def lease(self, model: ServedModel) -> AsyncIterator[None]:
        """Hold ``model`` resident for the duration of one request.

        A lease is what makes switching safe. While any lease is outstanding the
        resident model cannot be replaced, so a generation can never have its
        weights pulled away mid-flight. A request for a different model while
        leases are held is refused with a clear conflict rather than served by
        the wrong weights or allowed to corrupt the other session.

        It is also the definition of inference activity. Taking one stops the
        idle clock; dropping the last one starts it again.
        """
        async with self._gate:
            # First thing under the gate, and before any await: from here on the
            # model is spoken for, and a timer that fired a moment ago is now
            # answering a question nobody is asking.
            self._disarm_idle_timer()

            try:
                # `load_identity`, not `slug`: the slug is the name a request
                # asked for, and the identity is what decides which weights
                # answer it. A model whose adapter changed keeps its name and is
                # not the resident model any more.
                if (
                    self._current is not None
                    and self._current.load_identity != model.load_identity
                ):
                    difference = _differs_by(self._current, model)
                    if self._in_flight > 0:
                        raise ModelBusyError(
                            model.slug, self._current.slug, self._in_flight, detail=difference
                        )
                    logger.info(
                        "Switching model %s -> %s%s",
                        self._current.slug,
                        model.slug,
                        f" ({difference})" if difference else "",
                    )
                    self._unload_reason = UnloadReason.MODEL_SWITCH

                if self._current is None or self._current.load_identity != model.load_identity:
                    await self._load(model)
            except BaseException:
                # The lease was never taken, so the ``finally`` below -- the only
                # other place the timer is re-armed -- will not run. Leaving it
                # disarmed would hold whatever is still resident for the rest of
                # the process's life, silently: `auto_unload_armed` would report
                # `false` and nobody would know why. A refused switch, a load
                # that failed, a cancelled one: each ends here.
                self._restore_idle_policy()
                raise

            self._in_flight += 1
            self._idle_since = None

        try:
            yield
        finally:
            # No lock and no await: the event loop is single-threaded, so the
            # decrement and the re-arm happen together and cannot interleave
            # with the checks above.
            self._in_flight -= 1
            if self._in_flight == 0:
                self._idle_since = time.perf_counter()
                self._arm_idle_timer()

    async def _load(self, model: ServedModel) -> None:
        if not model.path:
            raise ModelBusyError(model.slug, model.slug, 0)
        self._error = None
        try:
            # `adapter_path` is passed unconditionally, including when it is
            # `None`. Passing it only when set would leave every engine double
            # in the tests green while the adapter never reached the real
            # engine, which is the one failure this wiring cannot afford.
            loaded = await self._engine.load(
                model.path,
                model.slug,
                model.context_window,
                adapter_path=model.adapter_path,
            )
        except BaseException as exc:
            # ``BaseException``, not ``Exception``: a cancelled await must leave
            # the same bookkeeping behind as a failed one. It used to escape
            # here, so `_current`, `_epoch` and `_state_since` all kept
            # describing a residency that no longer existed.
            #
            # What this does *not* claim to undo: the MLX worker thread cannot
            # be interrupted, so a load whose awaiter was cancelled may still
            # finish and leave weights in the engine that no lease ever
            # authorised. `snapshot` refuses to call that READY (it reports what
            # the supervisor owns, which is nothing), and the engine's own
            # `_load_on_worker` releases whatever it holds before the next load,
            # so the next request recovers. Between those two points the memory
            # is held. Closing that properly means making the load
            # non-abandonable, which is a change to the load path, not to this
            # handler.
            self._error = str(exc) or exc.__class__.__name__
            self._current = None
            self._state_since = time.perf_counter()
            self._epoch += 1
            raise
        self._current = model
        self._state_since = time.perf_counter()
        self._epoch += 1
        logger.info(
            "Model %s resident (%s, %d ctx%s)",
            loaded.served_name,
            loaded.quantization or "unquantized",
            loaded.context_length,
            ""
            if loaded.adapter is None
            else f", {loaded.adapter.applied_tensors}/"
            f"{loaded.adapter.adapter_tensors} adapter tensors",
        )

    # -- releasing ------------------------------------------------------------

    async def unload(self, reason: UnloadReason = UnloadReason.MANUAL) -> bool:
        """Release the resident model. Refused while any lease is held.

        Returns whether anything was actually released, so a caller can tell
        "there was nothing to do" from "the model is gone now" — asking twice is
        harmless and must not read as a second release.
        """
        async with self._gate:
            self._disarm_idle_timer()
            if self._in_flight > 0:
                raise ModelInUseError(
                    self._current.slug if self._current else None, self._in_flight
                )
            if self._current is None:
                return False
            await self._release(reason)
            return True

    async def _release(self, reason: UnloadReason) -> None:
        """Free the weights. The gate is held and nothing is in flight.

        The single unload operation: the idle timer, the management endpoint and
        the CLI all arrive here, so automatic and manual release cannot diverge
        in what they clear.
        """
        slug = self._current.slug if self._current else "?"
        self._unloading = True
        try:
            await self._engine.unload()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            # What the engine still holds is now unknown, so claiming the model
            # is resident would be a guess. Recorded as an error and treated as
            # not resident: the next request goes through the normal load path,
            # which releases whatever survived before loading again.
            self._error = f"unloading {slug} failed: {exc}"
            logger.error("Unloading %s failed: %s", slug, exc, exc_info=True)
        finally:
            # In `finally` so a failure cannot leave status reporting
            # MODEL_UNLOADING for the rest of the process's life.
            self._unloading = False
            self._current = None
            self._idle_since = None
            self._state_since = time.perf_counter()
            self._epoch += 1
            self._unload_reason = reason
        logger.info("Model %s released (%s)", slug, reason.value)

    # -- the idle timer -------------------------------------------------------

    def _arm_idle_timer(self, *, delay: float | None = None) -> None:
        """Schedule the release that follows this period of inactivity.

        ``delay`` defaults to the whole configured period, which is what real
        inference finishing earns. :meth:`_restore_idle_policy` passes what is
        left of the current period instead, because nothing happened.

        Synchronous on purpose. It runs in the lease's ``finally`` immediately
        after the in-flight count reaches zero, and an await between those two
        statements would open exactly the window this is meant to close.
        """
        self._disarm_idle_timer()
        # The configured timeout, never ``delay``, decides whether the feature
        # is on at all: a remaining time of zero means "release now", not
        # "automatic release is disabled".
        if self._idle_timeout <= 0 or self._current is None or self._stopping:
            return
        self._idle_task = asyncio.get_running_loop().create_task(
            self._idle_watch(self._epoch, self._idle_timeout if delay is None else delay),
            name=f"idle-unload:{self._current.slug}",
        )

    def _restore_idle_policy(self) -> None:
        """Put the idle timer back where this residency says it belongs.

        Called when a lease could not be taken -- the one path that disarms the
        timer without reaching the ``finally`` that re-arms it.

        Two rules, and the second is the one worth stating. *Armed exactly when
        a model is resident and nothing is using it*: nothing resident means
        nothing to release, so not arming is the answer rather than an omission,
        and something still in flight means its holder will arm on the way out.
        *And the clock does not restart*: a refused request is not inference, so
        it may not buy the weights another full period. Resetting it here would
        make "only inference counts as activity" false for anyone able to
        produce a failing request -- a loop asking for a model that is not
        installed would pin the resident one in memory indefinitely.
        """
        if self._in_flight > 0:
            return
        if self._current is None:
            self._idle_since = None
            return
        if self._idle_since is None:
            self._idle_since = time.perf_counter()
        elapsed = time.perf_counter() - self._idle_since
        self._arm_idle_timer(delay=max(0.0, self._idle_timeout - elapsed))

    def _disarm_idle_timer(self) -> None:
        task, self._idle_task = self._idle_task, None
        if task is not None and not task.done():
            # Not awaited, and safe from any caller on the event loop -- which
            # matters, because they are not all under the gate: `lease` and
            # `unload` hold it, `_arm_idle_timer` and `begin_stopping` do not.
            #
            # What makes it safe is not the gate but the shield in
            # `_idle_watch`: a timer that has already begun deciding runs that
            # decision to the end, and one that has not cannot begin, because
            # every path to it re-checks `_stopping`, the epoch, `_in_flight`
            # and `_current` under the gate. So this either destroys a timer
            # that would have refused anyway, or lets a release that was already
            # under way finish cleanly. It never leaves one half-done.
            task.cancel()

    async def _idle_watch(self, epoch: int, delay: float) -> None:
        # The task is created inside the lease's `finally`, so it inherits a copy
        # of that request's context — including its id. Without this, a release
        # a minute later is logged under the request that happened to be last,
        # which reads as that request having done something it did not.
        # `create_task` copied the context, so clearing it here touches only this
        # task.
        set_request_id(None)
        try:
            await asyncio.sleep(delay)
            # Shielded, so a cancellation arriving mid-decision cannot stop a
            # release that has already taken the gate and started freeing
            # weights. `begin_stopping` cancels without holding the gate, and
            # `_release` catches `Exception` but not `CancelledError`, so
            # without this a shutdown landing at the wrong instant would tear
            # `engine.unload()` in half. The sleep above is *not* shielded:
            # cancelling a timer that is still waiting is the whole mechanism.
            await asyncio.shield(self._idle_expired(epoch))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.exception("The idle-unload timer failed")

    async def _idle_expired(self, epoch: int) -> bool:
        """Decide, under the gate, whether this timer may still release.

        The synchronisation point for the whole feature. Everything that makes
        the model needed — a lease being taken, a switch, an earlier release —
        happens under this same lock, so by the time this body runs the answer
        is settled rather than being raced for.
        """
        async with self._gate:
            if self._stopping:
                return False
            if self._epoch != epoch:
                # The residency this timer was armed for is gone: the model was
                # switched or already released. Acting now would free weights
                # that belong to a different, possibly active, residency.
                logger.debug("idle timer for epoch %d is stale (now %d)", epoch, self._epoch)
                return False
            if self._in_flight > 0:
                # A request took the gate first. It wins; the next release of
                # its lease arms a fresh timer.
                return False
            if self._current is None:
                return False

            logger.info(
                "Releasing %s after %.0fs without inference activity",
                self._current.slug,
                self._idle_timeout,
            )
            await self._release(UnloadReason.IDLE_TIMEOUT)
            return True
