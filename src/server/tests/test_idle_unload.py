"""Idle model residency.

Weights that nothing is using are memory held for no reason, so a resident
model is released after a period of inference inactivity and reloaded on demand.
Three properties have to hold for that to be safe rather than merely convenient,
and each has its own group below:

1. **Only inference counts.** Polling status must never keep weights alive, and
   a request must never fail to keep them alive.
2. **The decision is serialised.** The timer firing and a request arriving at
   the same instant has exactly two acceptable outcomes, and "the request was
   refused" is not one of them.
3. **A timer belongs to one residency.** After a switch, the timer armed for the
   previous model must be incapable of releasing the new one.

Timing is controlled rather than waited out: the supervisor takes its timeout in
seconds, so a test asks for a few hundredths and polls for the outcome. Nothing
here sleeps for a fixed period, and nothing sleeps for minutes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from quantum_codex.inference.engine import EngineState
from quantum_codex.lifecycle import (
    LifecycleState,
    ModelBusyError,
    ModelInUseError,
    ModelSupervisor,
    UnloadReason,
)
from quantum_codex.models import ServedModel

# A timeout short enough that a test finishes instantly, long enough that it
# cannot expire during the synchronous setup that precedes it.
BRIEF = 0.02


class FakeEngine:
    """Records what it was asked to do; loads and unloads instantly."""

    def __init__(self) -> None:
        self.state = EngineState.UNLOADED
        self.loads: list[str] = []
        self.unloads = 0
        self.load_elapsed_seconds = None

    async def load(self, path, served_name, context_length):  # noqa: ANN001
        self.loads.append(served_name)
        self.state = EngineState.READY
        return type(
            "Loaded",
            (),
            {
                "served_name": served_name,
                "quantization": "mxfp4-4bit",
                "context_length": context_length,
            },
        )()

    async def unload(self) -> None:
        self.unloads += 1
        self.state = EngineState.UNLOADED


def model(slug: str) -> ServedModel:
    return ServedModel(slug=slug, display_name=slug, context_window=131072, path=f"/m/{slug}")


async def until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> bool:
    """Wait for a condition, polling the event loop rather than sleeping blind.

    A fixed sleep would either be long enough to be slow or short enough to be
    flaky. This returns the moment the condition holds and gives up well before
    a test suite would look hung.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


# -- configuration ------------------------------------------------------------


def test_the_default_idle_timeout_is_ten_minutes() -> None:
    """The product default, read from the one place that defines it."""
    from quantum_codex.config import DEFAULT_IDLE_TIMEOUT_MINUTES

    assert DEFAULT_IDLE_TIMEOUT_MINUTES == 10
    assert ModelSupervisor(FakeEngine()).idle_timeout_seconds == 600


def test_the_schema_publishes_the_same_default_and_bounds() -> None:
    """The form renders this; it must not carry a second opinion of its own."""
    from quantum_codex.config import DEFAULT_IDLE_TIMEOUT_MINUTES, MAX_IDLE_TIMEOUT_MINUTES
    from quantum_codex.profile_schema import schema

    field = next(
        f for f in schema()["fields"] if f["name"] == "model_idle_timeout_minutes"
    )
    assert field["kind"] == "integer"
    assert field["default"] == DEFAULT_IDLE_TIMEOUT_MINUTES
    assert field["minimum"] == 0
    assert field["maximum"] == MAX_IDLE_TIMEOUT_MINUTES
    assert field["unit"] == "minutes"
    # A per-model setting would let the 20B and the 120B disagree about how this
    # daemon treats idle residency, which is not a model's property.
    from quantum_codex.profile_schema import model_field_names

    assert "model_idle_timeout_minutes" not in model_field_names()


def test_a_profile_written_before_this_existed_gets_the_product_default(tmp_path, monkeypatch):
    """Item 20: migration for existing profiles is by absence, not by rewrite."""
    from quantum_codex.config import DEFAULT_IDLE_TIMEOUT_MINUTES, load_profiles

    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    (tmp_path / "profiles.json").write_text(
        '{"version": 1, "default": "dev", "profiles": {"dev": {"port": 8123}}}'
    )

    profile = load_profiles().get("dev")

    assert profile.model_idle_timeout_minutes == DEFAULT_IDLE_TIMEOUT_MINUTES


def test_the_setting_survives_a_save_and_reload(tmp_path, monkeypatch) -> None:
    from quantum_codex.config import load_profiles, save_profiles

    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    profiles = load_profiles()
    profile = profiles.create("dev")
    profile.model_idle_timeout_minutes = 45
    save_profiles(profiles)

    assert load_profiles().get("dev").model_idle_timeout_minutes == 45


def test_a_model_switch_does_not_reset_the_setting(tmp_path, monkeypatch) -> None:
    """It describes the daemon, so nothing a model does may change it."""
    from quantum_codex.config import load_profiles, save_profiles

    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    profiles = load_profiles()
    profile = profiles.create("dev")
    profile.model_idle_timeout_minutes = 3
    profile.model = "gpt-oss-20b"
    save_profiles(profiles)

    reloaded = load_profiles()
    reloaded.get("dev").model = "gpt-oss-120b"
    save_profiles(reloaded)

    assert load_profiles().get("dev").model_idle_timeout_minutes == 3


# -- arming and firing --------------------------------------------------------


def test_a_zero_timeout_arms_nothing() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert supervisor.snapshot().auto_unload_armed is False
        # Long enough that any timer with a plausible delay would have fired.
        await asyncio.sleep(0.05)

    asyncio.run(run())
    assert engine.unloads == 0
    assert supervisor.current is not None


def test_a_positive_timeout_arms_once_inference_goes_idle() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=5)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            # Nothing is armed while the model is in use: it is not idle.
            assert supervisor.snapshot().auto_unload_armed is False
        snapshot = supervisor.snapshot()
        assert snapshot.auto_unload_armed is True
        assert snapshot.idle_seconds is not None
        supervisor.begin_stopping()

    asyncio.run(run())


def test_the_timeout_releases_the_model() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert await until(lambda: engine.unloads == 1)

    asyncio.run(run())
    assert supervisor.current is None
    assert supervisor.snapshot().state is LifecycleState.IDLE
    assert supervisor.snapshot().unload_reason == UnloadReason.IDLE_TIMEOUT.value


def test_the_daemon_stays_answerable_after_an_idle_release() -> None:
    """The property the whole feature depends on not breaking.

    A released model is a normal running daemon, not a failure: `snapshot`
    reports IDLE, and nothing about the supervisor is stopping or errored.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert await until(lambda: engine.unloads == 1)

    asyncio.run(run())
    snapshot = supervisor.snapshot()
    assert snapshot.state is LifecycleState.IDLE
    assert snapshot.error is None
    assert snapshot.in_flight == 0


def test_the_next_request_loads_the_model_again() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert await until(lambda: engine.unloads == 1)
        async with supervisor.lease(model("gpt-oss-20b")):
            assert supervisor.snapshot().state is LifecycleState.READY
        supervisor.begin_stopping()

    asyncio.run(run())
    assert engine.loads == ["gpt-oss-20b", "gpt-oss-20b"]


def test_new_inference_rearms_rather_than_inheriting_the_old_deadline() -> None:
    """A second turn buys a full timeout, not whatever was left of the first."""
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0.4)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        await asyncio.sleep(0.3)
        # Well past the point where the first deadline would be close.
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        await asyncio.sleep(0.3)
        # If the timer had not been re-armed, 0.6s of elapsed time against a
        # 0.4s timeout would have released it by now.
        assert engine.unloads == 0
        assert supervisor.snapshot().idle_seconds < 0.4
        supervisor.begin_stopping()

    asyncio.run(run())


# -- what does *not* count as activity ----------------------------------------


def test_status_polling_does_not_keep_the_model_alive() -> None:
    """`/health` and `/internal/status` read `snapshot()`; nothing else.

    This is the whole mechanism by which generic daemon traffic is excluded:
    reporting takes no lease, so it cannot touch the idle clock.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0.15)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        # A frontend polling twice a second for as long as the timeout lasts.
        for _ in range(20):
            supervisor.snapshot()
            await asyncio.sleep(0.01)
        assert await until(lambda: engine.unloads == 1)

    asyncio.run(run())


def test_library_and_download_activity_does_not_keep_the_model_alive() -> None:
    """Neither takes a lease, so neither can rearm the timer.

    Written against the same supervisor the server uses rather than against a
    mock of it: the claim is about what the *lease* covers, and a test that
    called some other method would be checking a fiction.
    """
    from quantum_codex.library import MANAGER

    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0.15)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        for _ in range(20):
            # Exactly what `/internal/downloads` and `/internal/models` read.
            assert MANAGER.active is None or MANAGER.active is not None
            await asyncio.sleep(0.01)
        assert await until(lambda: engine.unloads == 1)

    asyncio.run(run())


# -- what does prevent a release ----------------------------------------------


def test_an_active_request_prevents_an_expiring_timer_from_releasing() -> None:
    """The in-flight guard, exercised at the decision point itself."""
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            epoch = supervisor._epoch  # noqa: SLF001 - the timer's own argument
            assert await supervisor._idle_expired(epoch) is False  # noqa: SLF001
        supervisor.begin_stopping()

    asyncio.run(run())
    assert engine.unloads == 0


def test_work_queued_behind_the_worker_prevents_a_release() -> None:
    """Queued inference holds its lease before it reaches the worker.

    The lease is taken in the request handler, before generation is submitted,
    so a request waiting for the single worker thread is in flight from the
    supervisor's point of view — which is what makes "queued work prevents
    unload" true without a second queue-depth accounting.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)
    started = asyncio.Event()

    async def run() -> None:
        release = asyncio.Event()

        async def queued() -> None:
            async with supervisor.lease(model("gpt-oss-20b")):
                started.set()
                await release.wait()

        first = asyncio.create_task(queued())
        second = asyncio.create_task(queued())
        await started.wait()
        await asyncio.sleep(BRIEF * 3)

        assert supervisor.snapshot().in_flight == 2
        assert await supervisor._idle_expired(supervisor._epoch) is False  # noqa: SLF001
        release.set()
        await asyncio.gather(first, second)
        supervisor.begin_stopping()

    asyncio.run(run())
    assert engine.unloads == 0


def test_a_held_lease_refuses_a_manual_unload() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            with pytest.raises(ModelInUseError, match="currently in use"):
                await supervisor.unload()

    asyncio.run(run())
    assert engine.unloads == 0
    assert supervisor.current is not None


# -- the race -----------------------------------------------------------------


def test_a_request_arriving_before_the_decision_wins() -> None:
    """Timer fired, request took the gate first: the request wins.

    Deterministic rather than probabilistic: the lease is held when the timer's
    decision runs, which is exactly the state the race produces when the request
    gets there first.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        epoch = supervisor._epoch  # noqa: SLF001
        # The timer has finished sleeping. Before it reaches the gate, a request
        # arrives and takes it.
        async with supervisor.lease(model("gpt-oss-20b")):
            assert await supervisor._idle_expired(epoch) is False  # noqa: SLF001
            assert supervisor.current is not None
        supervisor.begin_stopping()

    asyncio.run(run())
    assert engine.unloads == 0
    assert engine.loads == ["gpt-oss-20b"]


def test_a_request_arriving_during_the_release_reloads_instead_of_failing() -> None:
    """Unload won the gate: the request must follow the normal load path.

    The other half of the race, forced deterministically by an engine whose
    unload blocks. The request is waiting on the gate while the weights are
    being freed — the exact interleaving that must not produce a rejection, and
    must not hand the request a model that is being torn down.
    """

    class BlockingUnload(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.may_finish = asyncio.Event()

        async def unload(self) -> None:
            self.entered.set()
            await self.may_finish.wait()
            await super().unload()

    engine = BlockingUnload()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)
    served = asyncio.Event()

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass

        # The timer is inside the gate, holding it, mid-release.
        await asyncio.wait_for(engine.entered.wait(), timeout=2)
        assert supervisor.snapshot().state is LifecycleState.MODEL_UNLOADING

        async def arriving() -> None:
            async with supervisor.lease(model("gpt-oss-20b")):
                served.set()

        request = asyncio.create_task(arriving())
        # It cannot proceed: the release holds the gate.
        await asyncio.sleep(0.02)
        assert not served.is_set()

        engine.may_finish.set()
        await asyncio.wait_for(request, timeout=2)
        supervisor.begin_stopping()

    asyncio.run(run())
    assert served.is_set()
    assert engine.unloads == 1
    # Released, then loaded again: never served from weights being freed.
    assert engine.loads == ["gpt-oss-20b", "gpt-oss-20b"]


def test_concurrent_arrival_and_expiry_never_rejects_and_never_loses_the_model() -> None:
    """Both orderings at once, asserted on the property rather than the winner.

    Which of the two wins is genuinely a race and pinning it would be pinning
    the scheduler. What must hold either way is that the request completes and
    the model it asked for is resident when it does.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)
    resident: list[str | None] = []

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        epoch = supervisor._epoch  # noqa: SLF001

        async def arriving() -> None:
            async with supervisor.lease(model("gpt-oss-20b")):
                resident.append(supervisor.current.slug if supervisor.current else None)

        await asyncio.gather(supervisor._idle_expired(epoch), arriving())  # noqa: SLF001
        supervisor.begin_stopping()

    asyncio.run(run())
    assert resident == ["gpt-oss-20b"]


def test_an_automatic_release_is_not_logged_under_a_finished_request() -> None:
    """The timer runs on its own account, not on the last request's.

    Found in a real run: the release was logged with the request id of a turn
    that had finished a minute earlier, because `create_task` copies the context
    it was created in. A background lifecycle event attributed to a request is a
    log that misleads whoever is reading it to explain something.
    """
    from quantum_codex.logs import get_request_id, set_request_id

    seen: list[str | None] = []

    class Recording(FakeEngine):
        async def unload(self) -> None:
            seen.append(get_request_id())
            await super().unload()

    supervisor = ModelSupervisor(Recording(), idle_timeout_seconds=BRIEF)

    async def run() -> None:
        set_request_id("req_deadbeef")
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert await until(lambda: bool(seen))

    asyncio.run(run())
    assert seen == [None]


# -- residency epochs ---------------------------------------------------------


def test_a_timer_armed_for_one_model_cannot_release_the_next() -> None:
    """The stale-timer invariant, stated directly.

    Cancellation destroys the old timer when the switch takes the gate, so in a
    running server this decision is normally never reached. What is asserted
    here is the property that holds if it ever were: the release is refused
    because the residency the timer names no longer exists.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=5)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        stale = supervisor._epoch  # noqa: SLF001

        async with supervisor.lease(model("gpt-oss-120b")):
            pass
        assert supervisor.current.slug == "gpt-oss-120b"

        assert await supervisor._idle_expired(stale) is False  # noqa: SLF001
        assert supervisor.current.slug == "gpt-oss-120b"
        supervisor.begin_stopping()

    asyncio.run(run())
    # Zero, and that is the assertion. This engine's `load` does not release
    # anything on its way in — the real one does, on the worker — so every
    # `unload` here would have to have come from a release the supervisor
    # decided on. The stale timer decided on none.
    assert engine.unloads == 0
    assert engine.loads == ["gpt-oss-20b", "gpt-oss-120b"]


def test_a_fresh_timer_applies_to_the_model_that_replaced_it() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        async with supervisor.lease(model("gpt-oss-120b")):
            pass
        assert await until(lambda: supervisor.current is None)

    asyncio.run(run())
    assert supervisor.snapshot().unload_reason == UnloadReason.IDLE_TIMEOUT.value


# -- manual release -----------------------------------------------------------


def test_a_manual_unload_releases_and_leaves_the_daemon_running() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert await supervisor.unload() is True

    asyncio.run(run())
    assert engine.unloads == 1
    assert supervisor.current is None
    assert supervisor.snapshot().state is LifecycleState.IDLE
    assert supervisor.snapshot().unload_reason == UnloadReason.MANUAL.value


def test_unloading_twice_is_idempotent_rather_than_destructive() -> None:
    """A second press must report "nothing to do", not free something twice."""
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert await supervisor.unload() is True
        assert await supervisor.unload() is False

    asyncio.run(run())
    assert engine.unloads == 1


def test_manual_and_automatic_release_use_one_operation() -> None:
    """Both arrive at `_release`, so neither can clear a different set of state.

    The comparison is the whole reported state, not a chosen few fields of it.
    Asserting only `(state, in_flight, model)` would have been satisfied by a
    second release path that forgot the idle clock, or left a timer armed
    against a model it had just freed, or skipped the engine entirely -- which
    is precisely the divergence "one operation" is claimed to prevent. Anything
    that legitimately differs is named below and excluded on purpose.
    """
    reports: list[dict[str, object]] = []

    async def run(supervisor: ModelSupervisor, engine: FakeEngine, automatic: bool) -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        if automatic:
            assert await until(lambda: supervisor.current is None)
        else:
            await supervisor.unload()
        report = supervisor.snapshot().as_dict()
        # Three legitimate differences, excluded by name so that adding a fourth
        # has to be a deliberate act: `elapsed_seconds` measures wall-clock,
        # `idle_timeout_seconds` is the configuration that decides which path
        # can even happen, and `unload_reason` is the one fact the two paths are
        # *meant* to disagree about.
        report.pop("elapsed_seconds")
        report.pop("idle_timeout_seconds")
        assert report.pop("unload_reason") == (
            UnloadReason.IDLE_TIMEOUT.value if automatic else UnloadReason.MANUAL.value
        )
        report["engine_state"] = engine.state.value
        report["engine_unloads"] = engine.unloads
        report["armed"] = supervisor.snapshot().auto_unload_armed
        reports.append(report)

    manual_engine = FakeEngine()
    asyncio.run(run(ModelSupervisor(manual_engine, idle_timeout_seconds=0), manual_engine, False))
    auto_engine = FakeEngine()
    asyncio.run(run(ModelSupervisor(auto_engine, idle_timeout_seconds=BRIEF), auto_engine, True))

    assert reports[0] == reports[1]
    # Pinned as values too, so the two agreeing on something wrong still fails.
    assert reports[0] == {
        "state": "idle",
        "model": None,
        "display_name": None,
        "in_flight": 0,
        "error": None,
        "idle_seconds": None,
        "auto_unload_armed": False,
        "engine_state": "unloaded",
        "engine_unloads": 1,
        "armed": False,
    }


def test_the_unloading_state_is_reported_while_it_happens() -> None:
    """MODEL_UNLOADING is a real backend state, not something a client infers."""

    class BlockingUnload(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.may_finish = asyncio.Event()

        async def unload(self) -> None:
            self.entered.set()
            await self.may_finish.wait()
            await super().unload()

    engine = BlockingUnload()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        releasing = asyncio.create_task(supervisor.unload())
        await asyncio.wait_for(engine.entered.wait(), timeout=2)

        snapshot = supervisor.snapshot()
        assert snapshot.state is LifecycleState.MODEL_UNLOADING
        # Not a dead or stopping daemon: nothing about the server is going away.
        assert snapshot.error is None

        engine.may_finish.set()
        await releasing
        assert supervisor.snapshot().state is LifecycleState.IDLE

    asyncio.run(run())


def test_shutting_down_outranks_unloading_in_what_is_reported() -> None:
    """A daemon that is stopping is stopping, whatever it is releasing."""
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)
    supervisor._unloading = True  # noqa: SLF001
    supervisor.begin_stopping()

    assert supervisor.snapshot().state is LifecycleState.STOPPING


# -- the management surface ----------------------------------------------------
#
# Called directly rather than over HTTP: the routes are thin, and what is worth
# pinning is that they reach the same supervisor operation and translate its
# refusal into a status a client can act on.


def _unload_route(supervisor):  # noqa: ANN001
    from types import SimpleNamespace

    from quantum_codex.api.management import build_router

    router = build_router(token="t0ken", context=SimpleNamespace(supervisor=supervisor))
    return next(r for r in router.routes if r.path == "/internal/model/unload").endpoint


def test_the_endpoint_releases_and_reports_the_new_lifecycle() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)
    unload = _unload_route(supervisor)

    async def run() -> dict:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        return await unload(authorization="Bearer t0ken")

    payload = asyncio.run(run())

    assert payload["released"] is True
    assert payload["lifecycle"]["state"] == LifecycleState.IDLE.value
    assert payload["lifecycle"]["unload_reason"] == UnloadReason.MANUAL.value
    assert engine.unloads == 1


def test_the_endpoint_reports_nothing_released_when_none_was_resident() -> None:
    supervisor = ModelSupervisor(FakeEngine(), idle_timeout_seconds=0)
    unload = _unload_route(supervisor)

    payload = asyncio.run(unload(authorization="Bearer t0ken"))

    assert payload["released"] is False


def test_the_endpoint_refuses_with_a_conflict_while_the_model_is_in_use() -> None:
    """A refusal a client can present, not a generic failure.

    409 rather than 500: nothing went wrong, the request simply arrived while
    the weights were being read, and retrying shortly is the correct advice.
    """
    from quantum_codex.api.errors import ApiError

    supervisor = ModelSupervisor(FakeEngine(), idle_timeout_seconds=0)
    unload = _unload_route(supervisor)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            with pytest.raises(ApiError) as caught:
                await unload(authorization="Bearer t0ken")
            assert caught.value.status_code == 409
            assert "currently in use" in str(caught.value)

    asyncio.run(run())


def test_the_endpoint_needs_the_management_token() -> None:
    from quantum_codex.api.errors import ApiError

    supervisor = ModelSupervisor(FakeEngine(), idle_timeout_seconds=0)
    unload = _unload_route(supervisor)

    with pytest.raises(ApiError) as caught:
        asyncio.run(unload(authorization=None))
    assert caught.value.status_code == 401


# -- what the release actually frees -------------------------------------------
#
# The supervisor tests above prove the *decision*; these prove the effect. They
# run against the real `MlxEngine` with no weights: nothing is loaded, and the
# only MLX call involved is the allocator hint the unload path already makes.


def _fabricate_session(engine, tokens: list[int]) -> None:
    """Put a cache session in place without generating anything."""
    from quantum_codex.inference.prompt_cache import ModelIdentity

    class FakeKV:
        nbytes = 1024

    engine._prompt_cache.store(  # noqa: SLF001
        ModelIdentity(served_name="gpt-oss-20b", path="/m/20b", generation=1),
        tokens,
        [FakeKV()],
    )


def test_unloading_clears_the_resident_kv_state() -> None:
    """Item 14: the KV a released model owned does not outlive it."""
    from quantum_codex.inference.engine import MlxEngine

    engine = MlxEngine()
    try:
        _fabricate_session(engine, [1, 2, 3, 4])
        assert engine._prompt_cache.stats().entries == 1  # noqa: SLF001

        asyncio.run(engine.unload())

        assert engine._prompt_cache.stats().entries == 0  # noqa: SLF001
        assert engine._prompt_cache.stats().bytes == 0  # noqa: SLF001
    finally:
        engine.shutdown()


def test_the_published_cache_snapshot_reflects_the_release() -> None:
    """What the dashboard reads must change, not just the worker's own view.

    Status reads the published snapshot rather than asking the worker, so a
    release that cleared the sessions without republishing would leave the
    Prompt Cache panel describing memory nothing holds.
    """
    from quantum_codex.inference.engine import MlxEngine

    engine = MlxEngine()
    try:
        _fabricate_session(engine, [1, 2, 3, 4])
        asyncio.run(engine.cache_stats())
        assert engine.cache_snapshot.entries == 1

        asyncio.run(engine.unload())

        assert engine.cache_snapshot.entries == 0
        assert engine.cache_snapshot.bytes == 0
    finally:
        engine.shutdown()


def test_storing_a_session_is_visible_without_asking_the_worker() -> None:
    """The regression: a resident session that status could not see.

    `/health` and `/internal/status` read the published snapshot, never the
    worker. Before this, only `/internal/cache` republished — as a side effect
    of being called — so a daemon could hold a live session while every cheap
    status surface reported none.

    Note what is *not* called here: `cache_stats()`. Reading the snapshot
    directly is exactly what the status endpoints do.
    """
    from quantum_codex.inference.engine import MlxEngine

    engine = MlxEngine()
    try:
        assert engine.cache_snapshot.entries == 0

        _fabricate_session(engine, [1, 2, 3, 4])

        assert engine.cache_snapshot.entries == 1
        assert engine.cache_snapshot.bytes > 0
    finally:
        engine.shutdown()


def test_a_lookup_publishes_its_counters_too() -> None:
    """Otherwise `/health` and `/internal/cache` disagree about the same cache.

    A hit or a miss is recorded at lookup. If only `store` republished, a
    request that ended before the first token would leave the two surfaces
    reporting different totals.
    """
    from quantum_codex.inference.engine import MlxEngine
    from quantum_codex.inference.prompt_cache import ModelIdentity

    engine = MlxEngine()
    try:
        identity = ModelIdentity(served_name="gpt-oss-20b", path="/m/20b", generation=1)
        engine._prompt_cache.fetch(identity, [9, 9, 9])  # noqa: SLF001 - a miss

        assert engine.cache_snapshot.misses == 1
        assert engine.cache_snapshot.entries == 0
    finally:
        engine.shutdown()


def test_every_status_surface_agrees_about_resident_state() -> None:
    """The three readings must never disagree because one is stale.

    `/internal/cache` asks the worker; `/health` and `/internal/status` read the
    snapshot. This walks the full sequence the regression report named:
    empty → store → status sees it → unload → status sees zero → lifetime
    counters intact.
    """
    from quantum_codex.inference.engine import MlxEngine
    from quantum_codex.inference.prompt_cache import ModelIdentity

    engine = MlxEngine()
    try:
        identity = ModelIdentity(served_name="gpt-oss-20b", path="/m/20b", generation=1)

        # 1. empty
        assert engine.cache_snapshot.entries == 0

        # 2. a live session, and a hit against it
        _fabricate_session(engine, [1, 2, 3])
        engine._prompt_cache.fetch(identity, [1, 2, 3, 4])  # noqa: SLF001

        # 3. the cheap surfaces and the authoritative one give one answer
        cheap = engine.cache_snapshot
        authoritative = asyncio.run(engine.cache_stats())
        assert (cheap.entries, cheap.bytes) == (authoritative.entries, authoritative.bytes)
        assert cheap.entries == 1
        assert cheap.hits == 1

        # 4. release
        asyncio.run(engine.unload())

        # 5. resident state is zero on every surface
        after = engine.cache_snapshot
        assert (after.entries, after.bytes) == (0, 0)
        assert after.entries == asyncio.run(engine.cache_stats()).entries

        # 6. lifetime counters are a record of what happened, not resident state
        assert after.hits == 1
        assert after.cached_tokens_total == 3
    finally:
        engine.shutdown()


def test_lifetime_counters_survive_a_release_that_clears_the_sessions() -> None:
    """Resident state and the record of what happened are different facts.

    Zeroing the hit and miss totals on unload would make the cache look cold
    when it had in fact been working, and would destroy the only evidence that
    prefix reuse is happening at all.
    """
    from quantum_codex.inference.engine import MlxEngine
    from quantum_codex.inference.prompt_cache import ModelIdentity

    engine = MlxEngine()
    try:
        identity = ModelIdentity(served_name="gpt-oss-20b", path="/m/20b", generation=1)
        _fabricate_session(engine, [1, 2, 3])
        engine._prompt_cache.fetch(identity, [1, 2, 3, 4])  # noqa: SLF001 - one hit

        before = engine._prompt_cache.stats()  # noqa: SLF001
        assert before.hits == 1

        asyncio.run(engine.unload())

        after = engine.cache_snapshot
        assert after.entries == 0
        assert after.hits == before.hits
        assert after.cached_tokens_total == before.cached_tokens_total
    finally:
        engine.shutdown()


def test_a_release_makes_the_previous_generations_state_unusable() -> None:
    """No snapshot from the released model may be reused after a reload.

    The load counter is what enforces it: identity carries the generation, so an
    entry built before a release can never compare equal to one looked up after
    it, whatever the tokens say.
    """
    from quantum_codex.inference.prompt_cache import ModelIdentity, PromptCache

    class FakeKV:
        nbytes = 1024

    cache = PromptCache()
    first = ModelIdentity(served_name="gpt-oss-20b", path="/m/20b", generation=1)
    reloaded = ModelIdentity(served_name="gpt-oss-20b", path="/m/20b", generation=2)
    cache.store(first, [1, 2, 3], [FakeKV()])

    assert cache.fetch(reloaded, [1, 2, 3, 4]).hit is False


# -- failure ------------------------------------------------------------------


def test_a_failing_unload_reports_the_error_and_does_not_stick() -> None:
    """Item 12: the daemon survives, and status recovers a coherent answer."""

    class FailingUnload(FakeEngine):
        async def unload(self) -> None:
            self.unloads += 1
            raise RuntimeError("metal allocator refused")

    engine = FailingUnload()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        # Reported, not raised at the caller and not swallowed into a log line.
        assert await supervisor.unload() is True

    asyncio.run(run())
    snapshot = supervisor.snapshot()
    assert snapshot.state is not LifecycleState.MODEL_UNLOADING
    assert snapshot.error is not None
    assert "metal allocator refused" in snapshot.error


def test_a_request_after_a_failed_unload_takes_the_clean_reload_path() -> None:
    """What the engine still holds is unknown, so the next request reloads.

    Claiming the model is resident after a failed release would be a guess about
    an engine whose state nobody established.
    """

    class FailingOnce(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        async def unload(self) -> None:
            self.unloads += 1
            if self.fail:
                self.fail = False
                raise RuntimeError("transient")
            self.state = EngineState.UNLOADED

    engine = FailingOnce()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        await supervisor.unload()
        async with supervisor.lease(model("gpt-oss-20b")):
            assert supervisor.snapshot().state is LifecycleState.READY

    asyncio.run(run())
    assert engine.loads == ["gpt-oss-20b", "gpt-oss-20b"]
    # The error from the failed release is cleared by a load that succeeded.
    assert supervisor.snapshot().error is None


# -- a lease that is never taken ----------------------------------------------
#
# `lease` disarms the timer as its first act under the gate, and re-arms it in a
# `finally` that only runs once the lease has actually been taken. Everything
# that raises in between therefore leaves the timer disarmed, and nothing else
# ever arms it: the model stays resident for the life of the process while
# `auto_unload_armed` quietly reports `false`.
#
# The property these pin is not "a timer exists". It is that the idle policy
# always describes the residency: armed when something is resident and idle, not
# armed when there is nothing to release, and never restarted by a request that
# did no inference.


class RefusingLoad(FakeEngine):
    """A load that fails after releasing whatever was resident.

    Faithful to `MlxEngine._load_on_worker`, which unloads the previous model
    before it can fail: by the time an error surfaces there is nothing left in
    memory, which is why the supervisor is right to record no residency.
    """

    def __init__(self, message: str = "weights are corrupt") -> None:
        super().__init__()
        self.message = message

    async def load(self, path, served_name, context_length):  # noqa: ANN001
        if self.state is EngineState.READY:
            await self.unload()
        self.state = EngineState.FAILED
        raise RuntimeError(self.message)


def test_a_failed_replacement_leaves_no_residency_and_no_armed_timer() -> None:
    """Nothing resident, so nothing to release: not armed is the right answer."""
    engine = FakeEngine()
    # Long, deliberately: what is under test is that the timer is *not* armed
    # after the failure, and a timeout short enough to fire on its own during
    # the setup would reach the same end state for the wrong reason.
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=5)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert supervisor.snapshot().auto_unload_armed is True

        supervisor._engine = RefusingLoad()  # noqa: SLF001 - the next load fails
        supervisor._engine.state = EngineState.READY  # noqa: SLF001
        with pytest.raises(RuntimeError, match="corrupt"):
            async with supervisor.lease(model("gpt-oss-120b")):
                pass

        snapshot = supervisor.snapshot()
        assert supervisor.current is None
        assert snapshot.model is None
        assert snapshot.state is LifecycleState.ERROR
        assert snapshot.error is not None
        # No model, no idle clock, no timer. All three, or one of them is lying.
        assert snapshot.auto_unload_armed is False
        assert snapshot.idle_seconds is None

    asyncio.run(run())


def test_a_refused_lease_leaves_the_still_resident_model_armed() -> None:
    """The regression: a resident model with its automatic release switched off.

    `lease` disarms before it knows whether it will succeed. When the request is
    refused *without* the engine being touched -- here, a model with no path --
    the previous model is still resident, and a timer that is never re-armed
    holds it for the rest of the process's life.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=5)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert supervisor.snapshot().auto_unload_armed is True

        pathless = ServedModel(slug="gpt-oss-120b", display_name="x", context_window=131072)
        with pytest.raises(ModelBusyError):
            async with supervisor.lease(pathless):
                pass

        snapshot = supervisor.snapshot()
        # Untouched: the engine was never asked to do anything.
        assert supervisor.current is not None
        assert supervisor.current.slug == "gpt-oss-20b"
        assert snapshot.auto_unload_armed is True
        supervisor.begin_stopping()

    asyncio.run(run())
    assert engine.unloads == 0


def test_a_refused_lease_still_releases_at_the_original_deadline() -> None:
    """A refused request is not activity, so it may not buy another period.

    Otherwise anything able to produce a failing request -- a client looping on
    a model that is not installed -- pins the resident weights in memory for as
    long as it keeps asking.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=0.3)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        await asyncio.sleep(0.2)  # most of the period already spent

        pathless = ServedModel(slug="gpt-oss-120b", display_name="x", context_window=131072)
        for _ in range(3):
            with pytest.raises(ModelBusyError):
                async with supervisor.lease(pathless):
                    pass

        # Three refusals later the original deadline still stands: had any of
        # them restarted the clock, 0.15s from now would be far too early.
        assert await until(lambda: engine.unloads == 1, timeout=0.15)
        assert supervisor.snapshot().unload_reason == UnloadReason.IDLE_TIMEOUT.value

    asyncio.run(run())


def test_a_busy_refusal_leaves_the_rearm_to_the_lease_that_holds_it() -> None:
    """Two leases must not produce two timers, and must not produce none."""
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=5)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            with pytest.raises(ModelBusyError):
                async with supervisor.lease(model("gpt-oss-120b")):
                    pass
            # Still in use, so nothing may be scheduled to take it away.
            assert supervisor.snapshot().auto_unload_armed is False

        # The holder left; the holder armed.
        assert supervisor.snapshot().auto_unload_armed is True
        supervisor.begin_stopping()

    asyncio.run(run())


# -- a load whose awaiter is cancelled -----------------------------------------


class BlockingLoad(FakeEngine):
    """A load that can be caught mid-flight."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def load(self, path, served_name, context_length):  # noqa: ANN001
        self.entered.set()
        self.state = EngineState.LOADING
        await asyncio.Event().wait()  # never completes; the test cancels it
        raise AssertionError("unreachable")


def test_a_cancelled_load_leaves_a_coherent_supervisor() -> None:
    """A cancelled await used to escape every handler `_load` had.

    `CancelledError` is a `BaseException`, so `except Exception` did not see it:
    `_current`, `_epoch` and `_state_since` were all left describing a residency
    that the cancellation had just ended, and the lease's re-arm never ran.
    """
    engine = BlockingLoad()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async def request() -> None:
            async with supervisor.lease(model("gpt-oss-120b")):
                pass

        task = asyncio.create_task(request())
        await asyncio.wait_for(engine.entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        snapshot = supervisor.snapshot()
        assert supervisor.current is None
        assert snapshot.model is None
        # Nothing resident that this supervisor owns, so nothing to schedule.
        assert snapshot.auto_unload_armed is False
        assert snapshot.idle_seconds is None
        assert snapshot.in_flight == 0
        # Recorded rather than swallowed: something did go wrong here.
        assert snapshot.error is not None

    asyncio.run(run())


def test_a_cancelled_load_does_not_strand_a_previously_resident_model() -> None:
    """The 20B is resident, a switch to the 120B is cancelled mid-load.

    The engine released the 20B on its way into that load, so there is nothing
    left for a timer to release and not arming is correct. What must not happen
    is the supervisor still claiming the 20B is resident.
    """

    class BlockingSwitch(FakeEngine):
        """Loads the first model instantly, then hangs inside the second load.

        The release comes first, as `_load_on_worker` performs it: two GPT-OSS
        models do not fit in unified memory at once, so the old weights are gone
        before the new ones are asked for.
        """

        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.block_next = False

        async def load(self, path, served_name, context_length):  # noqa: ANN001
            if not self.block_next:
                return await super().load(path, served_name, context_length)
            if self.state is EngineState.READY:
                await self.unload()
            self.state = EngineState.LOADING
            self.entered.set()
            await asyncio.Event().wait()  # never completes; the test cancels it
            raise AssertionError("unreachable")

    engine = BlockingSwitch()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=BRIEF)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        assert supervisor.current.slug == "gpt-oss-20b"
        engine.block_next = True

        async def request() -> None:
            async with supervisor.lease(model("gpt-oss-120b")):
                pass

        task = asyncio.create_task(request())
        await asyncio.wait_for(engine.entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert supervisor.current is None
        assert engine.unloads == 1
        assert supervisor.snapshot().auto_unload_armed is False

    asyncio.run(run())


def test_the_next_request_after_a_cancelled_load_still_works() -> None:
    """Recovery, which is the property that makes the above survivable."""

    class BlockingOnce(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.block = True

        async def load(self, path, served_name, context_length):  # noqa: ANN001
            if self.block:
                self.block = False
                self.entered.set()
                await asyncio.Event().wait()
            return await super().load(path, served_name, context_length)

    engine = BlockingOnce()
    supervisor = ModelSupervisor(engine, idle_timeout_seconds=5)

    async def run() -> None:
        async def request() -> None:
            async with supervisor.lease(model("gpt-oss-20b")):
                pass

        task = asyncio.create_task(request())
        await asyncio.wait_for(engine.entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async with supervisor.lease(model("gpt-oss-20b")):
            assert supervisor.snapshot().state is LifecycleState.READY
        assert supervisor.snapshot().auto_unload_armed is True
        # A load that succeeded clears the error the cancelled one recorded.
        assert supervisor.snapshot().error is None
        supervisor.begin_stopping()

    asyncio.run(run())
