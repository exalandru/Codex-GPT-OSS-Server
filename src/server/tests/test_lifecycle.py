"""The daemon's lifetime and the model's are separate.

A daemon holding no weights is a normal, healthy state. These tests pin the two
properties that make that safe: the catalogue advertises only what could
actually load, and weights are never swapped out from under a request.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from quantum_codex.config import ConfigError
from quantum_codex.inference.engine import EngineState
from quantum_codex.library.registry import ModelState
from quantum_codex.lifecycle import LifecycleState, ModelBusyError, ModelSupervisor
from quantum_codex.models import (
    ServedModel,
    resolve_served_catalogue,
    served_models_from_library,
    slug_for,
)

# -- the catalogue -----------------------------------------------------------


@dataclass
class FakeEntry:
    name: str
    path: str
    id: str = ""


@dataclass
class FakeReport:
    entry: FakeEntry
    state: ModelState
    context_length: int | None = 131072
    quantization: str | None = "mxfp4-4bit"


def report(
    name: str,
    state: ModelState = ModelState.READY,
    *,
    model_id: str = "",
    path: str | None = None,
    **kwargs,
) -> FakeReport:
    return FakeReport(
        entry=FakeEntry(name=name, path=path or f"/models/{name}", id=model_id),
        state=state,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        ("gpt-oss-20b-mxfp4-bf16", "gpt-oss-20b"),
        ("gpt-oss-120b-mxfp4-bf16", "gpt-oss-120b"),
        ("gpt-oss-20b-MXFP4-Q8", "gpt-oss-20b"),
        ("gpt-oss-20b", "gpt-oss-20b"),
    ],
)
def test_a_slug_drops_the_quantisation_suffix(directory: str, expected: str) -> None:
    """What a user types after `--model`, not a filesystem detail."""
    assert slug_for(directory) == expected


def test_every_usable_model_is_advertised() -> None:
    models = served_models_from_library(
        [report("gpt-oss-20b-mxfp4-bf16"), report("gpt-oss-120b-mxfp4-bf16")]
    )

    assert [m.slug for m in models] == ["gpt-oss-20b", "gpt-oss-120b"]


@pytest.mark.parametrize(
    "state",
    [
        ModelState.MISSING_VOLUME,
        ModelState.MISSING,
        ModelState.PARTIAL_DOWNLOAD,
        ModelState.INCOMPATIBLE,
        ModelState.INVALID,
    ],
)
def test_a_model_that_could_not_load_is_not_advertised(state: ModelState) -> None:
    """Offering it would fail at the moment a user tried to start work."""
    models = served_models_from_library([report("gpt-oss-20b-mxfp4-bf16", state)])

    assert models == ()


def test_a_contested_served_name_is_served_by_neither_model() -> None:
    """Two models answering to one name makes the served weights unknowable.

    This replaces a witness for the previous contract, which kept the first
    entry and logged the loser. Keeping one is a choice the server has no basis
    to make: nothing distinguishes the two, and the request that arrives asks
    for a name, not for weights.
    """
    catalogue = resolve_served_catalogue(
        [report("gpt-oss-20b-mxfp4-bf16"), report("gpt-oss-20b-mxfp4")]
    )

    assert catalogue.models == ()
    assert [problem.reason for problem in catalogue.problems] == ["duplicate"]
    assert catalogue.problems[0].served_name == "gpt-oss-20b"


def test_a_contested_name_does_not_take_the_rest_of_the_catalogue_with_it() -> None:
    """Containment: one ambiguous pair must not stop the daemon serving.

    The strict reading raises, and that reading belongs to the boundary that
    stores a configuration. A running server keeps every model whose name is
    unambiguous, so a library that has become ambiguous is still a library a
    user can work with -- and fix.
    """
    reports = [
        report("gpt-oss-20b-mxfp4-bf16"),
        report("gpt-oss-20b-mxfp4"),
        report("gpt-oss-120b-mxfp4-bf16"),
    ]

    catalogue = resolve_served_catalogue(reports)

    assert [m.slug for m in catalogue.models] == ["gpt-oss-120b"]
    with pytest.raises(ConfigError, match="gpt-oss-20b"):
        served_models_from_library(reports)


def test_a_served_name_that_no_client_could_use_is_refused() -> None:
    """The name reaches a generated `config.toml` verbatim; a quote breaks it."""
    catalogue = resolve_served_catalogue(
        [report("custom-model", model_id="custom-model")],
        overrides={"custom-model": {"served_model_name": 'evil"\nmodel = "other'}},
    )

    assert catalogue.models == ()
    assert catalogue.problems[0].reason == "invalid"


def test_settings_do_not_leak_between_two_identically_named_directories() -> None:
    """A disambiguated id is a *different* model, not the same one renamed."""
    first = report("shared-model", model_id="shared-model", path="/first/shared-model")
    second = report("shared-model", model_id="shared-model-ab12cd34", path="/second/shared-model")

    catalogue = resolve_served_catalogue(
        [first, second], overrides={"shared-model": {"served_model_name": "alpha"}}
    )

    assert {m.id: m.slug for m in catalogue.models} == {
        "shared-model": "alpha",
        "shared-model-ab12cd34": "shared-model",
    }
    assert catalogue.problems == ()


def test_the_catalogue_carries_the_path_needed_to_load() -> None:
    model = served_models_from_library([report("gpt-oss-20b-mxfp4-bf16")])[0]

    assert model.path == "/models/gpt-oss-20b-mxfp4-bf16"
    assert model.context_window == 131072


# -- the supervisor ----------------------------------------------------------


class FakeEngine:
    """Records what it was asked to do; loads instantly."""

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


def test_a_fresh_daemon_holds_no_model() -> None:
    supervisor = ModelSupervisor(FakeEngine())
    snapshot = supervisor.snapshot()

    assert snapshot.state is LifecycleState.IDLE
    assert snapshot.model is None


def test_a_lease_loads_the_requested_model() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            assert supervisor.snapshot().state is LifecycleState.READY
            assert supervisor.snapshot().model == "gpt-oss-20b"

    asyncio.run(run())
    assert engine.loads == ["gpt-oss-20b"]


def test_a_second_request_for_the_same_model_does_not_reload() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine)

    async def run() -> None:
        for _ in range(3):
            async with supervisor.lease(model("gpt-oss-20b")):
                pass

    asyncio.run(run())
    assert engine.loads == ["gpt-oss-20b"]


def test_switching_models_is_explicit_when_nothing_is_in_flight() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        async with supervisor.lease(model("gpt-oss-120b")):
            pass

    asyncio.run(run())
    assert engine.loads == ["gpt-oss-20b", "gpt-oss-120b"]


def test_a_model_is_never_swapped_out_from_under_a_live_request() -> None:
    """The property the lease exists for.

    A generation holds weights it is mid-way through using. Switching under it
    would corrupt that session, so the second request is refused instead.
    """
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            with pytest.raises(ModelBusyError) as caught:
                async with supervisor.lease(model("gpt-oss-120b")):
                    pass
            assert "gpt-oss-120b" in str(caught.value)
            assert "gpt-oss-20b" in str(caught.value)

    asyncio.run(run())
    assert engine.loads == ["gpt-oss-20b"]
    assert engine.unloads == 0


def test_the_switch_succeeds_once_the_request_finishes() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine)

    async def run() -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            pass
        async with supervisor.lease(model("gpt-oss-120b")):
            assert supervisor.current.slug == "gpt-oss-120b"

    asyncio.run(run())
    assert engine.loads == ["gpt-oss-20b", "gpt-oss-120b"]


def test_two_concurrent_requests_for_the_same_model_both_proceed() -> None:
    engine = FakeEngine()
    supervisor = ModelSupervisor(engine)
    seen = []

    async def one(tag: str) -> None:
        async with supervisor.lease(model("gpt-oss-20b")):
            seen.append(tag)
            await asyncio.sleep(0)

    async def both() -> None:
        await asyncio.wait_for(asyncio.gather(one("a"), one("b")), timeout=5)

    asyncio.run(both())
    assert sorted(seen) == ["a", "b"]
    assert engine.loads == ["gpt-oss-20b"]


def test_a_load_failure_is_reported_and_leaves_no_current_model() -> None:
    class Failing(FakeEngine):
        async def load(self, path, served_name, context_length):  # noqa: ANN001
            raise RuntimeError("weights unreadable")

    supervisor = ModelSupervisor(Failing())

    async def run() -> None:
        with pytest.raises(RuntimeError, match="weights unreadable"):
            async with supervisor.lease(model("gpt-oss-20b")):
                pass

    asyncio.run(run())
    snapshot = supervisor.snapshot()
    assert snapshot.model is None
    assert snapshot.error == "weights unreadable"


def test_stopping_is_reported_even_while_a_model_is_resident() -> None:
    engine = FakeEngine()
    engine.state = EngineState.READY
    supervisor = ModelSupervisor(engine)

    supervisor.begin_stopping()

    assert supervisor.snapshot().state is LifecycleState.STOPPING


@pytest.mark.parametrize(
    ("engine_state", "expected"),
    [
        (EngineState.UNLOADED, LifecycleState.IDLE),
        (EngineState.LOADING, LifecycleState.MODEL_LOADING),
        (EngineState.WARMING_UP, LifecycleState.MODEL_WARMING_UP),
        (EngineState.READY, LifecycleState.READY),
        (EngineState.FAILED, LifecycleState.ERROR),
    ],
)
def test_every_engine_state_maps_to_a_reportable_lifecycle_state(
    engine_state: EngineState, expected: LifecycleState
) -> None:
    """A state with no mapping would show as IDLE while something was happening."""
    engine = FakeEngine()
    engine.state = engine_state
    supervisor = ModelSupervisor(engine)
    # READY is the one engine state that says something about *residency* and
    # not just about the worker, and the supervisor is what owns residency. So
    # the fixture has to agree with itself: an engine reporting READY under a
    # supervisor holding nothing is a contradiction the supervisor now refuses
    # to pass on (see the test below). Every other row is unaffected, and
    # deleting any row from the mapping still fails this.
    if engine_state is EngineState.READY:
        supervisor._current = model("gpt-oss-20b")  # noqa: SLF001

    assert supervisor.snapshot().state is expected


def test_a_ready_engine_the_supervisor_owns_nothing_in_is_not_reported_ready() -> None:
    """READY has to mean "there is a model, and you may use it".

    The engine can end up holding weights no lease authorised -- an MLX load
    runs on a worker thread that cannot be interrupted, so an awaiter that is
    cancelled leaves the load to finish anyway. Passing the engine's READY
    through produced `state=ready` beside `model=null`: the dashboard offered
    *Unload model*, and pressing it answered "no model was resident", because
    the supervisor was right and the state line was not.
    """
    engine = FakeEngine()
    engine.state = EngineState.READY
    supervisor = ModelSupervisor(engine)

    snapshot = supervisor.snapshot()

    assert supervisor.current is None
    assert snapshot.model is None
    assert snapshot.state is LifecycleState.IDLE
