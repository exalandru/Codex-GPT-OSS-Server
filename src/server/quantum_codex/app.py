"""The FastAPI application and the `serve` entry point.

The HTTP layer validates, normalises into the IR, and hands work to the engine.
It never touches MLX (D3) and never passes a raw request dictionary downstream
(D2).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .api.errors import ApiError, api_error_handler, context_overflow, invalid_request
from .api.management import build_router as build_management_router
from .api.management import mint_token
from .api.responses import build_response, build_usage, new_response_id
from .api.schemas import parse_request, to_canonical_turn
from .api.sse import ResponseStream, function_call_item, message_item, reasoning_item
from .canonical import (
    CanonicalTurn,
    CanonicalTurnResult,
    FinishReason,
    ReasoningEffort,
    ToolCall,
    Usage,
)
from .codex import CAPABILITIES, build_models_response
from .codex.capabilities import log_dropped_tools
from .config import (
    DEFAULT_IDLE_TIMEOUT_MINUTES,
    RuntimeState,
    clear_runtime_state,
    migrate_app_support_root,
    write_runtime_state,
)
from .diagnostics import Diagnostics, Outcome, RequestRecord, ToolCallRecord
from .harmony import (
    ANALYSIS,
    COMMENTARY,
    FINAL,
    HarmonyRenderer,
    StreamingParser,
    parse_completion,
)
from .inference import MlxEngine
from .inference.engine import GenerationOutcome, PrefillProgress
from .inference.prompt_cache import DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES
from .library import volume_for
from .library.registry import load_registry
from .lifecycle import ModelBusyError, ModelSupervisor
from .logs import configure as configure_logging
from .logs import set_request_id
from .model_settings import load_model_settings
from .models import (
    ModelRegistry,
    ServedModel,
    resolve_served_catalogue,
)
from .routing import ToolRouter

logger = logging.getLogger(__name__)

# Reserved for the answer when neither the request nor the model sets one. The
# real cap is whatever is left of the context window; this only stops an
# unbounded generation from running until the window is exhausted.
DEFAULT_MAX_OUTPUT_TOKENS = 32768

#: What a client is told when a turn ended with neither of the two things an
#: assistant turn can end with. Content-free by construction: it describes the
#: shape of the turn and quotes nothing the model produced.
REASONING_ONLY_COMPLETION = (
    "Generation stopped without a final answer or a tool call. "
    "The turn was not reported as complete."
)

#: The wire code for that classification, on both the JSON and the SSE path.
EMPTY_COMPLETION_CODE = "empty_completion"


@dataclass(frozen=True)
class ServerDefaults:
    """What a request inherits when it does not say otherwise.

    The middle of the three levels the cahier separates (26): the model's
    capability bounds what is possible, this is the server's default, and a
    request may narrow it. Keeping them distinct is what stops a profile setting
    from silently overriding a client, or a client from exceeding the model.
    """

    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0


class ServerContext:
    """Everything one running server owns."""

    def __init__(
        self,
        engine: MlxEngine,
        registry: ModelRegistry,
        renderer: HarmonyRenderer,
        defaults: ServerDefaults,
        *,
        idle_timeout_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.renderer = renderer
        self.defaults = defaults
        self.diagnostics = Diagnostics()
        self.started_at = time.time()
        # Minutes are the unit a user configures; the supervisor enforces
        # seconds. Converted once, here, at the boundary between the two.
        self.supervisor = ModelSupervisor(
            engine, idle_timeout_seconds=max(0, idle_timeout_minutes) * 60
        )


def refresh_registry(context: ServerContext) -> None:
    """Rebuild the served catalogue from the model library.

    Read from disk rather than cached for the process lifetime: a user can
    import or download a model while the daemon runs, and a catalogue that only
    reflected startup would tell them their new model is not served.
    """
    try:
        reports = load_registry().report()
        overrides = load_model_settings().overrides
        catalogue = resolve_served_catalogue(
            reports,
            default_effort=context.defaults.reasoning_effort,
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 - a broken library must not kill the daemon
        logger.warning("Could not read the model library: %s", exc)
        return

    # An unusable served name removes *that* name from the catalogue and nothing
    # else. Refusing to build the catalogue at all would let one ambiguous pair
    # stop the daemon from starting, taking every unaffected model down with it
    # and leaving no running server through which to fix the configuration.
    for problem in catalogue.problems:
        logger.warning("Not serving a model: %s", problem.message)
    context.registry.replace_all(catalogue.models)


async def _preload(context: ServerContext, selector: str) -> None:
    """Make one model resident in the background, after the server is answering.

    ``selector`` is a QCS-internal reference -- a stable library id (what a
    profile stores), a served name, or the model's path -- not a wire model id.

    Failure is logged and nothing else: the daemon is already serving, and a
    preload that cannot run is not a reason to take it down.
    """
    model = context.registry.select(selector)
    if model is None:
        logger.warning("Cannot preload %r: it is not an installed, usable model", selector)
        return
    try:
        async with context.supervisor.lease(model):
            pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Preload of %s failed: %s", model.slug, exc)


def create_app(
    *,
    cache_max_entries: int = DEFAULT_MAX_ENTRIES,
    cache_max_bytes: int = DEFAULT_MAX_BYTES,
    host: str = "127.0.0.1",
    port: int = 8123,
    defaults: ServerDefaults | None = None,
    preload: str | None = None,
    idle_timeout_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES,
) -> FastAPI:
    """Build the daemon.

    No model is required. The daemon serves its catalogue and management plane
    immediately and loads weights when a request names them, so pressing Start
    never means waiting half a minute for a socket to appear.

    ``preload`` optionally names a slug to make resident in the background once
    the server is already answering. It is a convenience, never a prerequisite.
    """
    # Fresh every start, so a token recovered from a stale runtime file is
    # useless against this server (D1).
    migrate_app_support_root()
    management_token = mint_token()
    engine = MlxEngine(cache_max_entries=cache_max_entries, cache_max_bytes=cache_max_bytes)
    registry = ModelRegistry()
    renderer = HarmonyRenderer()
    context = ServerContext(
        engine,
        registry,
        renderer,
        defaults or ServerDefaults(),
        idle_timeout_minutes=idle_timeout_minutes,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Nothing here may block: every statement runs before uvicorn starts
        # accepting connections. Loading weights here is what used to leave the
        # port refusing connections for ~29 s with no way to see why.
        refresh_registry(context)

        write_runtime_state(
            RuntimeState(
                pid=os.getpid(),
                host=host,
                port=port,
                model=None,
                management_token=management_token,
                started_at=context.started_at,
            )
        )

        warm: asyncio.Task[None] | None = None
        if preload:
            warm = asyncio.create_task(_preload(context, preload))

        try:
            yield
        finally:
            context.supervisor.begin_stopping()
            if warm is not None:
                warm.cancel()
            # Removed first: a runtime file outliving its server sends every
            # reader to a dead endpoint.
            clear_runtime_state()
            await engine.unload()
            engine.shutdown()

    app = FastAPI(title="Quantum-Codex-OSS-MLX-Server", lifespan=lifespan)
    app.state.context = context
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(build_management_router(token=management_token, context=context))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        # Every value here is a plain attribute read. Nothing is submitted to
        # the worker, so health answers in milliseconds while a 60 GB load or a
        # long generation is occupying it -- which is exactly when something is
        # watching.
        active, queued = engine.queue_depth
        lifecycle = context.supervisor.snapshot()
        return {
            # `ok` means the daemon is answering, which is now independent of
            # whether any weights are resident. A daemon with no model is a
            # normal, healthy state, not a degraded one.
            "status": "ok",
            "lifecycle": lifecycle.as_dict(),
            "model": lifecycle.model,
            "uptime_seconds": round(time.time() - context.started_at, 1),
            "active_requests": active,
            "queued_requests": queued,
            # Cache behaviour is a product feature, not an internal detail: it
            # has to be visible to be trusted (cahier 63.3). Counters only --
            # nothing here reveals prompt content.
            "prompt_cache": engine.cache_snapshot.as_dict(),
        }

    @app.get("/v1/models")
    async def list_models(client_version: str | None = None) -> dict[str, Any]:
        """Model metadata, in the schema Codex's model manager expects (D5).

        ``client_version`` is what Codex appends to the URL. It is recorded
        rather than ignored, so a future contract change can be versioned
        against a version actually observed in the wild instead of a guess.
        """
        if client_version:
            logger.debug("models requested by codex client_version=%s", client_version)
        return build_models_response(registry.all())

    @app.post("/v1/responses")
    async def create_response(request: Request) -> Response:
        # Stamped before anything can fail, so even a rejected request is
        # traceable across the channels that examined it — and, since the engine
        # carries it across the worker boundary, into inference and cache lines.
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        set_request_id(request_id)
        body = await _read_json(request)
        parsed = parse_request(body)

        # Fail closed on anything undeclared, so a client evolution surfaces as
        # a clear error rather than as quietly changed behaviour (D4).
        for key in parsed.unknown_fields:
            raise invalid_request(f"Unsupported request field: {key}", param=key)

        for problem in (
            CAPABILITIES.check_include(parsed.include),
            CAPABILITIES.check_tool_choice(parsed.tool_choice),
            CAPABILITIES.check_text_format(parsed.text),
        ):
            if problem is not None:
                raise invalid_request(problem.message, param=problem.param)

        forwarded_tools: list[dict[str, Any]] = []
        dropped_tools: tuple[Any, ...] = ()
        if parsed.tools:
            plan = CAPABILITIES.plan_tools(parsed.tools)
            if plan.problem is not None:
                raise invalid_request(plan.problem.message, param=plan.problem.param)
            dropped_tools = plan.dropped
            log_dropped_tools(plan.dropped)
            # Only the tools that survived reach the model, and only they are
            # echoed back. A tool the server cannot route must never appear in
            # the rendered prompt (cahier 5.5).
            forwarded_tools = [tool for _, tool in plan.kept]
            parsed = parsed.model_copy(update={"tools": forwarded_tools})
            logger.debug(
                "tools declared=%d forwarded=%s",
                len(plan.kept) + len(plan.dropped),
                [tool.get("name") for tool in forwarded_tools],
            )

        model = _resolve_model(context, registry, parsed.model)
        turn = to_canonical_turn(parsed, default_effort=model.default_reasoning_effort)

        if not model.supports_effort(turn.reasoning_effort):
            supported = ", ".join(effort.value for effort in model.reasoning_efforts)
            raise invalid_request(
                f"Unsupported reasoning effort: {turn.reasoning_effort.value}. "
                f"Supported: {supported}.",
                param="reasoning.effort",
            )

        prompt_tokens = renderer.render(turn)
        max_output = _resolve_max_output(
            requested=turn.max_output_tokens,
            prompt_length=len(prompt_tokens),
            context_window=model.context_window,
            # This model's own budget when it has one, else the daemon's.
            server_default=model.max_output_tokens or context.defaults.max_output_tokens,
        )
        record = context.diagnostics.begin(
            request_id=request_id,
            model=model.slug,
            streamed=parsed.stream,
            reasoning_effort=turn.reasoning_effort.value,
        )
        record.tools_declared = len(parsed.tools or []) + len(dropped_tools)
        record.tools_forwarded = len(forwarded_tools)

        generate_args = {
            "stop_tokens": renderer.stop_tokens,
            "max_tokens": max_output,
            "temperature": (
                turn.temperature
                if turn.temperature is not None
                else (
                    model.temperature
                    if model.temperature is not None
                    else context.defaults.temperature
                )
            ),
            "top_p": (
                turn.top_p
                if turn.top_p is not None
                else (model.top_p if model.top_p is not None else context.defaults.top_p)
            ),
            # Recorded, never trusted: prefix reuse is decided by comparing the
            # actual tokens, not by believing the client's key.
            "cache_hint": parsed.prompt_cache_key,
        }

        # The lease makes the named model resident and keeps it that way until
        # this request is done. Acquired before the response begins so a busy
        # conflict or a load failure is an HTTP error the client can read,
        # rather than a stream that opens and then dies.
        async with AsyncExitStack() as stack:
            try:
                await stack.enter_async_context(context.supervisor.lease(model))
            except ModelBusyError as exc:
                context.diagnostics.finish(record, Outcome.FAILED, error=str(exc))
                raise ApiError(
                    str(exc), status_code=409, error_type="model_busy", param="model"
                ) from exc
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                context.diagnostics.finish(record, Outcome.FAILED, error=str(exc))
                raise ApiError(
                    f"Model `{model.slug}` could not be loaded: {exc}",
                    status_code=503,
                    error_type="server_error",
                    param="model",
                ) from exc

            if parsed.stream:
                # Ownership of the lease moves to the generator: the weights
                # must stay resident for as long as tokens are still being
                # produced, which is long after this function returns.
                lease = stack.pop_all()
                return StreamingResponse(
                    _hold(
                        lease,
                        _stream_response(
                            engine=engine,
                            renderer=renderer,
                            model=model,
                            turn=turn,
                            prompt_tokens=prompt_tokens,
                            generate_args=generate_args,
                            tools=forwarded_tools,
                            diagnostics=context.diagnostics,
                            record=record,
                        ),
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
                )

            try:
                outcome = await engine.generate(prompt_tokens, **generate_args)
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                context.diagnostics.finish(record, Outcome.FAILED, error=str(exc))
                raise

            generation = parse_completion(outcome.tokens)
            router = ToolRouter(turn)
            resolved: list[ToolCall] = []
            for call in generation.tool_calls:
                name, namespace = router.resolve(call.name, call.namespace)
                resolved.append(
                    ToolCall(
                        call_id=f"call_{uuid.uuid4().hex[:24]}",
                        name=name,
                        arguments=call.arguments,
                        namespace=namespace,
                    )
                )
            result = _build_result(
                renderer,
                generation.text,
                generation.reasoning,
                tuple(resolved),
                outcome,
            )
            _log_result(model.slug, result, outcome)
            completion_error = _completion_error(result, outcome)
            _record_result(
                context.diagnostics,
                record,
                result,
                outcome,
                renderer=renderer,
                completion_error=completion_error,
            )
            if completion_error is not None:
                raise ApiError(
                    completion_error,
                    status_code=500,
                    error_type="server_error",
                    code=EMPTY_COMPLETION_CODE,
                )

            return JSONResponse(
                build_response(
                    response_id=new_response_id(),
                    model=model.slug,
                    result=result,
                    instructions=turn.instructions,
                    tools=forwarded_tools,
                )
            )

    return app


async def _hold(lease: AsyncExitStack, frames: AsyncIterator[str]) -> AsyncIterator[str]:
    """Stream ``frames`` while keeping ``lease`` open.

    The weights must stay resident until the last token is emitted. Closing the
    lease when the handler returned would let another request swap the model out
    from under a generation still in progress.
    """
    try:
        async for frame in frames:
            yield frame
    finally:
        await lease.aclose()


async def _stream_response(
    *,
    engine: MlxEngine,
    renderer: HarmonyRenderer,
    model: ServedModel,
    turn: CanonicalTurn,
    prompt_tokens: list[int],
    generate_args: dict[str, Any],
    tools: list[dict[str, Any]],
    diagnostics: Diagnostics,
    record: RequestRecord,
) -> AsyncIterator[str]:
    """Emit one response as SSE.

    Items are opened lazily and closed as soon as the model leaves their
    channel, so `output_item.done` lands while the turn is still running rather
    than all at the end. That is what makes the reasoning item a real item in
    Codex's conversation instead of transient display text.
    """
    stream = ResponseStream(
        response_id=new_response_id(),
        model=model.slug,
        created_at=int(time.time()),
        tools=tools,
    )
    parser = StreamingParser()
    router = ToolRouter(turn)

    yield stream.created()
    yield stream.in_progress()

    completed_items: list[dict[str, Any]] = []
    reasoning_segments: list[str] = []
    text_segments: list[str] = []
    tool_calls: list[ToolCall] = []

    # The open item is identified by (channel, tool target): the model can leave
    # `commentary` for a tool call and come back, and those are different items
    # even though the channel is the same.
    open_kind: tuple[str, tuple[str, str | None] | None] | None = None
    open_item_id: str | None = None
    open_call_id: str | None = None
    buffer: list[str] = []
    outcome: GenerationOutcome | None = None

    def close_open_item() -> str | None:
        nonlocal open_kind, open_item_id, open_call_id, buffer
        if open_item_id is None or open_kind is None:
            return None
        channel, target = open_kind
        text = "".join(buffer)

        if target is not None:
            name, namespace = target
            call = ToolCall(
                call_id=open_call_id or "", name=name, arguments=text, namespace=namespace
            )
            tool_calls.append(call)
            item = function_call_item(
                open_item_id,
                call_id=call.call_id,
                name=name,
                arguments=text,
                namespace=namespace,
            )
        elif channel == ANALYSIS:
            item = reasoning_item(open_item_id, text)
            reasoning_segments.append(text)
        else:
            item = message_item(open_item_id, text)
            text_segments.append(text)

        completed_items.append(item)
        frame = stream.item_done(item)
        open_kind, open_item_id, open_call_id, buffer = None, None, None, []
        return frame

    try:
        async for produced in engine.generate_stream(prompt_tokens, **generate_args):
            if isinstance(produced, GenerationOutcome):
                outcome = produced
                break

            if isinstance(produced, PrefillProgress):
                # Emitted while the prompt is still being evaluated, which for a
                # long Codex conversation is the part that takes seconds.
                yield stream.heartbeat(produced.processed, produced.total)
                continue

            parsed_delta = parser.push(produced)
            if parsed_delta is None:
                continue
            channel, delta = parsed_delta

            # Resolved before the item is announced: the namespace is part of
            # the `output_item.added` payload, and a client cannot be told the
            # route changed after the fact.
            target = parser.tool_target if channel == COMMENTARY else None
            if target is not None:
                target = router.resolve(*target)

            # Commentary with no recipient is the model thinking out loud to the
            # client rather than calling anything. It is neither an answer nor a
            # tool call, so it is not turned into an output item.
            if channel not in (ANALYSIS, FINAL) and target is None:
                continue

            kind = (channel, target)
            if kind != open_kind:
                closing = close_open_item()
                if closing is not None:
                    yield closing

                open_kind = kind
                if target is not None:
                    name, namespace = target
                    open_item_id = f"fc_{uuid.uuid4().hex}"
                    # The call id is minted here. Harmony has no concept of one;
                    # the Responses API needs it to pair the client's result
                    # with this call, so it must be stable from the moment the
                    # item is announced.
                    open_call_id = f"call_{uuid.uuid4().hex[:24]}"
                    item = function_call_item(
                        open_item_id,
                        call_id=open_call_id,
                        name=name,
                        arguments="",
                        namespace=namespace,
                    )
                elif channel == ANALYSIS:
                    open_item_id = f"rs_{uuid.uuid4().hex}"
                    item = reasoning_item(open_item_id, "")
                else:
                    open_item_id = f"msg_{uuid.uuid4().hex}"
                    item = message_item(open_item_id, "")
                yield stream.item_added(item)

            buffer.append(delta)
            assert open_item_id is not None
            if target is not None:
                yield stream.function_arguments_delta(open_item_id, open_call_id or "", delta)
            elif channel == ANALYSIS:
                yield stream.reasoning_delta(open_item_id, delta)
            else:
                yield stream.text_delta(open_item_id, delta)

        closing = close_open_item()
        if closing is not None:
            yield closing

    except GeneratorExit:
        # The client went away mid-stream. Recorded rather than swallowed: a
        # disconnect is an outcome, and one that explains a truncated session.
        diagnostics.finish(record, Outcome.CANCELLED, error="client disconnected")
        raise
    except Exception as exc:  # noqa: BLE001 - the client must learn the stream died
        logger.exception("generation failed mid-stream")
        diagnostics.finish(record, Outcome.FAILED, error=str(exc))
        yield stream.failed(message=str(exc))
        yield stream.done()
        return

    if outcome is None:
        diagnostics.finish(record, Outcome.FAILED, error="generation produced no result")
        yield stream.failed(message="Generation produced no result.")
        yield stream.done()
        return

    result = _build_result(
        renderer,
        "".join(text_segments),
        tuple(reasoning_segments),
        tuple(tool_calls),
        outcome,
    )
    _log_result(model.slug, result, outcome, streamed=True)
    completion_error = _completion_error(result, outcome)
    _record_result(
        diagnostics,
        record,
        result,
        outcome,
        renderer=renderer,
        completion_error=completion_error,
    )

    if completion_error is not None:
        yield stream.failed(
            message=completion_error,
            error_type="server_error",
            code=EMPTY_COMPLETION_CODE,
        )
        yield stream.done()
        return

    yield stream.completed(output=completed_items, usage=build_usage(result))
    yield stream.done()


def _build_result(
    renderer: HarmonyRenderer,
    text: str,
    reasoning: tuple[str, ...],
    tool_calls: tuple[ToolCall, ...],
    outcome: GenerationOutcome,
) -> CanonicalTurnResult:
    # A turn that ended at a tool call is reported as such, not as a plain stop:
    # the distinction is what tells a client the turn is waiting on it.
    finish_reason = (
        FinishReason.TOOL_CALL
        if tool_calls and outcome.finish_reason is FinishReason.STOP
        else outcome.finish_reason
    )
    return CanonicalTurnResult(
        text=text,
        reasoning=reasoning,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=outcome.input_tokens,
            output_tokens=len(outcome.tokens),
            reasoning_tokens=_count_reasoning_tokens(renderer, reasoning),
            cached_tokens=outcome.cached_tokens,
        ),
        finish_reason=finish_reason,
        timing=outcome.timing,
    )


def _record_result(
    diagnostics: Diagnostics,
    record: RequestRecord,
    result: CanonicalTurnResult,
    outcome: GenerationOutcome,
    *,
    renderer: HarmonyRenderer,
    completion_error: str | None = None,
) -> None:
    """Close a diagnostic record from what actually happened.

    Only counts, timings and tool identities. No prompt, no reasoning text, no
    tool arguments — this describes execution, not conversation.
    """
    record.input_tokens = result.usage.input_tokens
    record.cached_tokens = result.usage.cached_tokens
    record.output_tokens = result.usage.output_tokens
    record.reasoning_tokens = result.usage.reasoning_tokens
    record.queue_wait_seconds = outcome.queue_wait_seconds
    if outcome.timing is not None:
        record.prefill_seconds = outcome.timing.prefill_seconds
        record.decode_seconds = outcome.timing.decode_seconds
    record.tool_calls = [
        ToolCallRecord(name=call.name, namespace=call.namespace) for call in result.tool_calls
    ]
    record.terminal_token_class = renderer.terminal_token_class(outcome.stop_token_id)
    record.had_reasoning = any(segment.strip() for segment in result.reasoning)
    record.had_tool_call = bool(result.tool_calls)
    record.had_final_output = bool(result.text.strip())
    record.empty_completion_detected = completion_error is not None

    diagnostics.finish(
        record,
        (
            Outcome.FAILED
            if completion_error is not None
            else {
                FinishReason.LENGTH: Outcome.INCOMPLETE,
                FinishReason.CANCELLED: Outcome.CANCELLED,
            }.get(result.finish_reason, Outcome.COMPLETED)
        ),
        error=completion_error,
        finish_reason=result.finish_reason.value,
        last_channel=("commentary" if result.tool_calls else "final" if result.text else "analysis"),
    )


def _completion_error(
    result: CanonicalTurnResult, outcome: GenerationOutcome
) -> str | None:
    """Reject the terminal shape that masquerades as success.

    Length, cancellation and engine errors already have protocol outcomes that
    say what happened. A turn that simply *stopped*, carrying neither of the two
    things an assistant turn can end with -- a final answer or a tool call --
    has none: reporting it as completed is what puts `last_agent_message=null`
    in front of a client that has no way to tell it apart from a real answer.

    Reasoning is not part of the test. It is recorded (`had_reasoning`) because
    it distinguishes the observed incident from an empty generation, but a turn
    that produced nothing at all is no more complete than one that thought and
    then produced nothing.
    """
    if (
        outcome.finish_reason is FinishReason.STOP
        and not result.tool_calls
        and not result.text.strip()
    ):
        return REASONING_ONLY_COMPLETION
    return None


def _log_result(
    slug: str, result: CanonicalTurnResult, outcome: GenerationOutcome, *, streamed: bool = False
) -> None:
    logger.info(
        "response model=%s stream=%s in=%d cached=%d out=%d reasoning=%d "
        "prefill=%.2fs decode=%.1f tok/s finish=%s",
        slug,
        streamed,
        result.usage.input_tokens,
        result.usage.cached_tokens,
        result.usage.output_tokens,
        result.usage.reasoning_tokens,
        outcome.timing.prefill_seconds,
        outcome.timing.decode_tokens_per_second(result.usage.output_tokens),
        result.finish_reason.value,
    )


async def _read_json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001 - any parse failure is one client error
        raise invalid_request("Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise invalid_request("Request body must be a JSON object.")
    return body


def _resolve_model(
    context: ServerContext, registry: ModelRegistry, requested: str | None
) -> ServedModel:
    """Resolve the model a request asked for, against what is installed.

    The catalogue is what is *installed and usable*, not what is resident: the
    client chooses, and the named model is loaded on demand. A request naming a
    model that is not installed is refused rather than served by whatever
    happens to be in memory — answering as another model is a correctness
    problem, not a convenience.
    """
    available = registry.all()
    if not available:
        # The library may have gained a model since startup; look again before
        # telling a user they have none.
        refresh_registry(context)
        available = registry.all()

    if not available:
        raise ApiError(
            "No usable GPT-OSS model is installed. Import or download one, then retry.",
            status_code=503,
            error_type="server_error",
        )

    if requested is None:
        # No slug: prefer what is already resident, so an unspecified request
        # never triggers a load — or a switch away from a warm model.
        current = context.supervisor.current
        return current if current is not None else available[0]

    model = registry.get(requested)
    if model is None:
        refresh_registry(context)
        model = registry.get(requested)

    if model is None:
        served = ", ".join(candidate.slug for candidate in registry.all())
        raise invalid_request(
            f"Model `{requested}` is not installed on this server. Available: {served}.",
            param="model",
        )
    return model


def _resolve_max_output(
    *,
    requested: int | None,
    prompt_length: int,
    context_window: int,
    server_default: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> int:
    """How many tokens may be generated, given what the prompt already used.

    Overflow is refused here rather than absorbed by a rotating KV cache. A
    rotating cache would keep answering while silently dropping the oldest part
    of the conversation, which is indistinguishable from success until the model
    contradicts something it can no longer see.
    """
    remaining = context_window - prompt_length
    if remaining <= 0:
        raise context_overflow(
            f"The prompt is {prompt_length} tokens but the context window is "
            f"{context_window}. Reduce the input.",
            param="input",
        )

    if requested is None:
        return min(server_default, remaining)

    if requested > remaining:
        raise context_overflow(
            f"`max_output_tokens` is {requested} but only {remaining} tokens remain in the "
            f"{context_window}-token context window after a {prompt_length}-token prompt.",
            param="max_output_tokens",
        )
    return requested


def _count_reasoning_tokens(renderer: HarmonyRenderer, reasoning: tuple[str, ...]) -> int:
    """Real token count for the analysis channel.

    Counted with the tokenizer, never estimated from character length
    (cahier 21). This measures the reasoning *text*, so it excludes the channel
    framing tokens -- which is what the Responses API reports.
    """
    if not reasoning:
        return 0
    encoding = renderer.encoding
    return sum(len(encoding.encode(text, allowed_special=set())) for text in reasoning)


def serve(
    *,
    model: str | None = None,
    served_model_name: str | None = None,
    host: str,
    port: int,
    context_length: int = 131072,
    cache_max_entries: int = DEFAULT_MAX_ENTRIES,
    cache_max_bytes: int = DEFAULT_MAX_BYTES,
    log_level: str = "INFO",
    defaults: ServerDefaults | None = None,
    idle_timeout_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES,
) -> int:
    """Run the daemon.

    ``model`` is optional and only names a model to make resident *after* the
    server is answering. The daemon starts, advertises every installed usable
    model, and loads whichever one a request asks for. Weights are never a
    prerequisite for having a server.
    """
    import uvicorn

    configure_logging(log_level)

    preload: str | None = None
    if model:
        # A path or a slug. A path is checked now, because "the drive is not
        # attached" is worth saying at startup rather than at first request --
        # but it is a warning, not a refusal: the daemon still starts, and the
        # user can attach the drive or pick another model without restarting.
        candidate = Path(model).expanduser()
        if candidate.exists() or candidate.is_absolute():
            volume = volume_for(candidate)
            if not volume.mounted:
                logger.warning(
                    "Not preloading %s: the volume %r holding it is not mounted. "
                    "The server is starting anyway; attach the drive and send a request.",
                    candidate,
                    volume.name,
                )
            elif not candidate.exists():
                logger.warning(
                    "Not preloading %s: the volume is available, so the model directory "
                    "has been moved or removed. The server is starting anyway.",
                    candidate,
                )
            else:
                # The path itself, resolved against the library by
                # `ModelRegistry.select`. Deriving a name from the directory
                # here would be a third opinion about what this model is called,
                # and it would be the wrong one for any model whose served name
                # the user has since changed.
                preload = served_model_name or str(candidate)
        else:
            # A stable library id (what a profile stores) or a served name.
            preload = model

    app = create_app(
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
        host=host,
        port=port,
        defaults=defaults,
        preload=preload,
        idle_timeout_minutes=idle_timeout_minutes,
    )

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower(), access_log=False)
    return 0
