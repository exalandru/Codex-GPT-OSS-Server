"""The management plane (D1).

Separate from ``/v1`` on purpose. ``/v1`` is a protocol contract owed to Codex;
``/internal`` is this project's own operational surface, free to change with the
CLI and the GUI that consume it and owed to nobody else.

Every route requires a bearer token minted at startup and written to the
owner-readable runtime file. The server binds loopback, so the token is not
about the network — it is about the other processes on the same machine. A
loopback port is reachable by every user process, and this surface can clear a
cache and read operational state.

Failures are deliberately indistinguishable: a missing header, a malformed one
and a wrong token all produce the same 401, so the endpoint cannot be used to
learn whether a token is close to right.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Header, Request

from ..config import ConfigError
from ..library import MANAGER, DownloadError, load_registry, mounted_volumes, save_registry
from ..lifecycle import ModelInUseError, UnloadReason
from .errors import ApiError, invalid_request

logger = logging.getLogger(__name__)


def mint_token() -> str:
    """A fresh management token.

    New on every start, so a token recovered from an old runtime file is useless
    against the next server.
    """
    return secrets.token_urlsafe(32)


def _authorise(expected: str, authorization: str | None) -> None:
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    # Constant-time even though a local attacker has better options: the cost is
    # nil and the alternative invites a timing oracle the moment this moves.
    if not hmac.compare_digest(provided, expected):
        # Logged because a rejected management request means something on this
        # machine is probing the control surface, or a client is holding a token
        # from a previous server. Both are worth seeing.
        logger.warning("management request rejected: bad or missing token")
        raise ApiError(
            "Management endpoints require the token from the runtime file.",
            status_code=401,
            error_type="unauthorized",
        )


def build_router(*, token: str, context: Any) -> APIRouter:
    """Routes for one running server.

    ``context`` is the :class:`ServerContext`; passed in rather than imported to
    keep this module free of an import cycle with the app.
    """
    router = APIRouter(prefix="/internal", tags=["management"])

    @router.get("/status")
    async def status(
        request: Request, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _authorise(token, authorization)
        engine = context.engine
        active, queued = engine.queue_depth
        lifecycle = context.supervisor.snapshot()
        current = context.supervisor.current
        installed = context.registry.all()

        # Residency is the supervisor's answer; the *shape* of what is resident
        # is the engine's. Reading `engine.loaded_model` on its own made this
        # block a second opinion about whether anything is loaded at all, and
        # the two could disagree -- `lifecycle.model` null beside a populated
        # `model` object. Gated, each question has one owner.
        loaded = engine.loaded_model if current is not None else None

        # Nothing below touches the worker thread. Status must answer while a
        # model is loading, because that is precisely when a user is asking
        # whether anything is happening.
        return {
            "server": {
                # The daemon's own state, no longer conflated with the model's.
                # A running daemon holding no weights is normal.
                "state": "running",
                "lifecycle": lifecycle.state.value,
                "uptime_seconds": round(time.time() - context.started_at, 1),
                "endpoint": str(request.base_url).rstrip("/"),
            },
            "model": (
                {
                    "served_name": loaded.served_name,
                    "path": loaded.path,
                    "quantization": loaded.quantization,
                    "context_length": loaded.context_length,
                    "layers": loaded.num_hidden_layers,
                    # What was *applied*, measured at load, not the setting that
                    # asked for it. `installed_models` below reports the
                    # setting; the two are deliberately different questions,
                    # because an adapter that applied to nothing would answer
                    # the first one wrongly and only the second one right.
                    "adapter": (
                        {
                            "path": loaded.adapter.path,
                            "fine_tune_type": loaded.adapter.fine_tune_type,
                            "applied_tensors": loaded.adapter.applied_tensors,
                            "adapter_tensors": loaded.adapter.adapter_tensors,
                        }
                        if loaded.adapter
                        else None
                    ),
                }
                if loaded
                else None
            ),
            # What the model is doing, and for how long. `elapsed_seconds` is
            # measured, never a prediction of what remains.
            "lifecycle": lifecycle.as_dict(),
            "installed_models": [
                {
                    "slug": model.slug,
                    "display_name": model.display_name,
                    "context_window": model.context_window,
                    "quantization": model.quantization,
                    # Configured, not necessarily resident, and not necessarily
                    # resident *with* it -- see `model.adapter` above.
                    "adapter_path": model.adapter_path,
                    "loaded": current is not None and current.slug == model.slug,
                }
                for model in installed
            ],
            "capabilities": (
                {
                    "reasoning_efforts": [e.value for e in installed[0].reasoning_efforts],
                    "default_reasoning_effort": installed[0].default_reasoning_effort.value,
                    "context_window": installed[0].context_window,
                    "effective_context_window": installed[0].effective_context_window,
                    "supports_tools": installed[0].supports_tools,
                    "supports_parallel_tool_calls": installed[0].supports_parallel_tool_calls,
                }
                if installed
                else None
            ),
            "inference": {"active_requests": active, "queued_requests": queued},
            "prompt_cache": engine.cache_snapshot.as_dict(),
        }

    @router.post("/model/unload")
    async def unload_model(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        """Release the resident model without stopping the daemon.

        The same supervisor operation the idle timer uses, so a model released
        by hand and one released by the timer leave identical state — a second
        implementation here would eventually clear a different set of things.

        Refused rather than queued while the model is serving something. A
        request that arrives after the button is pressed keeps the weights it is
        reading, and cancelling live inference to satisfy an unload would be a
        destructive surprise nobody asked for.
        """
        _authorise(token, authorization)
        try:
            released = await context.supervisor.unload(UnloadReason.MANUAL)
        except ModelInUseError as exc:
            raise ApiError(
                str(exc), status_code=409, error_type="model_in_use", param="model"
            ) from exc
        # `released` distinguishes "it is gone now" from "there was nothing to
        # release", so pressing the button twice reads as idempotent rather than
        # as two unloads.
        return {"released": released, "lifecycle": context.supervisor.snapshot().as_dict()}

    @router.get("/cache")
    async def cache_stats(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _authorise(token, authorization)
        return (await context.engine.cache_stats()).as_dict()

    @router.delete("/cache")
    async def clear_cache(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _authorise(token, authorization)
        before = await context.engine.cache_stats()
        await context.engine.clear_cache()
        # Report what was dropped rather than a bare acknowledgement: "cleared"
        # alone cannot be told apart from "there was nothing to clear".
        return {
            "cleared_entries": before.entries,
            "cleared_bytes": before.bytes,
            "prompt_cache": (await context.engine.cache_stats()).as_dict(),
        }

    # -- model library ------------------------------------------------------
    #
    # The library is disk state, independent of whatever model the engine has
    # loaded. A server happily serving the 20B can hold a library entry for a
    # 120B on a drive that is currently unplugged, and both facts are true.

    @router.get("/models")
    async def list_models(
        scan: bool = False, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _authorise(token, authorization)
        registry = load_registry()
        if scan and registry.discover():
            save_registry(registry)
        return {
            "roots": registry.roots,
            "models": [report.as_dict() for report in registry.report()],
            "volumes": [
                {
                    "name": volume.name,
                    "mount_point": str(volume.mount_point) if volume.mount_point else None,
                    "free_bytes": volume.free_bytes,
                    "total_bytes": volume.total_bytes,
                }
                for volume in mounted_volumes()
            ],
        }

    @router.post("/models")
    async def import_model(
        payload: Annotated[dict[str, Any], Body()],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _authorise(token, authorization)
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            raise invalid_request("`path` is required.", param="path")

        registry = load_registry()
        try:
            entry = registry.add(path)
        except ConfigError as exc:
            # The directory the user just chose is unusable, and saying why is
            # the whole value of validating at import time.
            raise invalid_request(str(exc), param="path") from exc
        save_registry(registry)
        return {"imported": entry.path}

    @router.delete("/models")
    async def forget_model(
        path: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _authorise(token, authorization)
        registry = load_registry()
        try:
            entry = registry.forget(path)
        except ConfigError as exc:
            raise invalid_request(str(exc), param="path") from exc
        save_registry(registry)
        # Stated explicitly in the response: forgetting is not deleting, and a
        # client should be able to tell the user so without guessing.
        return {"forgotten": entry.path, "files_removed": False}

    @router.post("/models/roots")
    async def add_root(
        payload: Annotated[dict[str, Any], Body()],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _authorise(token, authorization)
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            raise invalid_request("`path` is required.", param="path")
        registry = load_registry()
        registry.add_root(path)
        save_registry(registry)
        return {"roots": registry.roots}

    @router.delete("/models/roots")
    async def remove_root(
        path: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _authorise(token, authorization)
        registry = load_registry()
        try:
            registry.remove_root(path)
        except ConfigError as exc:
            raise invalid_request(str(exc), param="path") from exc
        save_registry(registry)
        return {"roots": registry.roots}

    # -- diagnostics --------------------------------------------------------

    @router.get("/requests")
    async def request_diagnostics(
        limit: int = 50, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """How recent requests executed, and the aggregates over them.

        Execution only: no prompt, no reasoning text, no tool arguments. The
        interpretation lives here too — medians, ratios, throughput — so no
        client has to compute its own and disagree.
        """
        _authorise(token, authorization)
        return context.diagnostics.as_dict(limit=max(1, min(limit, 200)))

    # -- downloads ----------------------------------------------------------

    @router.get("/downloads")
    async def download_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _authorise(token, authorization)
        active = MANAGER.active
        last = MANAGER.last
        return {
            "active": active.as_dict() if active else None,
            "last": last.as_dict() if last else None,
        }

    @router.post("/downloads")
    async def start_download(
        payload: Annotated[dict[str, Any], Body()],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _authorise(token, authorization)
        repo = payload.get("repo")
        if not isinstance(repo, str) or not repo:
            raise invalid_request("`repo` is required.", param="repo")
        destination = payload.get("destination")

        try:
            progress = MANAGER.start(
                repo, destination=Path(destination) if isinstance(destination, str) else None
            )
        except DownloadError as exc:
            # Refusals here are actionable — a malformed id, no space, a
            # download already running — so the reason is the response.
            raise invalid_request(str(exc), param="repo") from exc
        return progress.as_dict()

    @router.delete("/downloads")
    async def cancel_download(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _authorise(token, authorization)
        try:
            progress = MANAGER.cancel()
        except DownloadError as exc:
            raise invalid_request(str(exc)) from exc
        return progress.as_dict()

    return router
