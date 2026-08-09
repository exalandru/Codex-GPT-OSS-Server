"""Logging setup: channels and request correlation.

A Codex session produces interleaved work — protocol validation, Harmony
rendering, inference, cache decisions — and when something goes wrong the useful
question is "what happened to *this* request". Logger names are the channels
(cahier 34), and a request id ties one request's lines together across all of
them:

    quantum_codex.app                 server and protocol
    quantum_codex.codex.capabilities  Codex compatibility decisions
    quantum_codex.harmony             prompt rendering and parsing
    quantum_codex.inference.engine    model lifecycle and generation
    quantum_codex.inference.prompt_cache   prefix reuse

The id travels in a ``ContextVar`` rather than being threaded through call
signatures: it is ambient context, not an argument any of those layers should
have to accept and forward.

One caveat, deliberately visible: work handed to the inference worker runs on
another thread, and a ``ContextVar`` set on the event loop is not visible there.
Worker log lines carry no request id. Threading one through would mean giving
the engine a logging concern it does not otherwise have, so the correlation
stops at the boundary rather than being faked.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Attaches the current request id to every record.

    A filter rather than a formatter detail, so the id is available to any
    handler — including a structured one later — instead of only surviving as
    text in one format string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        return True


def configure(level: str = "INFO") -> None:
    """Install the root handler for the server process."""
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-38s [%(request_id)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))
