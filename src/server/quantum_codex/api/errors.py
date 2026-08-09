"""The OpenAI error envelope.

Clients read `error.type`, `error.param` and `error.message`, so error shape is
part of the protocol contract, not a presentation detail. Every refusal in the
server goes through :class:`ApiError` and comes out in this one shape.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """An error the client should see, with its protocol shape attached."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_payload(self) -> dict[str, object]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def invalid_request(message: str, *, param: str | None = None, code: str | None = None) -> ApiError:
    return ApiError(message, status_code=400, param=param, code=code)


def context_overflow(message: str, *, param: str | None = None) -> ApiError:
    """The prompt does not fit.

    A distinct code because it is the one client-recoverable limit error: Codex
    compacts its history in response to it, so it must be distinguishable from
    an ordinary malformed request.
    """
    return ApiError(message, status_code=400, param=param, code="context_length_exceeded")


def server_error(message: str) -> ApiError:
    return ApiError(message, status_code=500, error_type="server_error")


def service_unavailable(message: str) -> ApiError:
    return ApiError(message, status_code=503, error_type="server_error")


async def api_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())
