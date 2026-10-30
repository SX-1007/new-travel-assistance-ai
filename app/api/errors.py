"""Exception → HTTP response mapping for every router.

Routers let exceptions propagate; ``register_exception_handlers`` installs
one handler per known exception type, so a route never needs its own
``try/except`` around the graph. Starlette picks the handler by the
exception's MRO, so a subclass rule (Gemini rate limit) wins over its
parent's (any Gemini error) without the rules having to be ordered.

The client only ever sees a fixed ``detail`` string, never ``str(exc)``,
which can carry URLs, SQL or provider request ids.

Request validation (422) keeps FastAPI's structured list under ``errors``
but makes ``detail`` one readable sentence: the frontend shows ``detail``
only when it is a string, so a list left users with "failed with 422".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import httpx
import psycopg
import requests
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google.api_core import exceptions as google_errors
from langgraph.errors import GraphRecursionError
from postgrest.exceptions import APIError as SupabaseApiError
from psycopg_pool import PoolTimeout

from app.modules.budget.exceptions import CurrencyProviderError
from app.modules.profile.exceptions import ProfileUnavailable
from app.shared.exceptions import (
    InvalidTurnRequest,
    SessionAccessDenied,
    SessionBusy,
)
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)


class ServiceNotReady(RuntimeError):
    """A dependency the route needs has not finished starting up."""


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    status_code: int
    detail: str
    retry_after_seconds: int | None = None


UNEXPECTED_ERROR: Final = ErrorResponse(
    status.HTTP_500_INTERNAL_SERVER_ERROR,
    "An unexpected error occurred while processing your request.",
)

_STORE_UNAVAILABLE: Final = ErrorResponse(
    status.HTTP_503_SERVICE_UNAVAILABLE,
    "Trip storage is temporarily unavailable. Please retry shortly.",
    retry_after_seconds=5,
)
_UPSTREAM_TIMEOUT: Final = ErrorResponse(
    status.HTTP_504_GATEWAY_TIMEOUT,
    "An upstream service timed out. Please retry.",
)
_UPSTREAM_ERROR: Final = ErrorResponse(
    status.HTTP_502_BAD_GATEWAY,
    "An upstream service returned an error. Please retry.",
)

ERROR_RESPONSES: Final[Mapping[type[BaseException], ErrorResponse]] = {
    # ── Sessions ──
    # 404, not 403: a caller must not learn that someone else's id exists.
    SessionAccessDenied: ErrorResponse(status.HTTP_404_NOT_FOUND, "Trip session not found."),
    SessionBusy: ErrorResponse(
        status.HTTP_409_CONFLICT,
        "Your previous message for this trip is still being processed. "
        "Please wait for it to finish.",
        retry_after_seconds=5,
    ),
    InvalidTurnRequest: ErrorResponse(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "The request is missing a valid session or message.",
    ),
    ServiceNotReady: ErrorResponse(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The service is starting up. Please retry shortly.",
        retry_after_seconds=5,
    ),
    # ── Turn limits ──
    # ``asyncio.TimeoutError`` is ``TimeoutError`` (Python 3.11+): the
    # runtime's per-turn time limit.
    TimeoutError: ErrorResponse(
        status.HTTP_504_GATEWAY_TIMEOUT,
        "The request took too long to complete. Please retry.",
    ),
    GraphRecursionError: ErrorResponse(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "The assistant could not finish this request.",
    ),
    # ── Providers ──
    CurrencyProviderError: ErrorResponse(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Exchange-rate service temporarily unavailable. Please retry shortly.",
        retry_after_seconds=30,
    ),
    google_errors.ResourceExhausted: ErrorResponse(
        status.HTTP_429_TOO_MANY_REQUESTS,
        "The language model is rate limited. Please retry shortly.",
        retry_after_seconds=30,
    ),
    google_errors.DeadlineExceeded: _UPSTREAM_TIMEOUT,
    google_errors.ServiceUnavailable: ErrorResponse(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The language model is temporarily unavailable. Please retry shortly.",
        retry_after_seconds=30,
    ),
    google_errors.GoogleAPIError: _UPSTREAM_ERROR,
    # The providers use ``requests``; the old map only knew ``httpx``.
    requests.Timeout: _UPSTREAM_TIMEOUT,
    requests.RequestException: _UPSTREAM_ERROR,
    httpx.TimeoutException: _UPSTREAM_TIMEOUT,
    httpx.HTTPError: _UPSTREAM_ERROR,
    # ── Storage ──
    psycopg.OperationalError: _STORE_UNAVAILABLE,
    PoolTimeout: _STORE_UNAVAILABLE,
    SupabaseApiError: _STORE_UNAVAILABLE,
    ProfileUnavailable: ErrorResponse(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Your profile could not be loaded or saved. Please retry shortly.",
        retry_after_seconds=5,
    ),
}


def error_response_for(exc: BaseException) -> ErrorResponse:
    """The mapped response for ``exc`` by its nearest mapped base class."""

    for cls in type(exc).__mro__:
        mapped = ERROR_RESPONSES.get(cls)

        if mapped is not None:
            return mapped

    return UNEXPECTED_ERROR


_MAX_VALIDATION_MESSAGES: Final = 3
# Request containers, not field names, in an error's location.
_LOCATION_ROOTS: Final = frozenset({"body", "query", "path", "header", "cookie"})


def validation_message(errors: list[Any]) -> str:
    """``field: message`` for the first few errors, as one sentence."""

    parts: list[str] = []

    for error in errors[:_MAX_VALIDATION_MESSAGES]:
        location = [str(part) for part in error.get("loc", ()) if part not in _LOCATION_ROOTS]
        message = str(error.get("msg", "is invalid")).removeprefix("Value error, ")
        parts.append(f"{'.'.join(location)}: {message}" if location else message)

    return "; ".join(parts) or "The request is invalid."


def register_exception_handlers(app: FastAPI) -> None:
    for exc_type in ERROR_RESPONSES:
        app.add_exception_handler(exc_type, _handle_mapped)

    app.add_exception_handler(RequestValidationError, _handle_validation)

    # Starlette hands this one to its server-error middleware, which still
    # re-raises after responding so the server logs the traceback.
    app.add_exception_handler(Exception, _handle_unexpected)


async def _handle_mapped(request: Request, exc: Exception) -> JSONResponse:
    mapped = error_response_for(exc)
    _log(request, exc, mapped)
    return _json(mapped)


async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()

    # ``input`` would echo the submitted values back; ``ctx`` can hold
    # exception objects that are not JSON.
    return JSONResponse(
        {
            "detail": validation_message(list(errors)),
            "errors": [
                {"loc": list(error.get("loc", ())), "msg": error.get("msg", ""), "type": error.get("type", "")}
                for error in errors
            ],
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    _log(request, exc, UNEXPECTED_ERROR)
    return _json(UNEXPECTED_ERROR)


def _json(mapped: ErrorResponse) -> JSONResponse:
    headers = (
        {"Retry-After": str(mapped.retry_after_seconds)}
        if mapped.retry_after_seconds is not None
        else None
    )
    return JSONResponse({"detail": mapped.detail}, status_code=mapped.status_code, headers=headers)


def _log(request: Request, exc: BaseException, mapped: ErrorResponse) -> None:
    # The exception class name only: its message may hold provider data.
    log_event(
        logger,
        "api.error",
        context=LogContext(request_id=getattr(request.state, "request_id", None)),
        stage=request.url.path,
        outcome="error" if mapped.status_code >= 500 else "rejected",
        reason=type(exc).__name__,
        issue_codes=[f"http.{mapped.status_code}"],
        level=logging.ERROR if mapped is UNEXPECTED_ERROR else None,
    )


__all__ = [
    "ERROR_RESPONSES",
    "ErrorResponse",
    "ServiceNotReady",
    "UNEXPECTED_ERROR",
    "error_response_for",
    "register_exception_handlers",
    "validation_message",
]
