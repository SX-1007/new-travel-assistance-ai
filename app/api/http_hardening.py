"""Two pure ASGI middlewares that harden every HTTP response.

``RequestBodyLimit``: neither Starlette nor uvicorn limits a request body,
so a client could make the server read and parse megabytes before the
schemas reject them. A ``Content-Length`` over the limit is refused (413)
before anything is read; a body sent without one (chunked) is counted as it
streams in and refused as soon as it passes the limit. The header alone is
never trusted: a client can send more than it declares.

``SecurityHeaders``: what OWASP's REST and HTTP-headers cheat sheets ask of
a JSON API. Responses are never stored by browsers or shared caches
(``Cache-Control: no-store``: they hold per-user trips and profiles), never
sniffed as another type, never framed, send no referrer, and, in
production (always HTTPS), tell browsers to use HTTPS only. A header a route
set itself is kept. The development API docs are HTML that loads scripts,
so they get no Content-Security-Policy.

Pure ASGI (not ``BaseHTTPMiddleware``) so streamed bodies are not buffered
and the request's ``receive`` channel reaches the route unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


# The largest real request is a chat message of 8,192 characters. JSON may
# escape each one as "\uXXXX\uXXXX" (12 bytes for an emoji): 96 KiB, plus
# a few small fields.
MAX_REQUEST_BODY_BYTES: Final = 128 * 1024

_TOO_LARGE: Final = "The request is too large."
_DOCS_PATHS: Final = ("/docs", "/redoc")

_SECURITY_HEADERS: Final[tuple[tuple[str, str], ...]] = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
)
_API_CSP: Final = "default-src 'none'; frame-ancestors 'none'"
_HSTS: Final = "max-age=31536000; includeSubDomains"


class RequestBodyLimit:
    def __init__(self, app: ASGIApp, *, max_bytes: int = MAX_REQUEST_BODY_BYTES) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")

        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)

        if declared is not None and declared > self.max_bytes:
            await JSONResponse({"detail": _TOO_LARGE}, status_code=413)(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()

            if message["type"] == "http.request":
                received += len(message.get("body", b""))

                if received > self.max_bytes:
                    # Raised inside the route's body read: FastAPI re-raises
                    # HTTPException and Starlette turns it into the 413.
                    raise HTTPException(413, _TOO_LARGE)

            return message

        await self.app(scope, limited_receive, send)


class SecurityHeaders:
    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        docs = str(scope.get("path", "")).startswith(_DOCS_PATHS)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                _set_missing(headers, _SECURITY_HEADERS)

                if not docs:
                    _set_missing(headers, (("Content-Security-Policy", _API_CSP),))

                if self.hsts:
                    _set_missing(headers, (("Strict-Transport-Security", _HSTS),))

            await send(message)

        await self.app(scope, receive, send_with_headers)


def _set_missing(headers: MutableHeaders, values: Iterable[tuple[str, str]]) -> None:
    for name, value in values:
        if name not in headers:
            headers[name] = value


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                # Malformed: the server rejects it, or the count catches it.
                return None

    return None


__all__: Any = ["MAX_REQUEST_BODY_BYTES", "RequestBodyLimit", "SecurityHeaders"]
