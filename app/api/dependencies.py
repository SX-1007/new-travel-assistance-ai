"""FastAPI request-scoped dependencies.

This module contains HTTP-layer concerns only: authentication context and
request correlation IDs. Infrastructure clients and orchestration internals
must not be exposed from here.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import (
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)


# Alternative trusted-header names retained for compatibility
# with the old API.
_USER_ID_FALLBACK_HEADERS: tuple[str, ...] = (
    "x-uid",
    "x-auth-user-id",
)

_USER_ID_MIN_LEN = 1
_USER_ID_MAX_LEN = 128

_REQUEST_STATE_ATTR = "request_id"
_REQUEST_ID_MAX_LEN = 128


async def verify_user_token(
    request: Request,
    x_user_id: Annotated[
        str | None,
        Header(alias="X-User-ID"),
    ] = None,
) -> str:
    """Return the validated caller user ID from trusted request headers.

    This preserves the authentication behaviour of the old application.
    Token/JWT verification can later replace the trust step without changing
    route signatures because callers depend only on ``CurrentUser``.
    """
    user_id = x_user_id

    if not user_id:
        for alias in _USER_ID_FALLBACK_HEADERS:
            candidate = request.headers.get(alias)

            if candidate:
                user_id = candidate
                break

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing X-User-ID header.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    user_id = user_id.strip()

    if not (
        _USER_ID_MIN_LEN
        <= len(user_id)
        <= _USER_ID_MAX_LEN
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid user identifier format.",
        )

    return user_id


CurrentUser = Annotated[
    str,
    Depends(verify_user_token),
]


class UserChatContext:
    """Request context shared by chat-style endpoints."""

    __slots__ = (
        "user_id",
    )

    def __init__(
        self,
        user_id: Annotated[
            str,
            Depends(verify_user_token),
        ],
    ) -> None:
        self.user_id = user_id

    def __repr__(self) -> str:
        return (
            f"UserChatContext("
            f"user_id={self.user_id!r}"
            f")"
        )


async def get_request_id(
    request: Request,
    x_request_id: Annotated[
        str | None,
        Header(alias="X-Request-ID"),
    ] = None,
) -> str:
    """Return one stable correlation ID for the current HTTP request."""

    if x_request_id:
        request_id = x_request_id.strip()

        if (
            request_id
            and len(request_id)
            <= _REQUEST_ID_MAX_LEN
        ):
            setattr(
                request.state,
                _REQUEST_STATE_ATTR,
                request_id,
            )
            return request_id

    cached = getattr(
        request.state,
        _REQUEST_STATE_ATTR,
        None,
    )

    if cached:
        return str(cached)

    request_id = str(
        uuid.uuid4()
    )

    setattr(
        request.state,
        _REQUEST_STATE_ATTR,
        request_id,
    )

    return request_id


RequestId = Annotated[
    str,
    Depends(get_request_id),
]


__all__ = [
    "verify_user_token",
    "CurrentUser",
    "UserChatContext",
    "get_request_id",
    "RequestId",
]