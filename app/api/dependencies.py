"""FastAPI request-scoped dependencies.

This module contains HTTP-layer concerns only: authentication context and
request correlation IDs. The token check itself sits behind
``IdentityTokenVerifierPort``, supplied by ``providers.identity_verifier``.
Infrastructure clients and orchestration internals must not be exposed from
here.
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

from app.api.providers import identity_verifier
from app.modules.identity.exceptions import InvalidIdentityToken
from app.modules.identity.ports import IdentityTokenVerifierPort


# A Firebase ID token is about 1 KB; anything far larger is refused unread.
_TOKEN_MAX_LEN = 4096
_BEARER_PREFIX = "bearer "

_REQUEST_STATE_ATTR = "request_id"
_REQUEST_ID_MAX_LEN = 128


def _unauthorized(detail: str, *, invalid_token: bool) -> HTTPException:
    challenge = 'Bearer error="invalid_token"' if invalid_token else "Bearer"
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": challenge},
    )


def verify_user_token(
    verifier: Annotated[IdentityTokenVerifierPort, Depends(identity_verifier)],
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """The user id proven by the request's ``Authorization: Bearer`` token.

    Sync on purpose: FastAPI runs it in a worker thread, because the first
    check fetches Google's signing keys. A missing, malformed or rejected
    token is a 401; a check that couldn't run is a 503 (see ``errors.py``).
    """

    header = (authorization or "").strip()

    if not header:
        raise _unauthorized("Sign-in required.", invalid_token=False)

    if not header.lower().startswith(_BEARER_PREFIX):
        raise _unauthorized("Use an Authorization: Bearer token.", invalid_token=True)

    token = header[len(_BEARER_PREFIX):].strip()

    if not token or len(token) > _TOKEN_MAX_LEN:
        raise _unauthorized("The sign-in token is malformed.", invalid_token=True)

    try:
        return verifier.verify(token)
    except InvalidIdentityToken as exc:
        raise _unauthorized("Your sign-in has expired or is invalid. Please sign in again.", invalid_token=True) from exc


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

    # Several dependencies may ask within one request; all get the same id.
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