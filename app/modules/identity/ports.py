"""Contracts for proving who is calling."""

from __future__ import annotations

from typing import Protocol


class IdentityTokenVerifierPort(Protocol):
    """Check a sign-in token and return the user id it proves.

    Raises ``InvalidIdentityToken`` for a token that is malformed, expired,
    revoked or not issued for this app, and ``IdentityVerificationUnavailable``
    when the check itself could not run.
    """

    def verify(self, token: str) -> str:
        ...


__all__ = [
    "IdentityTokenVerifierPort",
]
