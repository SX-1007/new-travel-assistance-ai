"""Identity failure contracts."""

from __future__ import annotations


class InvalidIdentityToken(ValueError):
    """The sign-in token does not prove a user (the caller must sign in again)."""


class IdentityVerificationUnavailable(RuntimeError):
    """The token could not be checked, for example the signing keys were unreachable."""


__all__ = [
    "IdentityVerificationUnavailable",
    "InvalidIdentityToken",
]
