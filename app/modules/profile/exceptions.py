"""Profile failure contracts."""

from __future__ import annotations


class ProfileUnavailable(RuntimeError):
    """The profile store could not be read or written."""


__all__ = [
    "ProfileUnavailable",
]
