"""Destination-resolution failure contracts."""

from __future__ import annotations


class DestinationResolutionInvalid(ValueError):
    """A user-provided country/city combination is invalid."""


class DestinationResolutionUnavailable(RuntimeError):
    """A safe planning city could not be resolved or verified."""


__all__ = [
    "DestinationResolutionInvalid",
    "DestinationResolutionUnavailable",
]
