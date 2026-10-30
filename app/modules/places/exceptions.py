"""Place-search failure contracts."""

from __future__ import annotations


class PlaceSearchUnavailable(RuntimeError):
    """The place provider could not answer (transport, HTTP or payload)."""


class PlaceSearchQuotaExhausted(PlaceSearchUnavailable):
    """The caller's request budget is spent; no provider call was made."""


__all__ = [
    "PlaceSearchQuotaExhausted",
    "PlaceSearchUnavailable",
]
