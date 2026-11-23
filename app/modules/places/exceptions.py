"""Place-search failure contracts."""

from __future__ import annotations

from app.shared.exceptions import ProviderQuotaExhausted


class PlaceSearchUnavailable(RuntimeError):
    """The place provider could not answer (transport, HTTP or payload)."""


class PlaceSearchQuotaExhausted(PlaceSearchUnavailable):
    """The caller's request budget is spent; no provider call was made."""


class PlaceProviderQuotaExhausted(PlaceSearchUnavailable, ProviderQuotaExhausted):
    """The provider's own searches are used up for now (not the caller's
    budget). Callers that only handle ``PlaceSearchUnavailable`` treat it
    as a failed search; planning stops on it (``ProviderQuotaExhausted``)."""


__all__ = [
    "PlaceProviderQuotaExhausted",
    "PlaceSearchQuotaExhausted",
    "PlaceSearchUnavailable",
]
