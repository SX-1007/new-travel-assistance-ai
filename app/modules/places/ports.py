"""Contracts required by place search."""

from __future__ import annotations

from typing import Protocol

from app.modules.places.models import PlaceCandidate
from app.shared.geo import GeocodedPlace


class SearchQuotaPort(Protocol):
    """A per-run cap on paid provider requests."""

    def try_acquire(self) -> bool:
        """Reserve one request; False when the budget is spent."""
        ...


class PlaceSearchPort(Protocol):
    """Free-text place search around an optional map position."""

    def search(
        self,
        query: str,
        *,
        near: tuple[float, float] | None = None,
        zoom: int = 11,
        quota: SearchQuotaPort | None = None,
    ) -> list[PlaceCandidate]:
        """Return normalised candidates in provider order.

        ``near`` is (latitude, longitude). Raises ``PlaceSearchUnavailable``
        when the provider cannot answer, and ``PlaceSearchQuotaExhausted``
        when ``quota`` refuses a provider call. Cached answers never
        consume quota.
        """
        ...


class CityGeocoderPort(Protocol):
    """Place name → coordinates (used to anchor a trip city)."""

    def geocode(
        self,
        query: str,
        *,
        proximity: tuple[float, float] | None = None,
    ) -> GeocodedPlace | None:
        ...


class PlaceLocatorPort(Protocol):
    """Coordinates → country and locality evidence."""

    def resolve_country_code(
        self,
        latitude: float,
        longitude: float,
    ) -> str | None:
        ...

    def resolve_locality_names(
        self,
        latitude: float,
        longitude: float,
        destination_country_code: str,
    ) -> tuple[str, ...] | None:
        ...


__all__ = [
    "CityGeocoderPort",
    "PlaceLocatorPort",
    "PlaceSearchPort",
    "SearchQuotaPort",
]
