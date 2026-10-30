"""Place-search value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PlaceCandidate:
    """One normalised, NOT yet verified provider result."""

    title: str
    latitude: float
    longitude: float
    address: str = ""
    category: str = ""
    rating: float | None = None
    reviews: int | None = None
    thumbnail: str = ""
    phone: str | None = None


@dataclass(frozen=True, slots=True)
class SearchArea:
    """Where a place must be: one trip city inside one country."""

    country: str
    country_code: str
    requested_city: str
    # (latitude, longitude) of the city, or None when it could not be
    # geocoded. Without an anchor only exact-locality matches are accepted.
    anchor: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class VerifiedPlace:
    """A candidate proven to lie inside its search area."""

    candidate: PlaceCandidate
    country_code: str
    requested_city: str
    verified_locality: str
    # Straight-line distance to the city anchor, when one exists. Required
    # evidence whenever ``verified_locality`` differs from the city.
    distance_from_requested_city_km: float | None

    @property
    def name(self) -> str:
        return self.candidate.title

    def location_payload(self) -> dict[str, Any]:
        """The public ``ActivityLocation`` shape for this place."""

        payload: dict[str, Any] = {
            "place_name": self.candidate.title,
            "latitude": self.candidate.latitude,
            "longitude": self.candidate.longitude,
            "country_code": self.country_code,
            "requested_city": self.requested_city,
            "verified_locality": self.verified_locality,
        }

        if self.distance_from_requested_city_km is not None:
            payload["distance_from_requested_city_km"] = (
                self.distance_from_requested_city_km
            )

        return payload


__all__ = [
    "PlaceCandidate",
    "SearchArea",
    "VerifiedPlace",
]
