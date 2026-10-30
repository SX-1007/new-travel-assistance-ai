"""Contracts required by daily routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from app.shared.geo import GeocodedPlace


RouteProfile = Literal["driving", "walking", "cycling"]

# One map position as [longitude, latitude] (GeoJSON order).
LngLat = tuple[float, float]


@dataclass(frozen=True, slots=True)
class DirectionsRoute:
    """One provider route over an ordered list of waypoints."""

    distance_m: float
    duration_s: float
    coordinates: tuple[LngLat, ...]


class DirectionsPort(Protocol):
    """Ordered waypoints → one route for one travel profile."""

    def fetch_route(
        self,
        coordinates: Sequence[LngLat],
        profile: RouteProfile,
    ) -> DirectionsRoute | None:
        """Return the route, or None when no usable route is available."""
        ...


class PlaceGeocoderPort(Protocol):
    """Place name → coordinates (used for the day-one arrival airport)."""

    def geocode(
        self,
        query: str,
        *,
        proximity: tuple[float, float] | None = None,
    ) -> GeocodedPlace | None:
        ...


__all__ = [
    "DirectionsPort",
    "DirectionsRoute",
    "LngLat",
    "PlaceGeocoderPort",
    "RouteProfile",
]
