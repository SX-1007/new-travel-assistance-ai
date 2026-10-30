"""Contracts required by nearby-amenity search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class RouteEstimatorPort(Protocol):
    """Ordered (lng, lat) waypoints → per-profile distance and duration."""

    def route_between(
        self,
        coordinates: Sequence[tuple[float, float]],
    ) -> dict[str, dict[str, Any]]:
        """Return ``{profile: {"distance_km", "duration_mins"}}``; may be empty."""
        ...


__all__ = [
    "RouteEstimatorPort",
]
