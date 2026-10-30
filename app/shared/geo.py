"""Provider-neutral geographic value objects and math.

Shared by every feature that consumes geocoding or distances (destination
resolution, routing, place search, nearby amenities) so none of them has to
import another feature's models or a concrete map provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final


_EARTH_RADIUS_KM: Final = 6371.0


@dataclass(frozen=True, slots=True)
class GeocodedPlace:
    """One forward-geocoding match."""

    latitude: float
    longitude: float
    place_name: str


def great_circle_km(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float,
) -> float:
    """Haversine distance in km between two (latitude, longitude) points."""

    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlng = math.radians(lng2 - lng1)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
    )
    return _EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(min(1.0, h)))


def valid_lat_lng(latitude: Any, longitude: Any) -> tuple[float, float] | None:
    """Return finite, in-range (latitude, longitude) or None.

    Rejects booleans, non-numbers, NaN, infinity and impossible coordinates.
    """

    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return None

    try:
        lat = float(latitude)
        lng = float(longitude)

    except (TypeError, ValueError, OverflowError):
        return None

    if not (math.isfinite(lat) and math.isfinite(lng)):
        return None

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None

    return lat, lng


__all__ = [
    "GeocodedPlace",
    "great_circle_km",
    "valid_lat_lng",
]
