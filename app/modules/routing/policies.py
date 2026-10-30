"""Deterministic routing rules: profiles, distance caps and route metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from app.modules.routing.ports import DirectionsRoute, LngLat, RouteProfile
from app.shared.geo import great_circle_km


# Travel profiles surfaced to the user so they can toggle walking/driving/
# cycling views of the same day's route.
ROUTE_PROFILES: Final[tuple[RouteProfile, ...]] = (
    "driving",
    "walking",
    "cycling",
)

# Straight-line distance caps per profile. Requesting a walking route across
# ~500 km (e.g. an Osaka → Tokyo day transition) makes the Directions API run
# past its timeout every time, so such profiles are skipped up front.
# Driving is uncapped.
PROFILE_MAX_STRAIGHT_KM: Final[Mapping[str, float]] = MappingProxyType(
    {
        "walking": 30.0,
        "cycling": 80.0,
    }
)

def haversine_km(a: LngLat, b: LngLat) -> float:
    """Great-circle distance in km between two (lng, lat) points."""

    return great_circle_km(a[1], a[0], b[1], b[0])


def path_straight_km(coordinates: Sequence[LngLat]) -> float:
    """Sum of straight-line leg distances along the ordered waypoints."""

    return sum(
        haversine_km(coordinates[i], coordinates[i + 1])
        for i in range(len(coordinates) - 1)
    )


def profiles_to_request(
    coordinates: Sequence[LngLat],
    profiles: Sequence[RouteProfile] = ROUTE_PROFILES,
) -> tuple[RouteProfile, ...]:
    """Return the profiles worth asking the provider for."""

    if len(coordinates) < 2:
        return ()

    straight_km = path_straight_km(coordinates)

    return tuple(
        profile
        for profile in profiles
        if straight_km <= PROFILE_MAX_STRAIGHT_KM.get(profile, math.inf)
    )


def route_metrics(route: DirectionsRoute) -> tuple[float, float]:
    """Return public (distance_km, duration_mins), rounded to 2 places."""

    return (
        round(route.distance_m / 1000, 2),
        round(route.duration_s / 60, 2),
    )


def bind_route_endpoints(
    route_coordinates: Sequence[LngLat],
    waypoints: Sequence[LngLat],
) -> list[list[float]]:
    """Pin a provider line's ends to the exact first and last waypoints.

    Mapbox may snap endpoints to the nearest routable road. The public map
    keeps the provider path but starts and ends exactly at the server-owned
    waypoints, which the itinerary validator checks.
    """

    line = [[float(lng), float(lat)] for lng, lat in route_coordinates]

    if len(line) >= 2 and waypoints:
        line[0] = [float(waypoints[0][0]), float(waypoints[0][1])]
        line[-1] = [float(waypoints[-1][0]), float(waypoints[-1][1])]

    return line


__all__ = [
    "PROFILE_MAX_STRAIGHT_KM",
    "ROUTE_PROFILES",
    "bind_route_endpoints",
    "haversine_km",
    "path_straight_km",
    "profiles_to_request",
    "route_metrics",
]
