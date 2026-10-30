"""Public daily-map payloads built from waypoints and fetched routes.

The output shapes are the closed contracts in ``models.py`` and are checked
again by the itinerary validator, so they must not gain extra fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.modules.routing.ports import RouteProfile
from app.modules.routing.waypoints import Waypoint


@dataclass(frozen=True, slots=True)
class ProfileRoute:
    """One profile's public metrics and endpoint-bound line."""

    distance_km: float
    duration_mins: float
    line: list[list[float]]


def empty_feature_collection() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def build_feature_collection(
    waypoints: Sequence[Waypoint],
    routes: Mapping[RouteProfile, ProfileRoute] | None = None,
) -> dict[str, Any]:
    """Ordered point pins plus one route LineString per travel profile.

    Each line is tagged with ``profile`` so the frontend can toggle
    walking/driving/cycling.
    """

    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    waypoint.coordinates[0],
                    waypoint.coordinates[1],
                ],
            },
            "properties": {
                "name": waypoint.name,
                "type": waypoint.type,
                "order": waypoint.order,
            },
        }
        for waypoint in waypoints
    ]

    for profile, route in (routes or {}).items():
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(point) for point in route.line],
                },
                "properties": {
                    "type": "route",
                    "profile": profile,
                    "distance_km": route.distance_km,
                    "duration_mins": route.duration_mins,
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def route_summary(
    waypoints: Sequence[Waypoint],
    routes: Mapping[RouteProfile, ProfileRoute],
) -> dict[str, Any] | None:
    """The itinerary day's ``route`` field, or None when nothing was routed."""

    if not routes:
        return None

    return {
        "ordered_stops": [waypoint.name for waypoint in waypoints],
        "profiles": {
            profile: {
                "distance_km": route.distance_km,
                "duration_mins": route.duration_mins,
            }
            for profile, route in routes.items()
        },
    }


__all__ = [
    "ProfileRoute",
    "build_feature_collection",
    "empty_feature_collection",
    "route_summary",
]
