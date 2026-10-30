"""Daily routes and public GeoJSON maps for an itinerary.

Each day is processed independently: a provider failure on one day, or on
one travel profile, never removes the routes of another.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

from app.modules.routing.geojson import (
    ProfileRoute,
    build_feature_collection,
    empty_feature_collection,
    route_summary,
)
from app.modules.routing.policies import (
    ROUTE_PROFILES,
    bind_route_endpoints,
    profiles_to_request,
    route_metrics,
)
from app.modules.routing.ports import (
    DirectionsPort,
    PlaceGeocoderPort,
    RouteProfile,
)
from app.modules.routing.waypoints import (
    Waypoint,
    airport_waypoint,
    arrival_airport_query,
    day_waypoints,
)
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS: Final = 8


@dataclass(frozen=True, slots=True)
class DailyMapsResult:
    """Updated itinerary days plus the public map for each day number."""

    itinerary: list[dict[str, Any]]
    daily_maps: dict[int, dict[str, Any]]


class DailyRouteService:
    """Itinerary days → ordered stops, multi-profile routes and GeoJSON."""

    def __init__(
        self,
        *,
        directions: DirectionsPort,
        geocoder: PlaceGeocoderPort,
        profiles: Sequence[RouteProfile] = ROUTE_PROFILES,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self._directions = directions
        self._geocoder = geocoder
        self._profiles = tuple(profiles)
        self._max_workers = max_workers

    def build_daily_maps(
        self,
        itinerary: Sequence[Mapping[str, Any]],
        *,
        days: Collection[int] | None = None,
    ) -> DailyMapsResult:
        """Route every day in parallel and reassemble in original order.

        With ``days``, only those day numbers are routed: the other days are
        returned unchanged and have no entry in ``daily_maps``.

        The input is never mutated; each day is deep-copied first.
        """

        if not itinerary:
            return DailyMapsResult(itinerary=[], daily_maps={})

        selected = [
            index
            for index, day_plan in enumerate(itinerary)
            if days is None or day_plan.get("day", index + 1) in days
        ]

        results: dict[int, tuple[int, dict[str, Any], dict[str, Any]]] = {}

        if selected:
            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(selected)),
                thread_name_prefix="routing",
            ) as executor:
                futures = {
                    index: executor.submit(self._safe_build_day, index, itinerary[index])
                    for index in selected
                }
                results = {index: future.result() for index, future in futures.items()}

        ordered: list[dict[str, Any]] = []
        maps: dict[int, dict[str, Any]] = {}

        for index, day_plan in enumerate(itinerary):
            if index not in results:
                ordered.append(copy.deepcopy(dict(day_plan)))
                continue

            day_number, routed_plan, feature_collection = results[index]
            ordered.append(routed_plan)
            maps[day_number] = feature_collection

        return DailyMapsResult(itinerary=ordered, daily_maps=maps)

    def build_day(
        self,
        day_index: int,
        day_plan: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Return (day number, updated day copy, GeoJSON) for one day."""

        plan = copy.deepcopy(dict(day_plan))
        day_number = plan.get("day", day_index + 1)

        airport = (
            self._arrival_airport(plan)
            if day_number == 1
            else None
        )

        waypoints = day_waypoints(plan, airport=airport)
        routes = self._fetch_routes(waypoints)

        plan["route"] = route_summary(waypoints, routes)

        return day_number, plan, build_feature_collection(waypoints, routes)

    def route_between(
        self,
        coordinates: Sequence[tuple[float, float]],
    ) -> dict[str, dict[str, Any]]:
        """Per-profile metrics over ordered (lng, lat) points; no map.

        Used by features that need a route estimate without a day plan
        (for example, hotel → nearest pharmacy).
        """

        waypoints = [
            Waypoint(name="", coordinates=point, type="", order=None)
            for point in coordinates
        ]

        return {
            profile: {
                "distance_km": route.distance_km,
                "duration_mins": route.duration_mins,
            }
            for profile, route in self._fetch_routes(waypoints).items()
        }

    # ── Internals ───────────────────────────────────────────────────

    def _safe_build_day(
        self,
        day_index: int,
        day_plan: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        try:
            return self.build_day(day_index, day_plan)

        except Exception:
            safe_log(
                logger,
                logging.WARNING,
                "routing.day_processing_failed",
            )

            fallback = copy.deepcopy(dict(day_plan))
            # An empty map must not sit next to a stale route summary.
            fallback["route"] = None

            return (
                fallback.get("day", day_index + 1),
                fallback,
                empty_feature_collection(),
            )

    def _arrival_airport(
        self,
        day_plan: Mapping[str, Any],
    ) -> Waypoint | None:
        """Best-effort day-one airport stop; failure is not fatal."""

        name = arrival_airport_query(day_plan)

        if name is None:
            return None

        try:
            place = self._geocoder.geocode(name)

        except Exception:
            safe_log(
                logger,
                logging.WARNING,
                "routing.airport_geocode_failed",
            )

            return None

        if place is None:
            return None

        return airport_waypoint(name, (place.longitude, place.latitude))

    def _fetch_routes(
        self,
        waypoints: Sequence[Waypoint],
    ) -> dict[RouteProfile, ProfileRoute]:
        """Fetch each worthwhile profile over the SAME ordered waypoints.

        Profiles are fetched sequentially; days already run in parallel.
        """

        coordinates = [waypoint.coordinates for waypoint in waypoints]
        routes: dict[RouteProfile, ProfileRoute] = {}

        requested = profiles_to_request(coordinates, self._profiles)

        if len(requested) < len(self._profiles) and len(coordinates) >= 2:
            safe_log(
                logger,
                logging.INFO,
                "routing.profile_distance_capped",
            )

        for profile in requested:
            try:
                route = self._directions.fetch_route(coordinates, profile)

            except Exception:
                safe_log(
                    logger,
                    logging.WARNING,
                    "routing.profile_failed",
                )

                continue

            if route is None:
                continue

            distance_km, duration_mins = route_metrics(route)

            routes[profile] = ProfileRoute(
                distance_km=distance_km,
                duration_mins=duration_mins,
                line=bind_route_endpoints(route.coordinates, coordinates),
            )

        return routes


__all__ = [
    "DailyMapsResult",
    "DailyRouteService",
]
