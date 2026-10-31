"""Mapbox Directions: routes only.

It has its own HTTP session, separate from the forward and reverse
geocoders, so a slow or failing Directions request never ties up geocoding
connections.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Final

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.modules.routing.ports import DirectionsRoute, LngLat, RouteProfile
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


_REQUEST_TIMEOUT_SECONDS: Final = 10.0

_DIRECTIONS_URL: Final = "https://api.mapbox.com/directions/v5/mapbox"

_SUPPORTED_PROFILES: Final = frozenset({"driving", "walking", "cycling"})

# Mapbox Directions accepts at most 25 coordinates per request.
_MAX_WAYPOINTS: Final = 25


# Read timeouts are retried only once: long or unroutable requests (for
# example a 500 km walking leg) time out every time, and retrying them
# repeatedly stalls the whole planning request.
_RETRY_STRATEGY = Retry(
    total=3,
    connect=3,
    read=1,
    status=3,
    backoff_factor=0.5,
    status_forcelist=(
        429,
        500,
        502,
        503,
        504,
    ),
    allowed_methods=frozenset(
        {"GET"}
    ),
    respect_retry_after_header=True,
    raise_on_status=False,
)


def _build_http_session(
) -> requests.Session:
    session = requests.Session()

    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=_RETRY_STRATEGY,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def _valid_point(value: Any) -> LngLat | None:
    """Return a finite, in-range (longitude, latitude) pair or None."""

    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
    ):
        return None

    try:
        lng = float(value[0])
        lat = float(value[1])

    except (TypeError, ValueError, OverflowError):
        return None

    if not (math.isfinite(lng) and math.isfinite(lat)):
        return None

    if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
        return None

    return lng, lat


def _non_negative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        number = float(value)

    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) and number >= 0 else None


def _parse_route(payload: dict[str, Any]) -> DirectionsRoute | None:
    """The first route, or None unless it has a distance, a duration and a
    LineString of at least two valid points.
    """
    routes = payload.get("routes") or []

    if not routes or not isinstance(routes[0], dict):
        return None

    route = routes[0]
    distance = _non_negative(route.get("distance"))
    duration = _non_negative(route.get("duration"))

    if distance is None or duration is None:
        return None

    geometry = route.get("geometry")

    if (
        not isinstance(geometry, dict)
        or geometry.get("type") != "LineString"
    ):
        return None

    raw_coordinates = geometry.get("coordinates")

    if not isinstance(raw_coordinates, list):
        return None

    coordinates: list[LngLat] = []

    for raw_point in raw_coordinates:
        point = _valid_point(raw_point)

        if point is None:
            return None

        coordinates.append(point)

    if len(coordinates) < 2:
        return None

    return DirectionsRoute(
        distance_m=distance,
        duration_s=duration,
        coordinates=tuple(coordinates),
    )


class MapboxDirectionsClient:
    """Thread-safe Mapbox Directions client."""

    def __init__(
        self,
        *,
        access_token: str,
        session: (
            requests.Session | None
        ) = None,
    ) -> None:
        token = access_token.strip()

        if not token:
            raise ValueError(
                "access_token must not be empty"
            )

        self._access_token = token

        self._session = (
            session
            or _build_http_session()
        )

        self._owns_session = (
            session is None
        )

    def close(self) -> None:
        """Close the internally-owned HTTP session."""

        if self._owns_session:
            self._session.close()

    def fetch_route(
        self,
        coordinates: Sequence[LngLat],
        profile: RouteProfile,
    ) -> DirectionsRoute | None:
        """The first Mapbox route over the ordered coordinates.

        None means no usable route (bad input, provider failure, or no route).
        Nothing is cached: itinerary routes rarely repeat.
        """

        if profile not in _SUPPORTED_PROFILES:
            return None

        points: list[LngLat] = []

        for raw_point in coordinates:
            point = _valid_point(raw_point)

            if point is None:
                return None

            points.append(point)

        if not 2 <= len(points) <= _MAX_WAYPOINTS:
            return None

        path = ";".join(f"{lng},{lat}" for lng, lat in points)

        params = {
            "access_token": self._access_token,
            "geometries": "geojson",
            "overview": "full",
        }

        try:
            response = self._session.get(
                f"{_DIRECTIONS_URL}/{profile}/{path}",
                params=params,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )

        except requests.RequestException:
            safe_log(
                logger,
                logging.WARNING,
                "provider.mapbox.route_transport_failed",
            )

            return None

        if response.status_code != 200:
            safe_log(
                logger,
                logging.WARNING,
                "provider.mapbox.route_http_failed",
                status_code=response.status_code,
            )

            return None

        try:
            payload = response.json()

        except ValueError:
            safe_log(
                logger,
                logging.WARNING,
                "provider.mapbox.route_invalid_json",
            )

            return None

        if not isinstance(payload, dict):
            return None

        return _parse_route(payload)


__all__ = [
    "MapboxDirectionsClient",
]
