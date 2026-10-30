"""Find the nearest amenity (police, hospital, pharmacy, ...) to a location.

The search origin must be trustworthy: an itinerary hotel's stored
coordinates, or a place verified inside the trip city. Amenities themselves
are ranked by real straight-line distance from that origin, and the nearest
one gets route estimates.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.modules.amenities.policies import (
    MAX_RESULT_DISTANCE_KM,
    MAX_RESULTS,
    SEARCH_ZOOMS,
    amenity_queries,
)
from app.modules.amenities.ports import RouteEstimatorPort
from app.modules.places.exceptions import PlaceSearchUnavailable
from app.modules.places.models import PlaceCandidate
from app.modules.places.policies import (
    canonical_city,
    clean_text,
    join_query,
    text_key,
)
from app.modules.places.ports import PlaceSearchPort
from app.modules.places.service import VerifiedPlaceSearch
from app.shared.countries import country_alpha2
from app.shared.geo import great_circle_km, valid_lat_lng
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


def _hotel_origin(
    itinerary: Sequence[Mapping[str, Any]],
    location_name: str,
) -> dict[str, Any] | None:
    """An itinerary hotel with this exact name, using its stored coordinates.

    Hotel coordinates are already provider-verified by hotel search, so
    re-geocoding the name would be slower and less reliable.
    """

    target = text_key(location_name)

    for day in itinerary:
        hotel = day.get("hotel") if isinstance(day, Mapping) else None

        if not isinstance(hotel, Mapping) or text_key(hotel.get("hotel_name")) != target:
            continue

        location = hotel.get("location")

        if not isinstance(location, Mapping):
            continue

        coordinates = valid_lat_lng(
            location.get("latitude", location.get("lat")),
            location.get("longitude", location.get("lng")),
        )

        if coordinates is not None:
            return {
                "name": clean_text(hotel.get("hotel_name")),
                "latitude": coordinates[0],
                "longitude": coordinates[1],
            }

    return None


class NearbyAmenityService:
    """Nearest amenities around a trusted origin, with routes to the nearest."""

    def __init__(
        self,
        *,
        search: PlaceSearchPort,
        places: VerifiedPlaceSearch,
        routes: RouteEstimatorPort,
    ) -> None:
        self._search = search
        self._places = places
        self._routes = routes

    def find_nearest(
        self,
        *,
        category: str,
        location_name: str | None,
        itinerary: Sequence[Mapping[str, Any]],
        country: str,
        cities: Sequence[str],
        city_hint: str = "",
    ) -> dict[str, Any]:
        """Return the tool payload: origin, ranked results and nearest route."""

        if not clean_text(location_name):
            return {
                "error": (
                    "No specific_location provided. Ask the user for a "
                    "specific location to search near."
                )
            }

        origin = _hotel_origin(itinerary, location_name or "") or self._verified_origin(
            location_name or "",
            country=country,
            cities=cities,
            city_hint=city_hint,
        )

        if origin is None:
            return {
                "error": (
                    "I could not resolve the requested search origin to a "
                    "verified location inside the trip area."
                )
            }

        queries = amenity_queries(category, country_alpha2(country))

        if not queries:
            return {
                "error": "No amenity category was provided.",
                "search_category": category,
            }

        results = self._ranked_results(queries, origin)

        if not results:
            return {
                "search_category": category,
                "origin": origin,
                "results": [],
                "error": f"No {clean_text(category)} was found near {origin['name']}.",
            }

        nearest = dict(results[0])
        nearest["route_from_origin"] = {
            "origin": origin["name"],
            "destination": nearest["name"],
            "profiles": self._safe_routes(origin, nearest),
        }

        return {
            "search_category": category,
            "origin": origin,
            "nearest": nearest,
            "results": results,
        }

    # ── Internals ───────────────────────────────────────────────────

    def _verified_origin(
        self,
        location_name: str,
        *,
        country: str,
        cities: Sequence[str],
        city_hint: str,
    ) -> dict[str, Any] | None:
        configured = [city for city in map(clean_text, cities) if city]

        if not configured:
            return None

        city = canonical_city(city_hint, configured) or configured[0]
        area = self._places.area(country=country, city=city)

        if area is None:
            return None

        try:
            places = self._places.search(
                join_query(location_name, city, area.country),
                area=area,
                limit=1,
            )

        except PlaceSearchUnavailable:
            return None

        if not places:
            return None

        place = places[0]

        return {
            "name": place.name,
            "latitude": place.candidate.latitude,
            "longitude": place.candidate.longitude,
        }

    def _ranked_results(
        self,
        queries: Sequence[str],
        origin: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        origin_lat = float(origin["latitude"])
        origin_lng = float(origin["longitude"])
        found: dict[tuple[str, float, float], dict[str, Any]] = {}

        for zoom in SEARCH_ZOOMS:
            for query in queries:
                try:
                    candidates = self._search.search(
                        query,
                        near=(origin_lat, origin_lng),
                        zoom=zoom,
                    )

                except PlaceSearchUnavailable:
                    safe_log(logger, logging.WARNING, "amenities.search_failed")
                    continue

                for candidate in candidates:
                    result = _result(candidate, origin_lat, origin_lng)

                    if result is None:
                        continue

                    key = (
                        text_key(candidate.title),
                        round(candidate.latitude, 5),
                        round(candidate.longitude, 5),
                    )
                    found.setdefault(key, result)

            # Something close exists; no reason to widen the search.
            if found:
                break

        return sorted(found.values(), key=lambda item: item["distance_meters"])[
            :MAX_RESULTS
        ]

    def _safe_routes(
        self,
        origin: Mapping[str, Any],
        nearest: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        try:
            return self._routes.route_between(
                [
                    (float(origin["longitude"]), float(origin["latitude"])),
                    (float(nearest["longitude"]), float(nearest["latitude"])),
                ]
            )

        except Exception:
            safe_log(logger, logging.WARNING, "amenities.route_failed")
            return {}


def _result(
    candidate: PlaceCandidate,
    origin_lat: float,
    origin_lng: float,
) -> dict[str, Any] | None:
    distance_km = great_circle_km(
        origin_lat,
        origin_lng,
        candidate.latitude,
        candidate.longitude,
    )

    if distance_km > MAX_RESULT_DISTANCE_KM:
        return None

    return {
        "name": candidate.title,
        "address": candidate.address or None,
        "phone": candidate.phone,
        "category": candidate.category,
        "distance_meters": round(distance_km * 1000, 1),
        "latitude": candidate.latitude,
        "longitude": candidate.longitude,
    }


__all__ = [
    "NearbyAmenityService",
]
