"""Build one day's ordered map stops from an itinerary day payload.

Waypoint order is the logical daily route, NOT a shortest path:

  day 1 : [arrival airport?] → hotel → activities (in their given order)
  day N : hotel → activities (in their given order)

Hotel and activity coordinates come straight from their provider payloads,
so only the day-one airport needs geocoding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.modules.routing.ports import LngLat


@dataclass(frozen=True, slots=True)
class Waypoint:
    """One ordered map stop."""

    name: str
    coordinates: LngLat
    type: str
    order: Any


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)

    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def location_coordinates(location: Any) -> LngLat | None:
    """Return (lng, lat) from a location mapping, or None.

    Accepts both key styles in use:
      • hotels → {"lat": .., "lng": ..}
      • places → {"latitude": .., "longitude": ..}
    """

    if not isinstance(location, Mapping):
        return None

    lat = location.get("latitude")
    lng = location.get("longitude")

    if lat is None:
        lat = location.get("lat")

    if lng is None:
        lng = location.get("lng")

    lat_number = _number(lat)
    lng_number = _number(lng)

    if lat_number is None or lng_number is None:
        return None

    if not (-90.0 <= lat_number <= 90.0 and -180.0 <= lng_number <= 180.0):
        return None

    return lng_number, lat_number


def arrival_airport_query(day_plan: Mapping[str, Any]) -> str | None:
    """Return the day's arrival-airport name to geocode, if any.

    Must stay in step with the itinerary validator's expected airport name:
    the airport's name, else "<IATA> airport".
    """

    flights = day_plan.get("flight")
    flight: Any = None

    if isinstance(flights, list) and flights:
        flight = flights[0]

    elif isinstance(flights, Mapping):
        flight = flights

    if not isinstance(flight, Mapping):
        return None

    arrival = flight.get("arrival_airport")

    if not isinstance(arrival, Mapping):
        return None

    name = _text(arrival.get("name"))

    if name:
        return name

    airport_id = _text(arrival.get("id"))

    return f"{airport_id} airport" if airport_id else None


def airport_waypoint(name: str, coordinates: LngLat) -> Waypoint:
    return Waypoint(
        name=name,
        coordinates=coordinates,
        type="airport",
        order=0,
    )


def day_waypoints(
    day_plan: Mapping[str, Any],
    *,
    airport: Waypoint | None = None,
) -> list[Waypoint]:
    """Return the day's ordered stops: [airport] → hotel → activities."""

    waypoints: list[Waypoint] = []

    if airport is not None:
        waypoints.append(airport)

    hotel = day_plan.get("hotel")

    if isinstance(hotel, Mapping):
        coordinates = location_coordinates(hotel.get("location"))

        if coordinates is not None:
            waypoints.append(
                Waypoint(
                    name=_text(hotel.get("hotel_name")) or "Hotel",
                    coordinates=coordinates,
                    type="hotel",
                    order=0,
                )
            )

    for activity in day_plan.get("activities") or []:
        if not isinstance(activity, Mapping):
            continue

        coordinates = location_coordinates(activity.get("location"))

        if coordinates is None:
            continue

        waypoints.append(
            Waypoint(
                name=_text(activity.get("name")) or "Place",
                coordinates=coordinates,
                type=_text(activity.get("type")) or "activity",
                order=activity.get("order"),
            )
        )

    return waypoints


__all__ = [
    "Waypoint",
    "airport_waypoint",
    "arrival_airport_query",
    "day_waypoints",
    "location_coordinates",
]
