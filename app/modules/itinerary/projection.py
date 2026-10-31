"""Compact, model-facing projection of an itinerary.

Reviewers and answer writers need the facts a reply can mention (days,
dates, bookings, places, costs, travel times), not the stored payload.
Coordinates, GeoJSON maps, thumbnails, provider links and descriptions are
left out: they would make every review call several times larger without
supporting any claim a reply makes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final


_FLIGHT_FIELDS = (
    "airline",
    "flight_number",
    "travel_class",
    "departure_time",
    "arrival_time",
    "duration",
    "stops",
    "price",
    "over_budget",
)
_HOTEL_FIELDS = (
    "hotel_name",
    "hotel_class",
    "overall_rating",
    "price_per_night",
    "check_in_time",
    "check_out_time",
    "over_budget",
)
_ACTIVITY_FIELDS = (
    "name",
    "type",
    "order",
    "estimated_cost",
    "is_estimated",
    "suggested_time",
    "category",
    "rating",
)
_MAX_TEXT = 160


def compact_itinerary(itinerary: Any) -> list[dict[str, Any]]:
    """Every day reduced to the facts a reply may state."""

    if not isinstance(itinerary, Sequence) or isinstance(itinerary, (str, bytes)):
        return []

    return [_compact_day(day) for day in itinerary if isinstance(day, Mapping)]


def _compact_day(day: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "day": day.get("day"),
        "date": day.get("date"),
        "day_total_cost": day.get("day_total_cost"),
    }

    # "flights" accepts already-compact input, so compacting twice is safe.
    flights = day.get("flight", day.get("flights"))

    if isinstance(flights, Mapping):
        flights = [flights]

    if isinstance(flights, list) and flights:
        compact["flights"] = [_compact_flight(f) for f in flights if isinstance(f, Mapping)]

    hotel = day.get("hotel")

    if isinstance(hotel, Mapping):
        compact["hotel"] = _pick(hotel, _HOTEL_FIELDS)

    activities = day.get("activities")

    if isinstance(activities, list):
        compact["activities"] = [
            _pick(activity, _ACTIVITY_FIELDS)
            for activity in activities
            if isinstance(activity, Mapping)
        ]

    route = day.get("route")

    if isinstance(route, Mapping):
        compact["route"] = {
            key: route[key]
            for key in ("ordered_stops", "profiles")
            if key in route
        }

    return compact


def _compact_flight(flight: Mapping[str, Any]) -> dict[str, Any]:
    compact = _pick(flight, _FLIGHT_FIELDS)

    for side in ("departure_airport", "arrival_airport"):
        airport = flight.get(side)

        if isinstance(airport, Mapping):
            compact[side] = _pick(airport, ("id", "name", "time"))

    return compact


def _pick(source: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    picked: dict[str, Any] = {}

    for field in fields:
        value = source.get(field)

        if value is None:
            continue

        picked[field] = value[:_MAX_TEXT] if isinstance(value, str) else value

    return picked


NO_ITINERARY_TEXT: Final = "No itinerary drafted yet. You are in the initial planning phase."
NO_HISTORY_TEXT: Final = "No previous accepted itinerary revisions are available."


def itinerary_text(itinerary: Any) -> str:
    """Readable day-by-day summary for the agent prompt.

    Activities carry the 1-based ``[n]`` labels that ``edit_itinerary``
    indexes use. Every flight leg is shown (a same-day trip has two).
    """

    lines: list[str] = []

    for day in compact_itinerary(itinerary):
        parts = [f"- Day {_value(day.get('day'))} ({_value(day.get('date'), 'TBD')})"]
        parts.append(f"Cost: {_amount(day.get('day_total_cost'))}")

        for flight in day.get("flights") or []:
            parts.append(_flight_text(flight))

        hotel = day.get("hotel")

        if hotel:
            stars = hotel.get("hotel_class") or hotel.get("overall_rating") or "?"
            over = " [OVER BUDGET]" if hotel.get("over_budget") else ""
            parts.append(
                f"Hotel: {_value(hotel.get('hotel_name'), 'Unknown')} "
                f"({stars}*, {_amount(hotel.get('price_per_night'))}/night){over}"
            )

        lines.append(" | ".join(parts))
        activities = day.get("activities") or []

        if activities:
            lines.append("  Activities:")

            for index, activity in enumerate(activities, start=1):
                time = f" — {activity['suggested_time']}" if activity.get("suggested_time") else ""
                lines.append(
                    f"    [{index}] {_value(activity.get('type'), 'activity')}: "
                    f"{_value(activity.get('name'), 'Place')}{time} "
                    f"~{_amount(activity.get('estimated_cost'))}"
                    f"{' (est.)' if activity.get('is_estimated') else ''}"
                )

        route = day.get("route") or {}
        stops = route.get("ordered_stops") or []

        if stops:
            metrics = [
                f"{profile} {metric.get('distance_km', '?')}km/{metric.get('duration_mins', '?')}min"
                for profile, metric in (route.get("profiles") or {}).items()
                if isinstance(metric, Mapping)
            ]
            route_line = "  Route: " + " → ".join(map(str, stops))

            if metrics:
                route_line += " | " + "; ".join(metrics)

            lines.append(route_line)

    return "\n".join(lines) if lines else NO_ITINERARY_TEXT


def history_text(entries: Sequence[Mapping[str, Any]]) -> str:
    """Earlier accepted revisions, each as a short day/item list."""

    blocks: list[str] = []

    for entry in entries:
        lines = [f"Revision {entry.get('revision')} (previous accepted itinerary):"]

        for day in compact_itinerary(entry.get("draft_itinerary")):
            lines.append(f"- Day {_value(day.get('day'))} ({_value(day.get('date'), 'TBD')})")

            for flight in day.get("flights") or []:
                number = f" {flight['flight_number']}" if flight.get("flight_number") else ""
                lines.append(f"    Flight: {_value(flight.get('airline'), 'Unknown airline')}{number}")

            hotel = day.get("hotel")

            if hotel and hotel.get("hotel_name"):
                lines.append(f"    Hotel: {hotel['hotel_name']}")

            for index, activity in enumerate(day.get("activities") or [], start=1):
                lines.append(
                    f"    [{index}] {_value(activity.get('type'), 'activity')}: "
                    f"{_value(activity.get('name'), 'Place')}"
                )

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else NO_HISTORY_TEXT


def _flight_text(flight: Mapping[str, Any]) -> str:
    departure = flight.get("departure_airport") or {}
    arrival = flight.get("arrival_airport") or {}
    route = f"{_value(departure.get('id'), '?')} → {_value(arrival.get('id'), '?')}"
    times = f"{_value(departure.get('time') or flight.get('departure_time'), '?')} → "
    times += _value(arrival.get("time") or flight.get("arrival_time"), "?")
    over = " [OVER BUDGET]" if flight.get("over_budget") else ""

    return (
        f"Flight: {_value(flight.get('airline'), 'Unknown')} "
        f"{_value(flight.get('flight_number'), 'N/A')} ({route}, {times}, "
        f"{_amount(flight.get('price'))}){over}"
    )


def _value(value: Any, default: str = "?") -> str:
    if value is None or value == "":
        return default

    return str(value)


def _amount(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "?"

    return f"{value:.2f}"


__all__ = [
    "NO_HISTORY_TEXT",
    "NO_ITINERARY_TEXT",
    "compact_itinerary",
    "history_text",
    "itinerary_text",
]
