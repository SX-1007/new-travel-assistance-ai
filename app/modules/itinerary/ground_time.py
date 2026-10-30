"""How many hours of each trip day the traveller is actually at the destination.

Flights come from real schedules: an overnight outbound flight can land
after midnight on day 2, and a cheap return flight can leave at 02:15 on the
last day. The old planner put a full day of places on both boundary days
regardless (found by the live smoke test: Senso-ji on a day the traveller
was still in Kuala Lumpur, four Tokyo places on a day whose flight left at
02:15).

A day's window is 08:00-22:00 local time, starting no earlier than an hour
and a half after landing (immigration, transfer) and ending three hours
before the return flight (an international check-in). Flight times are
the airports' local times, as the provider reports them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from typing import Any, Final


DAY_START: Final = time(8, 0)
DAY_END: Final = time(22, 0)
ARRIVAL_SETTLE: Final = timedelta(hours=1, minutes=30)
DEPARTURE_BUFFER: Final = timedelta(hours=3)
# One place (attraction or meal) per this many hours; a full day → 8.
HOURS_PER_PLACE: Final = 1.75


def hours_on_ground(itinerary: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Day number → usable hours at the destination.

    Days without a valid number or date are left out (callers treat them as
    unconstrained); unparseable flight times impose no limit.
    """

    days = [day for day in itinerary if isinstance(day, Mapping)]

    if not days:
        return {}

    # The skeleton puts the outbound journey first on day 1 and the return
    # journey last on the final day (a same-day trip carries both on day 1).
    first_legs, last_legs = _flights(days[0]), _flights(days[-1])
    outbound = first_legs[0] if first_legs else None
    returning = last_legs[-1] if last_legs and last_legs[-1] is not outbound else None
    arrival = _flight_time(outbound, "arrival") if outbound is not None else None
    departure = _flight_time(returning, "departure") if returning is not None else None
    hours: dict[int, float] = {}

    for day in days:
        number = day.get("day")
        day_date = _date(day.get("date"))

        if isinstance(number, bool) or not isinstance(number, int) or day_date is None:
            continue

        start = datetime.combine(day_date, DAY_START)
        end = datetime.combine(day_date, DAY_END)

        if arrival is not None:
            start = max(start, arrival + ARRIVAL_SETTLE)

        if departure is not None:
            end = min(end, departure - DEPARTURE_BUFFER)

        hours[number] = round(max(0.0, (end - start).total_seconds() / 3600), 2)

    return hours


def places_capacity(hours: float) -> int:
    """How many places fit into ``hours`` (0: a travel day)."""

    return int(max(0.0, hours) // HOURS_PER_PLACE)


def day_place_limits(itinerary: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Day number → most places that fit (0: no time at the destination)."""

    return {day: places_capacity(hours) for day, hours in hours_on_ground(itinerary).items()}


def travel_days(itinerary: Sequence[Mapping[str, Any]]) -> frozenset[int]:
    """Days with no usable time at the destination."""

    return frozenset(day for day, limit in day_place_limits(itinerary).items() if limit == 0)


# ── Internals ───────────────────────────────────────────────────────


def _flights(day: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    flights = day.get("flight")

    if isinstance(flights, Mapping):
        return [flights]

    if isinstance(flights, list):
        return [flight for flight in flights if isinstance(flight, Mapping)]

    return []


def _flight_time(flight: Mapping[str, Any], side: str) -> datetime | None:
    airport = flight.get(f"{side}_airport")
    raw = flight.get(f"{side}_time")

    if not raw and isinstance(airport, Mapping):
        raw = airport.get("time")

    if not isinstance(raw, str):
        return None

    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None

    # Local airport time: compare as naive wall-clock time.
    return parsed.replace(tzinfo=None)


def _date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


__all__ = [
    "ARRIVAL_SETTLE",
    "DAY_END",
    "DAY_START",
    "DEPARTURE_BUFFER",
    "HOURS_PER_PLACE",
    "day_place_limits",
    "hours_on_ground",
    "places_capacity",
    "travel_days",
]
