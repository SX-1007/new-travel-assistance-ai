"""How many hours of each trip day the traveller is actually at the destination,
and how many places fit into them.

Flights come from real schedules: an overnight outbound flight can land
after midnight on day 2, and a cheap return flight can leave at 02:15 on the
last day. Planning a full day of places on such a day would send the
traveller to places while they are still in the air or at home.

The window itself (08:00-22:00, 1.5 h after landing, 3 h before the return
flight) is the booking rule in ``booking.flight_timing``; this module turns
it into places. A meal takes about an hour and a sight 1.75 h (with the
walk or ride to it), and a day has at most two meals (lunch and dinner).
Under one sight's worth of time makes a travel day with no places.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any, Final

from app.modules.booking.flight_timing import (
    ARRIVAL_SETTLE,
    DAY_END,
    DAY_START,
    DEPARTURE_BUFFER,
    flight_time,
    free_hours,
)


HOURS_PER_SIGHT: Final = 1.75
HOURS_PER_MEAL: Final = 1.0
MEALS_PER_DAY: Final = 2
# Less than this is a travel day. The frontend's travel-day rule
# (frontend/src/lib/groundTime.ts) uses the same number.
MIN_PLANNABLE_HOURS: Final = HOURS_PER_SIGHT


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
    arrival = flight_time(outbound, "arrival") if outbound is not None else None
    departure = flight_time(returning, "departure") if returning is not None else None
    hours: dict[int, float] = {}

    for day in days:
        number = day.get("day")
        day_date = _date(day.get("date"))

        if isinstance(number, bool) or not isinstance(number, int) or day_date is None:
            continue

        hours[number] = free_hours(day_date, arrival=arrival, departure=departure)

    return hours


def place_hours(place_type: Any) -> float:
    """Time one place takes: a meal is shorter than a sight."""

    return HOURS_PER_MEAL if place_type == "restaurant" else HOURS_PER_SIGHT


def places_capacity(hours: float) -> int:
    """Most places that fit into ``hours``: up to two meals, the rest sights
    (0: a travel day)."""

    if hours < MIN_PLANNABLE_HOURS:
        return 0

    meals = min(MEALS_PER_DAY, int(hours // HOURS_PER_MEAL))
    return meals + int((hours - meals * HOURS_PER_MEAL) // HOURS_PER_SIGHT)


def fits_in(place_types: Iterable[Any], hours: float) -> bool:
    """Whether these places, meals at an hour and sights at 1.75 h, fit."""

    types = list(place_types)
    return len(types) <= places_capacity(hours) and sum(map(place_hours, types)) <= hours


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
    "HOURS_PER_MEAL",
    "HOURS_PER_SIGHT",
    "MEALS_PER_DAY",
    "MIN_PLANNABLE_HOURS",
    "day_place_limits",
    "fits_in",
    "hours_on_ground",
    "place_hours",
    "places_capacity",
    "travel_days",
]
