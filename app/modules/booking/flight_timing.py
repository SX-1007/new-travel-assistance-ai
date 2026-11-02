"""How much time a trip's flights leave at the destination. Pure rules.

A trip day's window is 08:00-22:00 local time, starting no earlier than an
hour and a half after the outbound flight lands (immigration, transfer) and
ending three hours before the return flight leaves (an international
check-in). Flight times are the airports' local times, as the provider
reports them.

This is the one copy of that rule in the backend: flight choice
(``selection.pick_flight_pair``), the hotel check-in night (``hotels``) and
the itinerary's per-day limits (``itinerary.ground_time``) all use it. The
frontend mirrors it in ``frontend/src/lib/groundTime.ts``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from typing import Any, Final, Literal


DAY_START: Final = time(8, 0)
DAY_END: Final = time(22, 0)
ARRIVAL_SETTLE: Final = timedelta(hours=1, minutes=30)
DEPARTURE_BUFFER: Final = timedelta(hours=3)

FlightSide = Literal["arrival", "departure"]


def flight_time(flight: Any, side: FlightSide) -> datetime | None:
    """A flight's local arrival or departure time, or None when unreadable.

    Accepts a ``FlightResult`` or its JSON shape: ``<side>_time`` first, then
    the airport's ``time``.
    """

    if not isinstance(flight, Mapping) and callable(getattr(flight, "model_dump", None)):
        flight = flight.model_dump(mode="json")

    if not isinstance(flight, Mapping):
        return None

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


def free_window(
    day: date,
    *,
    arrival: datetime | None,
    departure: datetime | None,
) -> tuple[datetime, datetime]:
    """The day's usable (start, end); end <= start means no time at all.

    ``arrival`` / ``departure`` None impose no limit.
    """

    start = datetime.combine(day, DAY_START)
    end = datetime.combine(day, DAY_END)

    if arrival is not None:
        start = max(start, arrival + ARRIVAL_SETTLE)

    if departure is not None:
        end = min(end, departure - DEPARTURE_BUFFER)

    return start, end


def free_hours(
    day: date,
    *,
    arrival: datetime | None,
    departure: datetime | None,
) -> float:
    """Usable hours on ``day``, rounded to 0.01 h (never negative)."""

    start, end = free_window(day, arrival=arrival, departure=departure)
    return round(max(0.0, (end - start).total_seconds() / 3600), 2)


def trip_free_hours(
    start_date: str,
    end_date: str,
    *,
    arrival: datetime | None,
    departure: datetime | None,
) -> float | None:
    """Total usable hours over the trip's days (``start_date`` to
    ``end_date``, both included), or None when the dates are invalid."""

    try:
        first = date.fromisoformat(start_date)
        last = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return None

    return round(
        sum(
            free_hours(first + timedelta(days=offset), arrival=arrival, departure=departure)
            for offset in range((last - first).days + 1)
        ),
        2,
    )


__all__ = [
    "ARRIVAL_SETTLE",
    "DAY_END",
    "DAY_START",
    "DEPARTURE_BUFFER",
    "FlightSide",
    "flight_time",
    "free_hours",
    "free_window",
    "trip_free_hours",
]
