"""The limits every trip request must meet, wherever it comes from.

The trip form (``api/schemas/trip.py``) and the chat tool that changes a
planned trip (``orchestration/tools/trip.py``) both check these rules, so a
trip chat can't create is also one the form refuses, and the other way round.

Why each limit exists:

* travellers: Google Flights (through SerpAPI) searches at most 9 adults; a
  larger party made the flight search fail and the user saw "retry".
* trip length: every day is planned place by place, with paid searches.
* start and end dates: a past date makes the flight search fail, and
  Google Flights only sells about 11 months ahead. The end date is checked
  too, because the return flight is searched on it.
* cities and names: they go into prompts and search queries.

"Today" is UTC, from ``today()``, which tests pin with ``pinned_today``.
A traveller west of UTC can still be on the previous day, so a start date
one day before UTC's today is allowed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Final


MAX_TRAVELLERS: Final = 9
MAX_TRIP_DAYS: Final = 14
MAX_DAYS_AHEAD: Final = 330
START_DATE_SLACK_DAYS: Final = 1
MAX_CITIES: Final = 5
MAX_PLACE_NAME_LENGTH: Final = 100


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


_today_provider: Callable[[], date] = _utc_today


def today() -> date:
    """Today's date in UTC (replaceable in tests with ``pinned_today``)."""

    return _today_provider()


@contextmanager
def pinned_today(day: date) -> Iterator[None]:
    """Make ``today()`` return ``day`` inside the block (tests only)."""

    global _today_provider
    previous = _today_provider
    _today_provider = lambda: day  # noqa: E731

    try:
        yield
    finally:
        _today_provider = previous


def travellers_problem(num_people: int) -> str | None:
    if num_people < 1:
        return "There must be at least 1 traveller."

    if num_people > MAX_TRAVELLERS:
        return f"At most {MAX_TRAVELLERS} travellers can be planned at once (the flight search limit)."

    return None


def place_name_problem(value: str, label: str) -> str | None:
    if not value.strip():
        return f"{label} must not be empty."

    if len(value) > MAX_PLACE_NAME_LENGTH:
        return f"{label} must be at most {MAX_PLACE_NAME_LENGTH} characters."

    return None


def cities_problem(cities: Sequence[str]) -> str | None:
    if len(cities) > MAX_CITIES:
        return f"At most {MAX_CITIES} cities can be planned in one trip."

    for city in cities:
        if problem := place_name_problem(city, "Each city"):
            return problem

    return None


def trip_dates_problem(start: date, end: date) -> str | None:
    """Why these dates can't be planned, or None."""

    if end < start:
        return "The end date must be on or after the start date."

    days = (end - start).days + 1

    if days > MAX_TRIP_DAYS:
        return f"A trip can be at most {MAX_TRIP_DAYS} days long; this one is {days} days."

    current = today()
    earliest = current - timedelta(days=START_DATE_SLACK_DAYS)
    latest = current + timedelta(days=MAX_DAYS_AHEAD)

    if start < earliest:
        return f"The start date {start.isoformat()} has already passed; choose dates from today on."

    if end > latest:
        return (
            f"Flights can only be searched about {MAX_DAYS_AHEAD} days ahead: "
            f"the trip must end by {latest.isoformat()}."
        )

    return None


__all__ = [
    "MAX_CITIES",
    "MAX_DAYS_AHEAD",
    "MAX_PLACE_NAME_LENGTH",
    "MAX_TRAVELLERS",
    "MAX_TRIP_DAYS",
    "START_DATE_SLACK_DAYS",
    "cities_problem",
    "pinned_today",
    "place_name_problem",
    "today",
    "travellers_problem",
    "trip_dates_problem",
]
