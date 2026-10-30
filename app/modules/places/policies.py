"""Deterministic place-verification rules.

This is the ONE definition of "this provider result really is in the trip
area". Activity planning, chat place search and nearby-amenity origins all
use it, so they can never disagree.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from threading import Lock
from typing import Any, Final

from app.shared.geo import great_circle_km


# A place must lie within this straight-line distance of its trip city.
# 75 km admits normal metropolitan and day-trip places (Tokyo Disneyland in
# Urayasu for Tokyo) while rejecting same-named places in another region
# (Penang resolved as "Kuala Lumpur" is ~293 km away).
#
# Must not exceed the itinerary contract's MAX_ACTIVITY_CITY_DISTANCE_KM;
# a test enforces this.
MAX_KM_FROM_CITY: Final = 75.0

# Zoom used to bias provider search towards the trip city.
CITY_SEARCH_ZOOM: Final = 11

# Never inspect an unlimited candidate list: each candidate costs reverse
# geocoding calls.
MAX_CANDIDATES_INSPECTED: Final = 8


def clean_text(value: Any) -> str:
    """Normalise whitespace; non-strings become ''."""

    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def text_key(value: Any) -> str:
    """Case-insensitive comparison key."""

    return clean_text(value).casefold()


def join_query(*parts: Any) -> str:
    """Comma-join the non-empty parts of a search query."""

    return ", ".join(part for part in map(clean_text, parts) if part)


def canonical_city(
    candidate: Any,
    configured_cities: Sequence[str],
) -> str | None:
    """Bind a free-text city to one exact server-configured trip city.

    A blank candidate is accepted only when the trip has exactly one city.
    """

    canonical: dict[str, str] = {}

    for city in configured_cities:
        clean = clean_text(city)

        if clean:
            canonical.setdefault(clean.casefold(), clean)

    if not canonical:
        return None

    key = text_key(candidate)

    if not key:
        return next(iter(canonical.values())) if len(canonical) == 1 else None

    return canonical.get(key)


def distance_from_anchor_km(
    latitude: float,
    longitude: float,
    anchor: tuple[float, float] | None,
) -> float | None:
    """Rounded km to the anchor, or None without an anchor."""

    if anchor is None:
        return None

    return round(great_circle_km(latitude, longitude, anchor[0], anchor[1]), 2)


def choose_locality(
    locality_names: Iterable[str],
    requested_city: str,
    *,
    nearby_allowed: bool,
) -> str | None:
    """Pick the locality that proves the place belongs to the city.

    An exact match always wins. Otherwise the provider's first locality is
    accepted only when the caller holds distance evidence (``nearby_allowed``).
    """

    names = [clean_text(name) for name in locality_names]
    names = [name for name in names if name]
    requested_key = text_key(requested_city)

    for name in names:
        if name.casefold() == requested_key:
            return name

    if nearby_allowed and names:
        return names[0]

    return None


class SearchQuota:
    """Thread-safe hard cap on provider requests for one run."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("limit must be non-negative")

        self._limit = limit
        self._used = 0
        self._lock = Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._used >= self._limit:
                return False

            self._used += 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def limit(self) -> int:
        return self._limit


__all__ = [
    "CITY_SEARCH_ZOOM",
    "MAX_CANDIDATES_INSPECTED",
    "MAX_KM_FROM_CITY",
    "SearchQuota",
    "canonical_city",
    "choose_locality",
    "clean_text",
    "distance_from_anchor_km",
    "join_query",
    "text_key",
]
