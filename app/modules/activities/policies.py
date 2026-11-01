"""Deterministic activity-planning rules. No providers, no I/O."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from app.modules.activities.models import (
    ActivityPreferences,
    PlaceIdea,
    PlaceType,
)
from app.modules.places.models import VerifiedPlace
from app.modules.places.policies import canonical_city, clean_text, join_query, text_key


MAX_PLACES_PER_DAY: Final = 8

# Every day with time at the destination aims for at least this many places
# (fewer only when its hours fit fewer). Rescue rounds top up short days; a
# day still short after them is kept, not failed.
MIN_PLACES_PER_DAY: Final = 3

# Round 1 retries specific places; round 2 also broadens to generic searches.
MAX_RESCUE_ROUNDS: Final = 2

# Hard cap on PAID provider searches for one plan. Cache hits are free.
MAX_SEARCHES_PER_PLAN: Final = 40

LOOKUP_WORKERS: Final = 4

# Generic searches return several verified places so different days can
# receive different places instead of all getting the top result.
BROAD_SEARCH_LIMIT: Final = 3

MAX_EXCLUDED_NAMES: Final = 50

_PACING_VALUES: Final = frozenset({"relaxed", "moderate", "packed"})

RESTAURANT_ALIASES: Final = frozenset(
    {"restaurant", "restaurants", "food", "dining", "cafe", "café", "eatery"}
)


# ── Inputs ──────────────────────────────────────────────────────────


def _text_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return ()

    cleaned: list[str] = []
    seen: set[str] = set()

    for item in values:
        text = clean_text(item)

        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            cleaned.append(text)

    return tuple(cleaned)


def preferences_from_profile(profile: Mapping[str, Any] | None) -> ActivityPreferences:
    """Map a long-term-memory profile onto planning preferences."""

    if not isinstance(profile, Mapping):
        return ActivityPreferences()

    pacing = text_key(profile.get("travel_pacing"))

    return ActivityPreferences(
        interests=_text_values(profile.get("interests")),
        dietary=_text_values(profile.get("dietary_restrictions")),
        pacing=pacing if pacing in _PACING_VALUES else "moderate",  # type: ignore[arg-type]
    )


def place_type_for(category: Any) -> PlaceType:
    """Map a free-text chat category to the itinerary's activity type."""

    return "restaurant" if text_key(category) in RESTAURANT_ALIASES else "attraction"


def expected_day_numbers(itinerary: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    """Positive integer day numbers, in itinerary order, without repeats."""

    days: list[int] = []

    for day in itinerary:
        number = day.get("day") if isinstance(day, Mapping) else None

        if (
            isinstance(number, int)
            and not isinstance(number, bool)
            and number > 0
            and number not in days
        ):
            days.append(number)

    return tuple(days)


def itinerary_dates(itinerary: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        clean_text(day.get("date")) if isinstance(day, Mapping) else ""
        for day in itinerary
    )


def first_hotel_name(itinerary: Sequence[Mapping[str, Any]]) -> str:
    for day in itinerary:
        hotel = day.get("hotel") if isinstance(day, Mapping) else None

        if isinstance(hotel, Mapping) and clean_text(hotel.get("hotel_name")):
            return clean_text(hotel.get("hotel_name"))

    return ""


def finite_non_negative(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0

    try:
        number = float(value)

    except (TypeError, ValueError, OverflowError):
        return 0.0

    return number if math.isfinite(number) and number > 0 else 0.0


# ── Trust boundary for model output ─────────────────────────────────


def trusted_ideas(
    ideas: Iterable[PlaceIdea],
    *,
    allowed_days: Iterable[int],
    cities: Sequence[str],
    max_per_day: int = MAX_PLACES_PER_DAY,
    day_limits: Mapping[int, int] | None = None,
) -> list[PlaceIdea]:
    """Bind model ideas to server-owned days and cities; cap each day (at
    ``max_per_day``, or the day's own lower limit)."""

    allowed = set(allowed_days)
    per_day: dict[int, int] = {}
    trusted: list[PlaceIdea] = []

    for idea in sorted(ideas, key=lambda item: (item.day, item.order)):
        if idea.day not in allowed:
            continue

        city = canonical_city(idea.city, cities)

        cap = min(max_per_day, (day_limits or {}).get(idea.day, max_per_day))

        if city is None or per_day.get(idea.day, 0) >= cap:
            continue

        per_day[idea.day] = per_day.get(idea.day, 0) + 1
        trusted.append(idea.model_copy(update={"city": city}))

    return trusted


def generic_fallback_ideas(
    shortfall: Mapping[int, int],
    *,
    cities: Sequence[str],
) -> list[PlaceIdea]:
    """Provider-grounded generic places: ``shortfall[day]`` for each day,
    alternating attraction and restaurant (attraction first).

    Cities are assigned round-robin so a multi-city trip spreads fallbacks.
    """

    clean_cities = [city for city in map(clean_text, cities) if city]

    if not clean_cities:
        return []

    ideas: list[PlaceIdea] = []

    for day, needed in shortfall.items():
        city = clean_cities[(day - 1) % len(clean_cities)]

        for index in range(max(0, needed)):
            place_type: PlaceType = "attraction" if index % 2 == 0 else "restaurant"
            ideas.append(
                PlaceIdea(
                    name=f"popular {'tourist attraction' if place_type == 'attraction' else 'restaurant'} in {city}",
                    place_type=place_type,
                    city=city,
                    day=day,
                    order=index + 1,
                    suggested_time="Flexible",
                    estimated_cost=0.0,
                    reason=(
                        "Provider-grounded fallback for a day whose specific "
                        "suggestions could not be verified."
                    ),
                )
            )

    return ideas


# ── Provider queries ────────────────────────────────────────────────


def exact_query(idea: PlaceIdea, country: str) -> str:
    return join_query(idea.name, idea.city, country)


def broad_queries(place_type: PlaceType, city: str, country: str) -> tuple[str, ...]:
    if place_type == "restaurant":
        terms = ("popular restaurant in", "local restaurant in")
    else:
        terms = ("popular tourist attraction in", "museum or landmark in")

    return tuple(join_query(f"{term} {clean_text(city)}", country) for term in terms)


# ── Activity payloads ───────────────────────────────────────────────


def build_activity(
    idea: PlaceIdea,
    place: VerifiedPlace,
    *,
    country: str,
) -> dict[str, Any]:
    """Merge an idea with its verified place into the public activity shape.

    Costs are AI estimates (providers rarely price attractions or meals),
    so ``is_estimated`` is always True.
    """

    candidate = place.candidate

    return {
        "name": candidate.title,
        "type": idea.place_type,
        "description": clean_text(idea.reason),
        "category": candidate.category,
        "rating": candidate.rating,
        # The itinerary contract requires an address; fall back to the
        # verified locality rather than publishing an empty field.
        "address": candidate.address or join_query(place.verified_locality, country),
        "thumbnail": candidate.thumbnail,
        "suggested_time": clean_text(idea.suggested_time),
        "estimated_cost": round(finite_non_negative(idea.estimated_cost), 2),
        "is_estimated": True,
        "order": idea.order,
        "location": place.location_payload(),
    }


def clamp_costs(
    by_day: Mapping[int, Sequence[Mapping[str, Any]]],
    place_type: PlaceType,
    budget: float,
) -> dict[int, list[dict[str, Any]]]:
    """Scale one type's estimates down proportionally so their sum ≤ budget.

    Returns new lists; the input is not mutated. A non-positive budget means
    "no budget to enforce".
    """

    copied = {
        day: [dict(activity) for activity in activities]
        for day, activities in by_day.items()
    }

    if budget <= 0:
        return copied

    subset = [
        activity
        for activities in copied.values()
        for activity in activities
        if activity.get("type") == place_type
    ]
    total = sum(finite_non_negative(a.get("estimated_cost")) for a in subset)

    if total <= budget:
        return copied

    factor = budget / total

    for activity in subset:
        activity["estimated_cost"] = round(
            finite_non_negative(activity.get("estimated_cost")) * factor,
            2,
        )

    return copied


def activities_cost(activities: Any) -> float:
    if not isinstance(activities, list):
        return 0.0

    return sum(
        finite_non_negative(activity.get("estimated_cost"))
        for activity in activities
        if isinstance(activity, Mapping)
    )


def merge_into_itinerary(
    itinerary: Sequence[Mapping[str, Any]],
    by_day: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Project activities into the day skeleton.

    ``day_total_cost`` is recomputed as (existing total − previous activity
    costs + new activity costs), so planning the same itinerary twice never
    double-counts activities.
    """

    updated: list[dict[str, Any]] = []

    for raw_day in itinerary:
        day = dict(raw_day)
        activities = sorted(
            (dict(activity) for activity in by_day.get(day.get("day"), [])),
            key=lambda activity: activity.get("order", math.inf),
        )

        for order, activity in enumerate(activities, start=1):
            activity["order"] = order

        base_total = max(
            0.0,
            finite_non_negative(day.get("day_total_cost"))
            - activities_cost(day.get("activities")),
        )

        day["activities"] = activities
        day["day_total_cost"] = round(base_total + activities_cost(activities), 2)
        updated.append(day)

    return updated


__all__ = [
    "BROAD_SEARCH_LIMIT",
    "LOOKUP_WORKERS",
    "MAX_EXCLUDED_NAMES",
    "MAX_PLACES_PER_DAY",
    "MIN_PLACES_PER_DAY",
    "MAX_RESCUE_ROUNDS",
    "MAX_SEARCHES_PER_PLAN",
    "RESTAURANT_ALIASES",
    "activities_cost",
    "broad_queries",
    "build_activity",
    "clamp_costs",
    "exact_query",
    "expected_day_numbers",
    "finite_non_negative",
    "first_hotel_name",
    "generic_fallback_ideas",
    "itinerary_dates",
    "merge_into_itinerary",
    "place_type_for",
    "preferences_from_profile",
    "trusted_ideas",
]
