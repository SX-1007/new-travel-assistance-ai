"""Brainstorm, verify, rescue and deduplicate daily activities.

Pipeline for one itinerary:

1. An idea generator proposes day-by-day attractions and restaurants.
2. Each idea is searched and destination-verified (``VerifiedPlaceSearch``);
   unverifiable ideas are dropped so the map and routes stay accurate.
3. Days left empty are repaired for at most ``MAX_RESCUE_ROUNDS`` rounds;
   days that already have activities are never touched. The last round also
   broadens to generic provider searches.
4. Duplicates are removed, costs are clamped to the activity and food
   budgets, and activities are projected back into the day skeleton.

Every paid provider search in one plan shares one ``SearchQuota``.

With ``ActivityPlanRequest.replan_days`` set, step 1 is skipped: the other
days keep their activities (no other day may reuse those places) and the
named days are filled by the rescue rounds alone.

Only time actually spent at the destination is planned
(``itinerary.ground_time``): a day whose flight lands after midnight or
leaves before dawn is a travel day with no places, and a partly free day
gets only what fits its hours (a meal takes about an hour, a sight 1.75 h).
No day gets more places than the traveller's pacing allows
(``PLACES_PER_DAY_BY_PACING``), so one full day isn't packed while the
others are light.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Literal

from app.modules.activities.models import (
    ActivityPlanRequest,
    ActivityPlanResult,
    IdeaRequest,
    PlaceIdea,
)
from app.modules.activities.policies import (
    BROAD_SEARCH_LIMIT,
    LOOKUP_WORKERS,
    MAX_EXCLUDED_NAMES,
    MAX_RESCUE_ROUNDS,
    MIN_PLACES_PER_DAY,
    MAX_SEARCHES_PER_PLAN,
    broad_queries,
    build_activity,
    clamp_costs,
    exact_query,
    expected_day_numbers,
    finite_non_negative,
    first_hotel_name,
    fitting_activities,
    generic_fallback_ideas,
    itinerary_dates,
    merge_into_itinerary,
    place_type_for,
    places_per_day,
    trusted_ideas,
)
from app.modules.activities.ports import ActivityIdeaPort
from app.modules.itinerary.activity_identity import activity_identity_key
from app.modules.itinerary.activity_rescue import (
    deduplicate_activity_buckets,
    rescue_activity_buckets,
)
from app.modules.itinerary.ground_time import hours_on_ground, places_capacity
from app.modules.places.exceptions import (
    PlaceSearchQuotaExhausted,
    PlaceSearchUnavailable,
)
from app.modules.places.models import SearchArea, VerifiedPlace
from app.modules.places.policies import (
    SearchQuota,
    canonical_city,
    clean_text,
    join_query,
    text_key,
)
from app.modules.places.service import VerifiedPlaceSearch
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

ActivitiesByDay = dict[int, list[dict[str, Any]]]

# exact: the idea's own name only; exact_then_broad: also generic searches
# (last rescue round); broad: generic searches only (fallback ideas, whose
# "name" is itself a generic query).
LookupMode = Literal["exact", "exact_then_broad", "broad"]


class _PlanLookups:
    """Per-plan provider lookups: memoised, quota-bound and thread-safe.

    Identical queries are searched once per plan (successes AND failures),
    and generic searches hand out a different verified place to each day.
    """

    def __init__(
        self,
        *,
        places: VerifiedPlaceSearch,
        areas: Mapping[str, SearchArea],
        quota: SearchQuota,
    ) -> None:
        self._places = places
        self._areas = areas
        self._quota = quota
        self._memo: dict[tuple[str, str, int], list[VerifiedPlace]] = {}
        self._used_identities: set[str] = set()
        self._pinned_identities: set[str] = set()
        self._lock = Lock()

    def reserve(self, activity: Mapping[str, Any]) -> None:
        """Mark an accepted activity so generic searches skip it."""

        key = activity_identity_key(activity)

        if key is not None:
            with self._lock:
                self._used_identities.add(key)

    def pin(self, activity: Mapping[str, Any]) -> None:
        """Mark a kept activity: no other day may receive the same place."""

        key = activity_identity_key(activity)

        if key is not None:
            with self._lock:
                self._used_identities.add(key)
                self._pinned_identities.add(key)

    def is_pinned(self, activity: Mapping[str, Any]) -> bool:
        key = activity_identity_key(activity)

        with self._lock:
            return key is not None and key in self._pinned_identities

    def find(self, query: str, city: str, *, limit: int) -> list[VerifiedPlace]:
        area = self._areas.get(city)

        if area is None:
            return []

        memo_key = (text_key(query), text_key(city), limit)

        with self._lock:
            if memo_key in self._memo:
                return self._memo[memo_key]

        try:
            results = self._places.search(
                query,
                area=area,
                limit=limit,
                quota=self._quota,
            )

        except PlaceSearchQuotaExhausted:
            # Not memoised: nothing was learned about this query.
            return []

        except PlaceSearchUnavailable:
            safe_log(logger, logging.WARNING, "activities.search_failed")
            results = []

        with self._lock:
            self._memo[memo_key] = results

        return results

    def exact(self, idea: PlaceIdea, country: str) -> VerifiedPlace | None:
        results = self.find(exact_query(idea, country), idea.city, limit=1)
        return results[0] if results else None

    def broad(self, idea: PlaceIdea, country: str) -> VerifiedPlace | None:
        """First verified generic result not already used in this plan."""

        for query in broad_queries(idea.place_type, idea.city, country):
            for place in self.find(query, idea.city, limit=BROAD_SEARCH_LIMIT):
                key = activity_identity_key(
                    {"name": place.name, "location": place.location_payload()}
                )

                with self._lock:
                    if key is not None and key in self._used_identities:
                        continue

                    if key is not None:
                        self._used_identities.add(key)

                return place

        return None


def _kept_activities(
    itinerary: Sequence[Mapping[str, Any]],
    *,
    replan: frozenset[int],
) -> ActivitiesByDay:
    """Existing activities of every day that is not being replanned."""

    kept: ActivitiesByDay = {}

    for day in itinerary:
        number = day.get("day")

        if number in replan or not isinstance(number, int) or isinstance(number, bool):
            continue

        activities = day.get("activities")
        kept[number] = [
            dict(activity)
            for activity in (activities if isinstance(activities, list) else [])
            if isinstance(activity, Mapping)
        ]

    return kept


def _names(by_day: Mapping[int, Sequence[Mapping[str, Any]]]) -> set[str]:
    return {
        clean_text(activity.get("name"))
        for activities in by_day.values()
        for activity in activities
    }


class ActivityPlanningService:
    """Fill an itinerary skeleton with verified attractions and restaurants."""

    def __init__(
        self,
        *,
        ideas: ActivityIdeaPort,
        places: VerifiedPlaceSearch,
        max_searches: int = MAX_SEARCHES_PER_PLAN,
        max_workers: int = LOOKUP_WORKERS,
        min_places_per_day: int = MIN_PLACES_PER_DAY,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if min_places_per_day < 1:
            raise ValueError("min_places_per_day must be at least 1")

        self._ideas = ideas
        self._places = places
        self._max_searches = max_searches
        self._max_workers = max_workers
        self._min_places = min_places_per_day

    def plan(self, request: ActivityPlanRequest) -> ActivityPlanResult:
        itinerary = [dict(day) for day in request.itinerary if isinstance(day, Mapping)]
        all_days = expected_day_numbers(itinerary)
        requested = frozenset(request.replan_days) & frozenset(all_days)

        if request.replan_days and not requested:
            # Nothing the caller asked for exists in this itinerary.
            return ActivityPlanResult(
                itinerary=itinerary,
                missing_days=(),
                rescue_rounds_used=0,
                searches_used=0,
            )

        # Free hours per day (days without a date are unconstrained), and
        # the most places each day takes: what fits, never above the pacing.
        hours = hours_on_ground(itinerary)
        per_day = places_per_day(request.preferences.pacing)
        limits = {
            day: min(per_day, places_capacity(hours[day])) if day in hours else per_day
            for day in all_days
        }
        # Travel days being (re)planned end up empty, never "missing".
        cleared = {
            day: []
            for day in all_days
            if limits[day] == 0 and (not request.replan_days or day in requested)
        }
        expected_days = tuple(day for day in all_days if limits[day] > 0)
        replan = requested & frozenset(expected_days)

        if not expected_days or (request.replan_days and not replan):
            # Nothing at the destination left to plan: only clear travel days.
            untouched = _kept_activities(itinerary, replan=frozenset(cleared))
            return ActivityPlanResult(
                itinerary=merge_into_itinerary(itinerary, {**untouched, **cleared}),
                missing_days=(),
                rescue_rounds_used=0,
                searches_used=0,
            )

        cities = tuple(city for city in map(clean_text, request.cities) if city)
        country = clean_text(request.country)

        areas = {
            city: area
            for city in cities
            if (area := self._places.area(country=country, city=city)) is not None
        }

        if not expected_days or not areas:
            safe_log(logger, logging.WARNING, "activities.destination_unresolved")
            return ActivityPlanResult(
                itinerary=itinerary,
                missing_days=expected_days,
                rescue_rounds_used=0,
                searches_used=0,
            )

        quota = SearchQuota(self._max_searches)
        lookups = _PlanLookups(places=self._places, areas=areas, quota=quota)
        trusted_cities = tuple(areas)

        idea_request = IdeaRequest(
            country=country,
            cities=trusted_cities,
            dates=itinerary_dates(itinerary),
            day_limits=tuple(limits.get(day.get("day"), per_day) for day in itinerary),
            day_hours=tuple(hours.get(day.get("day")) for day in itinerary),
            num_people=max(1, int(request.num_people or 1)),
            hotel_name=first_hotel_name(itinerary),
            preferences=request.preferences,
            activity_budget=finite_non_negative(request.activity_budget),
            food_budget=finite_non_negative(request.food_budget),
            max_per_day=per_day,
        )

        tried_names: set[str] = set()
        accepted_names: set[str] = set()

        if replan:
            # ── 1. Keep every other day; its places can't be reused ────
            # The replanned days start empty, so the rescue rounds below fill
            # exactly those days with one targeted idea request each round.
            by_day = _kept_activities(itinerary, replan=replan)

            for activities in by_day.values():
                for activity in activities:
                    lookups.pin(activity)

        else:
            # ── 1. Normal brainstorm + strict verification ────────────
            initial = self._safe_ideas(lambda: self._ideas.propose(idea_request))
            tried_names.update(idea.name for idea in initial)

            by_day = self._verify(
                trusted_ideas(
                    initial,
                    allowed_days=expected_days,
                    cities=trusted_cities,
                    max_per_day=per_day,
                    day_limits=limits,
                    day_hours=hours,
                ),
                lookups=lookups,
                country=country,
                mode="exact",
            )

        accepted_names.update(_names(by_day))
        # Place types each day has so far, so the last round's generic
        # fallback picks what still fits (the final trim below enforces it).
        planned_types = {
            day: [activity.get("type") for activity in activities] for day, activities in by_day.items()
        }

        # ── 2. Top up the days below their target ─────────────────────
        def rescue_round(shortfall: Mapping[int, int], round_number: int) -> ActivitiesByDay:
            missing = tuple(shortfall)
            excluded = sorted(name for name in tried_names | accepted_names if name)

            ideas = self._safe_ideas(
                lambda: self._ideas.propose_rescue(
                    idea_request,
                    missing_days=missing,
                    excluded_places=excluded[-MAX_EXCLUDED_NAMES:],
                )
            )
            tried_names.update(idea.name for idea in ideas)

            trusted = trusted_ideas(
                ideas,
                allowed_days=missing,
                cities=trusted_cities,
                max_per_day=per_day,
                day_limits=limits,
                day_hours=hours,
            )
            final_round = round_number == MAX_RESCUE_ROUNDS

            rescued = self._verify(
                trusted,
                lookups=lookups,
                country=country,
                mode="exact_then_broad" if final_round else "exact",
            )

            # Built only AFTER model output passed the trust boundary, so a
            # discarded idea (wrong city/day) cannot suppress the fallback.
            if final_round:
                still_needed = {
                    day: needed - len(rescued.get(day, ()))
                    for day, needed in shortfall.items()
                    if needed > len(rescued.get(day, ()))
                }
                planned_now = {
                    day: [*planned_types.get(day, ()), *(a.get("type") for a in rescued.get(day, ()))]
                    for day in still_needed
                }
                fallback = self._verify(
                    generic_fallback_ideas(
                        still_needed,
                        cities=trusted_cities,
                        planned=planned_now,
                        day_hours=hours,
                    ),
                    lookups=lookups,
                    country=country,
                    mode="broad",
                )

                for day, activities in fallback.items():
                    rescued.setdefault(day, []).extend(activities)

            accepted_names.update(_names(rescued))

            for day, activities in rescued.items():
                planned_types.setdefault(day, []).extend(activity.get("type") for activity in activities)

            return rescued

        rescue = rescue_activity_buckets(
            # When replanning, only the named days may change, even if a
            # kept day happens to be empty.
            tuple(day for day in expected_days if day in replan) if replan else expected_days,
            by_day,
            rescue_round=rescue_round,
            max_rounds=MAX_RESCUE_ROUNDS,
            targets={day: min(self._min_places, limits[day]) for day in expected_days},
        )

        # ── 3. Final dedupe, budget clamp, projection ─────────────────
        cleaned, _ = deduplicate_activity_buckets(rescue.by_day)
        # Rescued and generic places can overfill a day: keep, in visiting
        # order, what fits its pacing cap and its free hours. Kept days of a
        # replan are left exactly as they are.
        planned_days = replan or frozenset(expected_days)
        cleaned = {
            day: (
                fitting_activities(activities, limit=limits.get(day, per_day), hours=hours.get(day))
                if day in planned_days
                else activities
            )
            for day, activities in cleaned.items()
        }
        cleaned.update(cleared)
        cleaned = clamp_costs(cleaned, "attraction", idea_request.activity_budget)
        cleaned = clamp_costs(cleaned, "restaurant", idea_request.food_budget)

        if rescue.missing_days:
            safe_log(logger, logging.WARNING, "activities.rescue_exhausted")

        if rescue.short_days:
            # Usable, just fewer places than the target: not a failure.
            safe_log(logger, logging.WARNING, "activities.days_below_target")

        logger.info(
            "activities.search_quota used=%d limit=%d",
            quota.used,
            quota.limit,
        )

        return ActivityPlanResult(
            itinerary=merge_into_itinerary(itinerary, cleaned),
            missing_days=rescue.missing_days,
            rescue_rounds_used=rescue.rounds_used,
            searches_used=quota.used,
        )

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _safe_ideas(call: Callable[[], list[PlaceIdea]]) -> list[PlaceIdea]:
        try:
            ideas = call()

        except Exception:
            safe_log(logger, logging.WARNING, "activities.ideas_failed")
            return []

        return [idea for idea in ideas or [] if isinstance(idea, PlaceIdea)]

    def _verify(
        self,
        ideas: Sequence[PlaceIdea],
        *,
        lookups: _PlanLookups,
        country: str,
        mode: LookupMode,
    ) -> ActivitiesByDay:
        """Look ideas up in parallel; keep verified ones, grouped by day."""

        if not ideas:
            return {}

        def lookup(idea: PlaceIdea) -> VerifiedPlace | None:
            place = lookups.exact(idea, country) if mode != "broad" else None

            if place is None and mode != "exact":
                place = lookups.broad(idea, country)

            return place

        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(ideas)),
            thread_name_prefix="activities",
        ) as executor:
            places = list(executor.map(lookup, ideas))

        by_day: ActivitiesByDay = {}
        dropped = 0

        for idea, place in zip(ideas, places):
            if place is None:
                dropped += 1
                continue

            activity = build_activity(idea, place, country=country)

            if lookups.is_pinned(activity):
                # Already on a day that is being kept.
                dropped += 1
                continue

            lookups.reserve(activity)
            by_day.setdefault(idea.day, []).append(activity)

        if dropped:
            safe_log(logger, logging.INFO, "activities.unverified_dropped")

        return by_day


class ActivitySearchService:
    """Chat-time attraction/restaurant search with the same verification."""

    def __init__(self, *, places: VerifiedPlaceSearch, limit: int = 3) -> None:
        self._places = places
        self._limit = limit

    def search(
        self,
        *,
        place_type: str,
        query: str,
        country: str,
        cities: Sequence[str],
        city_hint: str = "",
    ) -> dict[str, Any]:
        """Return up to ``limit`` verified, itinerary-ready candidates.

        The trip's configured city stays the trusted anchor. ``city_hint`` is
        only search text, so a nearby day-trip place (Urayasu for Tokyo) can
        be found, but a model-invented city can never move the anchor.
        """

        configured = [city for city in map(clean_text, cities) if city]

        if not configured:
            return {"error": "The trip has no configured destination city."}

        matched = canonical_city(city_hint, configured) if clean_text(city_hint) else None
        anchor_city = matched or configured[0]
        search_city = matched or clean_text(city_hint) or anchor_city

        area = self._places.area(country=country, city=anchor_city)

        if area is None:
            return {"error": "Destination country could not be verified."}

        full_query = join_query(query, search_city, area.country)

        try:
            places = self._places.search(full_query, area=area, limit=self._limit)

        except PlaceSearchUnavailable:
            return {"error": "Place search failed."}

        activity_type = place_type_for(place_type)

        results = [
            {
                "name": place.name,
                "type": activity_type,
                "description": place.candidate.category,
                "rating": place.candidate.rating,
                "address": place.candidate.address
                or join_query(place.verified_locality, area.country),
                "thumbnail": place.candidate.thumbnail,
                # Providers rarely price places; the itinerary requires a
                # number, so it is an explicit estimate.
                "estimated_cost": 0.0,
                "is_estimated": True,
                "location": place.location_payload(),
            }
            for place in places
        ]

        if not results:
            return {"error": f"No places found for '{full_query}'.", "count": 0}

        return {"category": activity_type, "results": results, "count": len(results)}


__all__ = [
    "ActivityPlanningService",
    "ActivitySearchService",
]
