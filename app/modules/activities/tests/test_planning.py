"""Activity planning and chat search against fake idea and place providers.

Place verification is the REAL ``VerifiedPlaceSearch``; only the network
edges (search, geocoder, locator) are faked.
"""

from types import SimpleNamespace

import pytest

from app.infrastructure.activities.idea_generator import (
    GeminiActivityIdeaGenerator,
    _RawIdea,
    _RawPlan,
    _to_ideas,
)
from app.infrastructure.llm.chat import ChatBackend
from app.modules.activities.models import (
    ActivityPlanRequest,
    ActivityPreferences,
    IdeaRequest,
    PlaceIdea,
)
from app.modules.activities.policies import (
    clamp_costs,
    generic_fallback_ideas,
    merge_into_itinerary,
    place_type_for,
    preferences_from_profile,
)
from app.modules.activities.service import ActivityPlanningService, ActivitySearchService
from app.modules.itinerary.activity_rescue import short_activity_days
from app.modules.itinerary.models import ActivityResult
from app.modules.places.exceptions import PlaceSearchUnavailable
from app.modules.places.models import PlaceCandidate
from app.modules.places.service import VerifiedPlaceSearch
from app.orchestration.workflows.activities import run_activity_planning
from app.shared.geo import GeocodedPlace


TOKYO = (35.6812, 139.7671)


def _candidate(title, offset, *, address="Tokyo address"):
    return PlaceCandidate(
        title=title,
        latitude=TOKYO[0] + offset,
        longitude=TOKYO[1] + offset,
        address=address,
        category="Tourist attraction",
        rating=4.4,
    )


PLACES = {
    "senso-ji": _candidate("Senso-ji", 0.01),
    "tokyo tower": _candidate("Tokyo Tower", 0.02),
    "ichiran": _candidate("Ichiran Shibuya", 0.03, address=""),
    "meiji shrine": _candidate("Meiji Shrine", 0.04),
    "ueno park": _candidate("Ueno Park", 0.05),
}
GENERIC = [_candidate("Generic A", 0.06), _candidate("Generic B", 0.07), _candidate("Generic C", 0.08)]
FOREIGN = PlaceCandidate(title="Osaka Castle", latitude=34.6873, longitude=135.5262)


class FakeSearch:
    def __init__(self):
        self.calls = []

    def search(self, query, *, near=None, zoom=11, quota=None):
        self.calls.append(query)
        if quota is not None and not quota.try_acquire():
            from app.modules.places.exceptions import PlaceSearchQuotaExhausted

            raise PlaceSearchQuotaExhausted("spent")
        head = query.split(",")[0].strip().casefold()
        if head.startswith(("popular", "museum", "local")):
            return list(GENERIC)
        if head == "osaka castle":
            return [FOREIGN]
        return [PLACES[head]] if head in PLACES else []


class FakeGeocoder:
    def geocode(self, query, *, proximity=None):
        return GeocodedPlace(latitude=TOKYO[0], longitude=TOKYO[1], place_name=query)


class FakeLocator:
    def resolve_country_code(self, latitude, longitude):
        return "JP"

    def resolve_locality_names(self, latitude, longitude, destination_country_code):
        if latitude < 35.0:
            return ("Osaka",)
        return ("Tokyo",)


class FakeIdeas:
    def __init__(self, initial=(), rescues=(), error=None):
        self.initial = list(initial)
        self.rescues = [list(batch) for batch in rescues]
        self.error = error
        self.rescue_calls = []

    def propose(self, request):
        if self.error:
            raise self.error
        return list(self.initial)

    def propose_rescue(self, request, *, missing_days, excluded_places):
        self.rescue_calls.append((tuple(missing_days), tuple(excluded_places)))
        return self.rescues.pop(0) if self.rescues else []


def idea(name, day, order, *, place_type="attraction", city="Tokyo", cost=0.0):
    return PlaceIdea(
        name=name,
        place_type=place_type,
        city=city,
        day=day,
        order=order,
        estimated_cost=cost,
        reason="Because.",
    )


def skeleton(days=2):
    return [
        {"day": n, "date": f"2026-11-0{n}", "activities": [], "day_total_cost": 100.0}
        for n in range(1, days + 1)
    ]


def make(ideas, *, max_searches=40, min_places=1):
    # min_places=1: most tests here are about other rules, and top-ups to the
    # real minimum (3) would add generic places to their results. The
    # minimum has its own tests below.
    search = FakeSearch()
    places = VerifiedPlaceSearch(search=search, geocoder=FakeGeocoder(), locator=FakeLocator())
    planner = ActivityPlanningService(
        ideas=ideas, places=places, max_searches=max_searches, min_places_per_day=min_places
    )
    return planner, search


def request(itinerary, **overrides):
    values = dict(
        itinerary=itinerary,
        country="Japan",
        cities=("Tokyo",),
        num_people=2,
        activity_budget=1000.0,
        food_budget=1000.0,
    )
    values.update(overrides)
    return ActivityPlanRequest(**values)


def names(day):
    return [activity["name"] for activity in day["activities"]]


def test_plan_produces_contract_valid_ordered_activities():
    ideas = FakeIdeas(
        initial=[
            idea("Tokyo Tower", 1, 2, cost=30),
            idea("Senso-ji", 1, 1, cost=0),
            idea("Ichiran", 1, 3, place_type="restaurant", cost=40),
            idea("Meiji Shrine", 2, 1, cost=10),
        ]
    )
    planner, _ = make(ideas)

    result = planner.plan(request(skeleton()))
    day1, day2 = result.itinerary

    assert names(day1) == ["Senso-ji", "Tokyo Tower", "Ichiran Shibuya"]
    assert [a["order"] for a in day1["activities"]] == [1, 2, 3]
    assert day1["day_total_cost"] == 170.0
    assert names(day2) == ["Meiji Shrine"]
    assert result.missing_days == ()
    assert result.rescue_rounds_used == 0
    for day in result.itinerary:
        for activity in day["activities"]:
            ActivityResult.model_validate(activity)
    # Empty provider address falls back to the verified locality.
    assert day1["activities"][2]["address"] == "Tokyo, Japan"


def test_unverifiable_and_untrusted_ideas_are_dropped():
    ideas = FakeIdeas(
        initial=[
            idea("Senso-ji", 1, 1),
            idea("Nowhere Place", 1, 2),
            idea("Osaka Castle", 1, 3),
            idea("Tokyo Tower", 1, 4, city="Kyoto"),
            idea("Ueno Park", 9, 1),
        ]
    )
    planner, _ = make(ideas)

    result = planner.plan(request(skeleton(1)))

    assert names(result.itinerary[0]) == ["Senso-ji"]


def test_only_empty_days_are_rescued():
    ideas = FakeIdeas(
        initial=[idea("Senso-ji", 1, 1), idea("Nowhere Place", 2, 1)],
        rescues=[[idea("Ueno Park", 2, 1), idea("Tokyo Tower", 1, 2)]],
    )
    planner, _ = make(ideas)

    result = planner.plan(request(skeleton()))

    assert ideas.rescue_calls[0][0] == (2,)
    assert "Nowhere Place" in ideas.rescue_calls[0][1]
    assert names(result.itinerary[0]) == ["Senso-ji"]
    assert names(result.itinerary[1]) == ["Ueno Park"]
    assert result.rescue_rounds_used == 1


def test_short_days_are_topped_up_to_the_minimum():
    # Found live: a full day kept the one place that verified.
    ideas = FakeIdeas(
        initial=[idea("Senso-ji", 1, 1), idea("Meiji Shrine", 2, 1), idea("Tokyo Tower", 2, 2),
                 idea("Ueno Park", 2, 3)],
        rescues=[[idea("Ichiran", 1, 1, place_type="restaurant")], []],
    )
    planner, _ = make(ideas, min_places=3)

    result = planner.plan(request(skeleton()))
    day1, day2 = result.itinerary

    # Day 2 already has 3; only day 1 (2 short) is rescued, and the last
    # round's generic search fills what the ideas couldn't.
    assert [call[0] for call in ideas.rescue_calls] == [(1,), (1,)]
    assert names(day1) == ["Senso-ji", "Ichiran Shibuya", "Generic A"]
    assert names(day2) == ["Meiji Shrine", "Tokyo Tower", "Ueno Park"]
    assert result.missing_days == ()


def test_a_day_still_short_after_rescue_is_kept_not_failed():
    ideas = FakeIdeas(initial=[idea("Senso-ji", 1, 1)], rescues=[[], []])
    # One paid search: the exact lookup; nothing left for generic searches.
    planner, _ = make(ideas, min_places=3, max_searches=1)

    result = planner.plan(request(skeleton(1)))

    assert names(result.itinerary[0]) == ["Senso-ji"]
    assert result.missing_days == ()


def test_generic_fallback_alternates_attractions_and_restaurants():
    ideas = generic_fallback_ideas({1: 3, 2: 0}, cities=["Tokyo"])

    assert [(i.day, i.place_type) for i in ideas] == [
        (1, "attraction"), (1, "restaurant"), (1, "attraction"),
    ]


def test_short_activity_days_counts_what_each_day_still_needs():
    by_day = {1: [{}], 2: [{}, {}, {}], 3: []}

    assert short_activity_days((1, 2, 3), by_day, {1: 3, 2: 3, 3: 2}) == {1: 2, 3: 2}
    # Without targets only empty days are short.
    assert short_activity_days((1, 2, 3), by_day) == {3: 1}


def test_generic_fallback_gives_each_day_a_different_place():
    ideas = FakeIdeas(initial=[], rescues=[[], []])
    planner, search = make(ideas)

    result = planner.plan(request(skeleton(3)))

    picked = [names(day) for day in result.itinerary]
    assert picked == [["Generic A"], ["Generic B"], ["Generic C"]]
    assert result.missing_days == ()
    # The shared generic query is searched once, not once per day.
    assert search.calls.count("popular tourist attraction in Tokyo, Japan") == 1


def test_idea_generator_failure_still_falls_back():
    planner, _ = make(FakeIdeas(error=RuntimeError("quota")))

    result = planner.plan(request(skeleton(1)))

    assert names(result.itinerary[0]) == ["Generic A"]


def test_search_quota_caps_paid_searches():
    ideas = FakeIdeas(initial=[idea(n, 1, i) for i, n in enumerate(PLACES, start=1)])
    planner, search = make(ideas, max_searches=2)

    result = planner.plan(request(skeleton(1)))

    assert result.searches_used == 2
    assert len(names(result.itinerary[0])) <= 2


def test_replanning_does_not_double_count_day_totals():
    ideas = FakeIdeas(initial=[idea("Senso-ji", 1, 1, cost=50)])
    planner, _ = make(ideas)

    first = planner.plan(request(skeleton(1)))
    second = planner.plan(request(first.itinerary))

    assert first.itinerary[0]["day_total_cost"] == 150.0
    assert second.itinerary[0]["day_total_cost"] == 150.0


def test_costs_are_clamped_to_budgets():
    ideas = FakeIdeas(
        initial=[idea("Senso-ji", 1, 1, cost=300), idea("Tokyo Tower", 1, 2, cost=100)]
    )
    planner, _ = make(ideas)

    result = planner.plan(request(skeleton(1), activity_budget=200.0))

    costs = [a["estimated_cost"] for a in result.itinerary[0]["activities"]]
    assert costs == [150.0, 50.0]


def test_unknown_destination_leaves_itinerary_unchanged():
    planner, search = make(FakeIdeas(initial=[idea("Senso-ji", 1, 1)]))

    result = planner.plan(request(skeleton(1), country="Atlantis"))

    assert result.itinerary == skeleton(1)
    assert result.missing_days == (1,)
    assert search.calls == []


# ── Pure helpers ────────────────────────────────────────────────────


def test_clamp_costs_does_not_mutate_input():
    by_day = {1: [{"type": "attraction", "estimated_cost": 10.0}]}
    clamp_costs(by_day, "attraction", 5.0)
    assert by_day[1][0]["estimated_cost"] == 10.0


def test_merge_renumbers_and_keeps_days_without_activities():
    merged = merge_into_itinerary(skeleton(2), {1: [{"order": 5, "estimated_cost": 1}]})
    assert merged[0]["activities"][0]["order"] == 1
    assert merged[1]["activities"] == []


def test_preferences_from_profile():
    prefs = preferences_from_profile(
        {"interests": ["Food", "food", " anime "], "dietary_restrictions": "halal", "travel_pacing": "Packed"}
    )
    assert prefs == ActivityPreferences(interests=("Food", "anime"), dietary=("halal",), pacing="packed")
    assert preferences_from_profile({"travel_pacing": "chaotic"}).pacing == "moderate"
    assert preferences_from_profile(None) == ActivityPreferences()


def test_place_type_for():
    assert place_type_for("Restaurant") == "restaurant"
    assert place_type_for("museum") == "attraction"


# ── Chat search ─────────────────────────────────────────────────────


def chat_search():
    search = FakeSearch()
    places = VerifiedPlaceSearch(search=search, geocoder=FakeGeocoder(), locator=FakeLocator())
    return ActivitySearchService(places=places), search


def test_chat_search_returns_itinerary_ready_results():
    service, search = chat_search()

    result = service.search(place_type="attraction", query="Senso-ji", country="Japan", cities=["Tokyo"])

    assert result["count"] == 1
    activity = {**result["results"][0], "order": 1}
    ActivityResult.model_validate(activity)
    assert search.calls == ["Senso-ji, Tokyo, Japan"]


def test_unmatched_city_hint_is_search_text_only():
    service, search = chat_search()

    result = service.search(
        place_type="attraction", query="Senso-ji", country="Japan", cities=["Tokyo"], city_hint="Urayasu"
    )

    assert search.calls == ["Senso-ji, Urayasu, Japan"]
    assert result["results"][0]["location"]["requested_city"] == "Tokyo"


def test_chat_search_errors():
    service, _ = chat_search()

    assert "error" in service.search(place_type="x", query="Nowhere", country="Japan", cities=["Tokyo"])
    assert "error" in service.search(place_type="x", query="Senso-ji", country="Japan", cities=[])
    assert "error" in service.search(place_type="x", query="Senso-ji", country="Atlantis", cities=["Tokyo"])


def test_chat_search_provider_failure():
    class Broken(FakeSearch):
        def search(self, *args, **kwargs):
            raise PlaceSearchUnavailable("down")

    places = VerifiedPlaceSearch(search=Broken(), geocoder=FakeGeocoder(), locator=FakeLocator())
    result = ActivitySearchService(places=places).search(
        place_type="attraction", query="Senso-ji", country="Japan", cities=["Tokyo"]
    )

    assert result == {"error": "Place search failed."}


# ── Gemini adapter (no network) ─────────────────────────────────────


def test_invalid_llm_ideas_are_dropped_individually():
    plan = _RawPlan(
        places=[
            _RawIdea(name="Senso-ji", place_type="attraction", day=1, order=1),
            _RawIdea(name="Bad cost", place_type="attraction", day=1, order=2, estimated_cost=-5),
            _RawIdea(name="Bad type", place_type="shop", day=1, order=3),
            _RawIdea(name="Bad day", place_type="attraction", day=0, order=1),
        ]
    )

    assert [i.name for i in _to_ideas(plan)] == ["Senso-ji"]


def test_rescue_keeps_only_requested_days():
    generator = GeminiActivityIdeaGenerator(backends=(ChatBackend("gemini", "key", "model"),))
    generator._rescue = SimpleNamespace(
        invoke=lambda inputs: _RawPlan(
            places=[
                _RawIdea(name="A", place_type="attraction", day=2, order=1),
                _RawIdea(name="B", place_type="attraction", day=1, order=1),
            ]
        )
    )
    request_ = IdeaRequest(
        country="Japan",
        cities=("Tokyo",),
        dates=("2026-11-01", "2026-11-02"),
        num_people=2,
        hotel_name="",
        preferences=ActivityPreferences(),
        activity_budget=0.0,
        food_budget=0.0,
        max_per_day=8,
    )

    ideas = generator.propose_rescue(request_, missing_days=[2], excluded_places=[])

    assert [i.name for i in ideas] == ["A"]


# ── Workflow adapter ────────────────────────────────────────────────


def test_workflow_maps_state_and_profile():
    planner, _ = make(FakeIdeas(initial=[idea("Senso-ji", 1, 1)]))
    state = SimpleNamespace(
        draft_itinerary=skeleton(1),
        country="Japan",
        city=["Tokyo"],
        num_people=2,
        budget_allocation={"activity": 500.0, "food": 300.0},
    )

    update = run_activity_planning(state, planner=planner, user_profile={"interests": ["temples"]})

    assert set(update) == {"draft_itinerary"}
    assert names(update["draft_itinerary"][0]) == ["Senso-ji"]


def test_workflow_without_itinerary_is_a_no_op():
    planner, _ = make(FakeIdeas())
    assert run_activity_planning(SimpleNamespace(draft_itinerary=[]), planner=planner) == {}


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError):
        ActivityPlanningService(ideas=FakeIdeas(), places=None, max_workers=workers)


def test_replan_days_keeps_other_days_and_never_reuses_their_places():
    planned = skeleton(3)
    planned[0]["activities"] = [
        {"name": "Senso-ji", "type": "attraction", "order": 1, "estimated_cost": 0.0,
         "location": {"latitude": TOKYO[0] + 0.01, "longitude": TOKYO[1] + 0.01}},
    ]
    planned[0]["route"] = {"kept": True}
    planned[1]["activities"] = [
        {"name": "Bad Place", "type": "attraction", "order": 1, "estimated_cost": 0.0},
    ]
    # The rescue proposes day 1's place again; it must be rejected.
    ideas = FakeIdeas(rescues=[[idea("Senso-ji", 2, 1), idea("Tokyo Tower", 2, 2)]])
    planner, _ = make(ideas)

    result = planner.plan(request(planned, replan_days=(2,)))
    day1, day2, day3 = result.itinerary

    assert names(day1) == ["Senso-ji"]
    assert day1["route"] == {"kept": True}
    assert names(day2) == ["Tokyo Tower"]
    # Day 3 was not asked for and is left exactly as it was.
    assert day3["activities"] == []
    assert ideas.rescue_calls[0][0] == (2,)
    assert "Senso-ji" in ideas.rescue_calls[0][1]
    assert result.missing_days == ()


def test_replan_days_that_do_not_exist_change_nothing():
    ideas = FakeIdeas()
    planner, search = make(ideas)

    result = planner.plan(request(skeleton(2), replan_days=(9,)))

    assert result.itinerary == skeleton(2)
    assert search.calls == [] and ideas.rescue_calls == []
