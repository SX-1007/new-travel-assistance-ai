"""Places per day by pacing (B33) and by free hours, meals shorter than sights (B32)."""

import pytest

from app.infrastructure.activities.idea_generator import _day_list
from app.modules.activities.models import ActivityPreferences, IdeaRequest
from app.modules.activities.policies import (
    PLACES_PER_DAY_BY_PACING,
    fitting_activities,
    generic_fallback_ideas,
    places_per_day,
    trusted_ideas,
)
from app.modules.activities.tests.test_planning import (
    FakeIdeas,
    TOKYO,
    idea,
    make,
    names,
    request,
    skeleton,
)
from app.modules.itinerary.ground_time import fits_in, places_capacity


FIVE = [
    idea("Senso-ji", 1, 1),
    idea("Tokyo Tower", 1, 2),
    idea("Ichiran", 1, 3, place_type="restaurant"),
    idea("Meiji Shrine", 1, 4),
    idea("Ueno Park", 1, 5),
]


def evening_skeleton():
    """Lands 16:25 on day 1 (free 17:55-22:00, 4.08 h); day 2 is full."""

    days = skeleton(2)
    days[0]["flight"] = [{"arrival_airport": {"id": "HND", "time": "2026-11-01 16:25"}}]
    return days


# ── Pacing ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(("pacing", "kept"), [("relaxed", 3), ("moderate", 5), ("packed", 5)])
def test_no_day_gets_more_places_than_the_pacing_allows(pacing, kept):
    planner, _ = make(FakeIdeas(initial=FIVE))

    result = planner.plan(request(skeleton(1), preferences=ActivityPreferences(pacing=pacing)))

    assert names(result.itinerary[0]) == [
        "Senso-ji", "Tokyo Tower", "Ichiran Shibuya", "Meiji Shrine", "Ueno Park",
    ][:kept]


def test_pacing_caps_and_unknown_pacing():
    assert dict(PLACES_PER_DAY_BY_PACING) == {"relaxed": 3, "moderate": 5, "packed": 8}
    assert places_per_day("chaotic") == 5


def test_the_model_is_told_the_pacing_cap_and_each_days_hours():
    ideas = FakeIdeas(initial=FIVE)
    planner, _ = make(ideas)

    seen = []
    original = ideas.propose
    ideas.propose = lambda req: seen.append(req) or original(req)
    planner.plan(request(evening_skeleton(), preferences=ActivityPreferences(pacing="relaxed")))

    assert seen[0].max_per_day == 3
    assert seen[0].day_limits == (3, 3)
    assert seen[0].day_hours == (4.08, 14.0)


def test_kept_days_of_a_replan_are_not_trimmed_to_the_pacing():
    planned = skeleton(2)
    planned[0]["activities"] = [
        {"name": f"Kept {n}", "type": "attraction", "order": n, "estimated_cost": 0.0,
         "location": {"latitude": TOKYO[0] + n / 10, "longitude": TOKYO[1] + n / 10}}
        for n in range(1, 6)
    ]
    planner, _ = make(FakeIdeas(rescues=[[idea("Tokyo Tower", 2, 1)]]))

    result = planner.plan(
        request(planned, replan_days=(2,), preferences=ActivityPreferences(pacing="relaxed"))
    )

    assert len(result.itinerary[0]["activities"]) == 5
    assert names(result.itinerary[1]) == ["Tokyo Tower"]


# ── Hours: a meal takes 1 h, a sight 1.75 h ─────────────────────────


def test_capacity_counts_up_to_two_meals_as_shorter_places():
    # Found live (B32): the 4.1 h Bangkok evening fitted 3 places, not 2.
    assert places_capacity(4.08) == 3
    assert places_capacity(14.0) == 8
    assert places_capacity(6.5) == 4
    assert places_capacity(1.74) == 0  # a travel day, as in the frontend
    assert fits_in(["attraction", "restaurant", "restaurant"], 4.08)
    assert not fits_in(["attraction", "attraction", "restaurant"], 4.08)
    assert not fits_in(["restaurant"] * 4, 4.08)  # more places than capacity


def test_an_evening_keeps_what_fits_and_skips_a_sight_for_a_later_meal():
    ideas = FakeIdeas(
        initial=[
            idea("Senso-ji", 1, 1),
            idea("Ichiran", 1, 2, place_type="restaurant"),
            idea("Tokyo Tower", 1, 3),  # 4.5 h with it: skipped
            idea("Ueno Park", 1, 4, place_type="restaurant"),
            idea("Meiji Shrine", 2, 1),
        ]
    )
    planner, _ = make(ideas)

    result = planner.plan(request(evening_skeleton()))

    assert names(result.itinerary[0]) == ["Senso-ji", "Ichiran Shibuya", "Ueno Park"]
    assert names(result.itinerary[1]) == ["Meiji Shrine"]


def test_trusted_ideas_and_the_final_trim_use_the_same_hours():
    kept = trusted_ideas(
        [idea("A", 1, 1), idea("B", 1, 2), idea("C", 1, 3, place_type="restaurant")],
        allowed_days=[1],
        cities=["Tokyo"],
        day_hours={1: 3.0},
    )
    trimmed = fitting_activities(
        [{"name": "A", "type": "attraction", "order": 1}, {"name": "C", "type": "restaurant", "order": 2},
         {"name": "B", "type": "attraction", "order": 3}],
        limit=5,
        hours=3.0,
    )

    assert [i.name for i in kept] == ["A", "C"]
    assert [a["name"] for a in trimmed] == ["A", "C"]
    # Unknown hours: only the count limit applies.
    assert len(fitting_activities([{"type": "attraction", "order": n} for n in range(6)], limit=4, hours=None)) == 4


def test_the_generic_fallback_leaves_room_for_the_places_after_it():
    # One sight already on a 4.08 h evening; two more fit only as meals.
    evening = generic_fallback_ideas({1: 2}, cities=["Tokyo"], planned={1: ["attraction"]}, day_hours={1: 4.08})
    # Nothing fits after two sights: nothing is added.
    full = generic_fallback_ideas({1: 1}, cities=["Tokyo"], planned={1: ["attraction"] * 2}, day_hours={1: 4.08})
    # A full day keeps the plain alternation.
    day = generic_fallback_ideas({1: 3}, cities=["Tokyo"], planned={}, day_hours={1: 14.0})

    assert [i.place_type for i in evening] == ["restaurant", "restaurant"]
    assert full == []
    assert [i.place_type for i in day] == ["attraction", "restaurant", "attraction"]


def test_the_day_list_states_each_days_free_hours():
    listed = _day_list(
        IdeaRequest(
            country="Japan",
            cities=("Tokyo",),
            dates=("2026-11-01", "2026-11-02", "2026-11-03"),
            num_people=2,
            hotel_name="",
            preferences=ActivityPreferences(),
            activity_budget=0.0,
            food_budget=0.0,
            max_per_day=5,
            day_limits=(3, 5, 0),
            day_hours=(4.08, 14.0, 0.0),
        )
    )

    assert listed == (
        "Day 1 (2026-11-01): about 4 h free, exactly 3 places; "
        "Day 2 (2026-11-02): about 14 h free, 3 to 5 places; "
        "Day 3 (2026-11-03): travel day, no places"
    )
