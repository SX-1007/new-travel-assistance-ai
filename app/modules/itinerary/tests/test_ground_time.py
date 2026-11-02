"""Usable hours per day from real flight times."""

from app.modules.itinerary.ground_time import day_place_limits, hours_on_ground, travel_days


def flight(departs, arrives):
    return {
        "departure_airport": {"id": "KUL", "time": departs},
        "arrival_airport": {"id": "HND", "time": arrives},
    }


def trip(outbound, returning, days=3):
    itinerary = [{"day": n, "date": f"2026-12-{9 + n:02d}", "flight": None} for n in range(1, days + 1)]
    itinerary[0]["flight"] = [outbound] if outbound else None
    if returning:
        itinerary[-1]["flight"] = (itinerary[-1]["flight"] or []) + [returning]
    return itinerary


def test_the_live_smoke_case_overnight_arrival_and_pre_dawn_departure():
    # Found live: lands 01:00 on day 2, leaves 02:15 on day 3.
    itinerary = trip(flight("2026-12-10 13:35", "2026-12-11 01:00"), flight("2026-12-12 02:15", "2026-12-12 18:15"))

    assert hours_on_ground(itinerary) == {1: 0.0, 2: 14.0, 3: 0.0}
    assert day_place_limits(itinerary) == {1: 0, 2: 8, 3: 0}
    assert travel_days(itinerary) == {1, 3}


def test_partly_free_boundary_days_get_their_remaining_hours():
    itinerary = trip(flight("2026-12-10 06:00", "2026-12-10 14:00"), flight("2026-12-12 19:00", "2026-12-13 01:00"))

    # Ground from 15:30 to 22:00, and 08:00 until 16:00 (3 h before 19:00).
    assert hours_on_ground(itinerary) == {1: 6.5, 2: 14.0, 3: 8.0}
    # Two meals (1 h each) and the rest sights (1.75 h): 6.5 h → 2 + 2.
    assert day_place_limits(itinerary) == {1: 4, 2: 8, 3: 5}


def test_a_same_day_trip_uses_both_legs_of_day_one():
    day_trip = [{
        "day": 1,
        "date": "2026-12-10",
        "flight": [flight("2026-12-10 06:00", "2026-12-10 08:00"), flight("2026-12-10 21:00", "2026-12-10 23:00")],
    }]

    assert hours_on_ground(day_trip) == {1: 8.5}  # 09:30 to 18:00


def test_unknown_times_or_dates_impose_no_limit():
    no_times = trip({"departure_airport": {"time": "soon"}}, None)
    no_dates = [{"day": 1, "date": None}, {"day": 2, "date": "2026-12-11"}]

    assert hours_on_ground(no_times) == {1: 14.0, 2: 14.0, 3: 14.0}
    assert hours_on_ground(no_dates) == {2: 14.0}
    assert hours_on_ground([]) == {}
