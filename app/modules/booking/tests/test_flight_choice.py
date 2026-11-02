"""Flight choice by price and time at the destination (``pick_flight_pair``)."""

from datetime import date, datetime

import pytest

from app.modules.booking.flight_timing import flight_time, free_hours, trip_free_hours
from app.modules.booking.models import FlightAirport, FlightResult
from app.modules.booking.selection import PRICE_TOLERANCE, pick_flight_pair


def leg(number, price, departs, arrives, *, origin="KUL", destination="BKK", stops=0, duration=0):
    return FlightResult(
        airline="AirTest",
        flight_number=number,
        price=price,
        stops=stops,
        duration=duration,
        departure_airport=FlightAirport(id=origin, time=departs),
        arrival_airport=FlightAirport(id=destination, time=arrives),
    )


def back(number, price, departs, arrives="", **kwargs):
    return leg(number, price, departs, arrives or departs, origin="BKK", destination="KUL", **kwargs)


TRIP = {"start_date": "2026-11-15", "end_date": "2026-11-17"}

# Found live (B31): the cheapest pair left 4 h on day 1 and none on day 3.
AFTERNOON_OUT = leg("VZ681", 100.0, "2026-11-15 15:10", "2026-11-15 16:25")
MORNING_OUT = leg("FD352", 110.0, "2026-11-15 07:00", "2026-11-15 08:25")
MORNING_BACK = back("VZ680", 100.0, "2026-11-17 11:00")
EVENING_BACK = back("FD353", 108.0, "2026-11-17 19:00")


def numbers(pair):
    return tuple(flight.flight_number if flight else None for flight in pair)


def test_the_bangkok_trip_gets_the_morning_outbound_and_the_evening_return():
    pair = pick_flight_pair([AFTERNOON_OUT, MORNING_OUT], [MORNING_BACK, EVENING_BACK], **TRIP)

    # 218 is within 15 % of 200, and the traveller gets 34 h instead of 18.
    assert numbers(pair) == ("FD352", "FD353")
    assert not any(flight.over_budget for flight in pair)


def test_a_pair_more_than_the_tolerance_above_the_cheapest_is_not_chosen():
    # The cheapest pair costs 200; this outbound makes a pair just over the limit.
    pricey_out = MORNING_OUT.model_copy(update={"price": 100.0 + 200.0 * PRICE_TOLERANCE + 1})

    pair = pick_flight_pair([AFTERNOON_OUT, pricey_out], [MORNING_BACK], **TRIP)

    assert numbers(pair) == ("VZ681", "VZ680")


def test_within_an_hour_of_the_best_time_the_cheaper_pair_wins():
    # Lands 40 minutes later than the pricier option: not worth paying for.
    cheap_out = leg("C1", 100.0, "2026-11-15 07:00", "2026-11-15 09:05")
    early_out = leg("E1", 112.0, "2026-11-15 06:00", "2026-11-15 08:25")

    pair = pick_flight_pair([cheap_out, early_out], [EVENING_BACK], **TRIP)

    assert numbers(pair) == ("C1", "FD353")


def test_the_transport_budget_bounds_the_choice():
    within = pick_flight_pair(
        [AFTERNOON_OUT, MORNING_OUT], [MORNING_BACK, EVENING_BACK], **TRIP, cap=210.0
    )
    over = pick_flight_pair(
        [AFTERNOON_OUT, MORNING_OUT], [MORNING_BACK, EVENING_BACK], **TRIP, cap=150.0
    )

    # 210 rules out both better legs (218). Two pairs give 26 h: the
    # evening return alone (208) and the morning outbound alone (210).
    assert numbers(within) == ("VZ681", "FD353") and not within[0].over_budget
    # Nothing fits: the cheapest pair, flagged, as before.
    assert numbers(over) == ("VZ681", "VZ680")
    assert over[0].over_budget and over[1].over_budget


def test_an_unreadable_time_never_beats_a_known_one():
    no_time = AFTERNOON_OUT.model_copy(
        update={"flight_number": "X1", "price": 90.0, "arrival_airport": FlightAirport(id="BKK")}
    )

    assert numbers(pick_flight_pair([no_time, AFTERNOON_OUT], [MORNING_BACK], **TRIP)) == ("VZ681", "VZ680")
    # With no readable time anywhere, price alone decides.
    assert numbers(pick_flight_pair([no_time], [MORNING_BACK.model_copy(update={"departure_airport": None})], **TRIP)) == (
        "X1",
        "VZ680",
    )


def test_ties_prefer_fewer_stops_then_a_shorter_journey():
    direct = leg("D1", 100.0, "2026-11-15 07:00", "2026-11-15 08:25", duration=85)
    one_stop = leg("S1", 100.0, "2026-11-15 07:00", "2026-11-15 08:25", stops=1, duration=85)
    slow = leg("L1", 100.0, "2026-11-15 07:00", "2026-11-15 08:25", duration=120)

    assert numbers(pick_flight_pair([one_stop, slow, direct], [EVENING_BACK], **TRIP)) == ("D1", "FD353")


def test_the_30_hour_journey_loses_to_a_direct_flight_within_the_tolerance():
    # Found live (B30): KUL 19:15 → HND 01:00 two days later.
    trip = {"start_date": "2026-11-08", "end_date": "2026-11-13"}
    long_haul = leg("L1", 1000.0, "2026-11-08 19:15", "2026-11-10 01:00", destination="HND")
    direct = leg("D1", 1100.0, "2026-11-08 09:00", "2026-11-08 17:00", destination="HND")
    home = back("R1", 900.0, "2026-11-13 20:00").model_copy(
        update={"departure_airport": FlightAirport(id="HND", time="2026-11-13 20:00")}
    )

    assert numbers(pick_flight_pair([long_haul, direct], [home], **trip)) == ("D1", "R1")


def test_one_missing_side_ranks_the_other_alone():
    assert numbers(pick_flight_pair([AFTERNOON_OUT, MORNING_OUT], [], **TRIP)) == ("FD352", None)
    assert numbers(pick_flight_pair([], [MORNING_BACK, EVENING_BACK], **TRIP)) == (None, "FD353")
    assert pick_flight_pair([], [], **TRIP) == (None, None)


def test_a_same_day_trip_scores_both_legs_on_the_one_day():
    day = {"start_date": "2026-11-15", "end_date": "2026-11-15"}
    returns = [back("R1", 100.0, "2026-11-15 17:00"), back("R2", 105.0, "2026-11-15 21:00")]

    assert numbers(pick_flight_pair([MORNING_OUT], returns, **day)) == ("FD352", "R2")


# ── The time rule itself ────────────────────────────────────────────


def test_free_hours_follow_the_window_rule():
    day = date(2026, 11, 15)

    assert free_hours(day, arrival=None, departure=None) == 14.0
    assert free_hours(day, arrival=datetime(2026, 11, 15, 16, 25), departure=None) == 4.08
    assert free_hours(day, arrival=None, departure=datetime(2026, 11, 15, 11, 0)) == 0.0
    assert free_hours(day, arrival=datetime(2026, 11, 16, 1, 0), departure=None) == 0.0


@pytest.mark.parametrize("dates", [("2026-11-15", "nope"), (None, "2026-11-17")])
def test_invalid_trip_dates_have_no_total(dates):
    assert trip_free_hours(*dates, arrival=None, departure=None) is None


def test_flight_time_reads_either_shape():
    as_json = {"arrival_time": "", "arrival_airport": {"time": "2026-11-15 16:25"}}

    assert flight_time(AFTERNOON_OUT, "arrival") == datetime(2026, 11, 15, 16, 25)
    assert flight_time(as_json, "arrival") == datetime(2026, 11, 15, 16, 25)
    assert flight_time({"departure_time": "soon"}, "departure") is None
    assert flight_time(None, "arrival") is None
