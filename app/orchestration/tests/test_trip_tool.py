"""update_trip_details and the re-validation used before applying it."""

import pytest

from app.orchestration.state import AgentState
from app.orchestration.tools.trip import update_trip_details, validated_trip_updates


STATE = AgentState(
    origin_country="Malaysia",
    country="Japan",
    city=["Tokyo"],
    num_people=2,
    start_date="2026-11-01",
    end_date="2026-11-05",
)


def call(**args):
    return update_trip_details.func(state=STATE, **args)


def test_only_real_changes_are_returned():
    result = call(num_people=4, country="japan", city=[" tokyo "], end_date="2026-11-06")

    assert result["status"] == "accepted"
    assert result["updates"] == {"num_people": 4, "end_date": "2026-11-06"}
    assert "re-planned" in result["message"]


def test_no_op_change_does_not_trigger_a_replan():
    result = call(country="Japan", num_people=2)

    assert result["status"] == "unchanged"
    assert "updates" not in result


def test_origin_can_change():
    assert call(origin_country="Singapore")["updates"] == {"origin_country": "Singapore"}


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"num_people": 0}, "num_people"),
        ({"start_date": "2026-13-01"}, "start_date"),
        ({"start_date": "2026-11-1"}, "start_date"),
        ({"city": ["  "]}, "city"),
        ({}, "No trip details"),
    ],
)
def test_invalid_input_is_refused(args, fragment):
    assert fragment in call(**args)["error"]


def test_dates_are_checked_against_the_current_trip():
    assert call(start_date="2026-11-10")["error"] == (
        "Invalid trip details: The end date must be on or after the start date."
    )


def test_duplicate_cities_are_collapsed():
    assert call(city=["Osaka", "Kyoto", "Osaka"])["updates"] == {"city": ["Osaka", "Kyoto"]}


def test_revalidation_rejects_tampered_payloads():
    assert validated_trip_updates(STATE, {"num_people": 3, "budget": 1}) == {"num_people": 3}
    assert validated_trip_updates(STATE, {"num_people": True}) == {}
    assert validated_trip_updates(STATE, {"end_date": "2026-10-01"}) == {}
    assert validated_trip_updates(STATE, "junk") == {}


# ── Trip limits (shared with the form; "today" is pinned to 2026-10-20) ──


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"num_people": 10}, "At most 9 travellers"),
        ({"end_date": "2026-11-15"}, "at most 14 days long; this one is 15 days"),
        ({"start_date": "2026-10-18", "end_date": "2026-10-20"}, "already passed"),
        ({"start_date": "2027-09-13", "end_date": "2027-09-16"}, "by 2027-09-15"),
        ({"city": ["A", "B", "C", "D", "E", "F"]}, "At most 5 cities"),
        ({"city": ["x" * 101]}, "at most 100 characters"),
        ({"country": "x" * 101}, "at most 100 characters"),
        ({"origin_country": "x" * 101}, "at most 100 characters"),
    ],
)
def test_each_limit_is_refused_just_outside_it(args, fragment):
    result = call(**args)

    assert fragment in result["error"]
    assert validated_trip_updates(STATE, args) == {}


@pytest.mark.parametrize(
    "args",
    [
        {"num_people": 9},
        {"end_date": "2026-11-14"},
        {"start_date": "2026-10-19", "end_date": "2026-10-21"},
        {"start_date": "2027-09-13", "end_date": "2027-09-15"},
        {"city": ["A", "B", "C", "D", "E"]},
        {"city": ["x" * 100]},
        {"country": "x" * 100},
    ],
)
def test_each_limit_is_accepted_just_inside_it(args):
    assert call(**args)["status"] == "accepted"
    assert validated_trip_updates(STATE, args) == args


def test_a_trip_whose_start_has_passed_cannot_be_replanned_without_new_dates():
    from datetime import date

    from app.shared.trip_limits import pinned_today

    with pinned_today(date(2026, 11, 10)):
        assert "2026-11-01 has already passed" in call(num_people=3)["error"]
        assert validated_trip_updates(STATE, {"num_people": 3}) == {}
        assert call(start_date="2026-12-01", end_date="2026-12-03")["status"] == "accepted"


def test_the_tool_description_states_the_current_limits():
    from app.shared import trip_limits

    description = update_trip_details.description

    assert f"num_people is 1 to {trip_limits.MAX_TRAVELLERS}" in description
    assert f"at most {trip_limits.MAX_TRIP_DAYS} days" in description
    assert f"about {trip_limits.MAX_DAYS_AHEAD} days" in description
    assert f"at most {trip_limits.MAX_CITIES} cities" in description
