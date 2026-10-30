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
    assert call(start_date="2026-11-10")["error"] == "end_date must be on or after start_date."


def test_duplicate_cities_are_collapsed():
    assert call(city=["Osaka", "Kyoto", "Osaka"])["updates"] == {"city": ["Osaka", "Kyoto"]}


def test_revalidation_rejects_tampered_payloads():
    assert validated_trip_updates(STATE, {"num_people": 3, "budget": 1}) == {"num_people": 3}
    assert validated_trip_updates(STATE, {"num_people": True}) == {}
    assert validated_trip_updates(STATE, {"end_date": "2026-10-01"}) == {}
    assert validated_trip_updates(STATE, "junk") == {}
