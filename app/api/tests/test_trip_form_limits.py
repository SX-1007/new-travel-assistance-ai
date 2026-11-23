"""The trip form refuses trips the planner can't plan: each limit is tested
just inside and just outside (``shared/trip_limits``). "Today" is pinned to
2026-10-20 by ``app/conftest.py``."""

from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.errors import register_exception_handlers
from app.api.schemas.trip import InitialFormRequest
from app.shared.trip_limits import pinned_today


VALID = {
    "country": "Japan",
    "city": ["Tokyo"],
    "num_people": 2,
    "total_budget": 5000,
    "start_date": "2026-11-01",
    "end_date": "2026-11-03",
}


def form(**changes):
    return InitialFormRequest.model_validate({**VALID, **changes})


def refused(**changes) -> str:
    with pytest.raises(ValidationError) as caught:
        form(**changes)

    return str(caught.value)


def test_travellers_one_to_nine():
    assert form(num_people=1).num_people == 1
    assert form(num_people=9).num_people == 9
    assert "greater than or equal to 1" in refused(num_people=0)
    assert "At most 9 travellers" in refused(num_people=10)


def test_trip_length_at_most_fourteen_days():
    assert form(start_date="2026-11-01", end_date="2026-11-14").end_date == "2026-11-14"
    assert "at most 14 days long; this one is 15 days" in refused(start_date="2026-11-01", end_date="2026-11-15")
    assert "on or after the start date" in refused(start_date="2026-11-03", end_date="2026-11-02")


def test_start_date_allows_one_day_of_slack_for_time_zones():
    assert form(start_date="2026-10-19", end_date="2026-10-21").start_date == "2026-10-19"
    assert "already passed" in refused(start_date="2026-10-18", end_date="2026-10-21")


def test_trip_must_end_within_330_days():
    # Pinned today is 2026-10-20; 330 days later is 2027-09-15.
    assert form(start_date="2027-09-13", end_date="2027-09-15").end_date == "2027-09-15"
    assert "by 2027-09-15" in refused(start_date="2027-09-13", end_date="2027-09-16")


def test_today_comes_from_the_replaceable_clock():
    with pinned_today(date(2027, 1, 1)):
        assert "already passed" in refused()


def test_at_most_five_cities_of_at_most_100_characters():
    assert len(form(city=["A", "B", "C", "D", "E"]).city) == 5
    assert "At most 5 cities" in refused(city=["A", "B", "C", "D", "E", "F"])
    assert form(city=["x" * 100]).city == ["x" * 100]
    assert "at most 100 characters" in refused(city=["x" * 101])


def test_country_at_most_100_characters():
    assert form(country="x" * 100).country == "x" * 100
    assert "at most 100 characters" in refused(country="x" * 101)


def test_the_route_answers_422_with_a_readable_detail():
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/check")
    def check(body: InitialFormRequest) -> dict:
        return {}

    response = TestClient(app).post("/check", json={**VALID, "num_people": 12})

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "num_people: At most 9 travellers can be planned at once (the flight search limit)."
    )
