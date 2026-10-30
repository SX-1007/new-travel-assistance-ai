"""Pure budget-assessment rules. No providers, cache or settings."""

import math
from datetime import datetime, timedelta, timezone
from typing import get_args

import pytest

from app.modules.budget.allocation import initial_budget_allocation
from app.modules.budget.assessment import (
    CALCULATION_VERSION,
    assessment_cache_key,
    assessment_matches_trip,
    build_assessment,
    is_budget_sufficient,
    primary_destination_city,
)
from app.modules.budget.models import BudgetCategory, BudgetTrip
from app.modules.budget.policies import DEFAULT_BUDGET_RATIOS


NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
TRIP = BudgetTrip(
    origin="Malaysia",
    destination="Japan",
    destination_city="Tokyo",
    start_date="2026-11-01",
    end_date="2026-11-05",
    num_people=2,
)


def make_assessment(trip=TRIP, *, hotel_price=100.0, now=NOW):
    return build_assessment(
        trip,
        base_currency="MYR",
        destination_currency="JPY",
        exchange_rate=32.0,
        outbound_flight={"price": 40000.0, "airline": "X"},
        return_flight={"price": 35000.0, "airline": "Y"},
        hotel={"price_per_night": hotel_price, "hotel_name": "H"},
        now=now,
    )


def test_ratios_cover_every_category_and_sum_to_one():
    assert set(DEFAULT_BUDGET_RATIOS) == set(get_args(BudgetCategory))
    assert math.isclose(sum(DEFAULT_BUDGET_RATIOS.values()), 1.0)


def test_initial_allocation_always_adds_up():
    for total in (1000, 333.33, 1234.57, 0.01):
        assert round(sum(initial_budget_allocation(total).values()), 2) == round(total, 2)


def test_minimum_uses_the_larger_of_flight_and_hotel_floor():
    flight_bound = make_assessment(hotel_price=100.0)
    # (40000 + 35000) / 0.25 = 300000 > (100 * 4) / 0.35
    assert flight_bound.minimum_destination_budget == pytest.approx(300000.0)
    assert flight_bound.recommended_minimum_budget == 9375.0  # 300000 / 32

    hotel_bound = make_assessment(hotel_price=50000.0)
    # (50000 * 4 nights) / 0.35 ≈ 571428.57 > 300000
    assert hotel_bound.minimum_destination_budget == pytest.approx(200000 / 0.35)


def test_recommendation_rounds_up_to_the_cent():
    assessment = build_assessment(
        TRIP,
        base_currency="MYR",
        destination_currency="JPY",
        exchange_rate=3.0,
        outbound_flight={"price": 1.0},
        return_flight={"price": 1.0},
        hotel={"price_per_night": 0.01},
        now=NOW,
    )
    # 8 / 3 = 2.6666… → 2.67, never 2.66 (which would be insufficient)
    assert assessment.recommended_minimum_budget == 2.67
    assert is_budget_sufficient(2.67, assessment)
    assert not is_budget_sufficient(2.66, assessment)


def test_same_day_trip_needs_no_hotel():
    same_day = TRIP.model_copy(update={"end_date": TRIP.start_date})
    assessment = make_assessment(same_day, hotel_price=0)
    assert assessment.evidence.hotel == {}
    assert assessment.evidence.hotel_nights == 0


def test_missing_prices_are_rejected():
    with pytest.raises(ValueError):
        build_assessment(
            TRIP,
            base_currency="MYR",
            destination_currency="JPY",
            exchange_rate=32.0,
            outbound_flight={},
            return_flight={"price": 1.0},
            hotel={"price_per_night": 1.0},
            now=NOW,
        )


def test_matches_only_the_exact_unexpired_untampered_trip():
    assessment = make_assessment()
    assert assessment_matches_trip(assessment, TRIP, now=NOW)

    # Different traveller count
    assert not assessment_matches_trip(
        assessment, TRIP.model_copy(update={"num_people": 3}), now=NOW
    )
    # Expired
    assert not assessment_matches_trip(assessment, TRIP, now=NOW + timedelta(hours=2))
    # Tampered price no longer matches the stored minimum
    tampered = assessment.model_dump()
    tampered["evidence"]["outbound_flight"]["price"] = 1.0
    assert not assessment_matches_trip(tampered, TRIP, now=NOW)
    # Old calculation version
    stale = assessment.model_copy(update={"calculation_version": "old"})
    assert not assessment_matches_trip(stale, TRIP, now=NOW)


def test_cache_key_matches_the_old_application_format():
    assert assessment_cache_key(
        TRIP, base_currency="myr", destination_currency="jpy"
    ) == {
        "origin": "malaysia",
        "destination": "japan",
        "destination_city": "tokyo",
        "start_date": "2026-11-01",
        "end_date": "2026-11-05",
        "num_people": 2,
        "base_currency": "MYR",
        "destination_currency": "JPY",
        "calculation_version": CALCULATION_VERSION,
    }


def test_primary_destination_city():
    assert primary_destination_city([" Tokyo ", "Osaka"]) == "Tokyo"
    assert primary_destination_city([]) == ""
    assert primary_destination_city(None) == ""


@pytest.mark.parametrize("bad", [None, True, -1, 0, float("nan"), "abc"])
def test_invalid_entered_budget_is_never_sufficient(bad):
    assert not is_budget_sufficient(bad, make_assessment())
