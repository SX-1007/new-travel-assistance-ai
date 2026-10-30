"""Shared booking validation rules. Pure: no network, cache or settings."""

import pytest

from app.modules.booking.flights import filter_flights_by_budget
from app.modules.booking.hotels import filter_hotels_by_budget
from app.modules.booking.models import FlightResult, HotelResult
from app.modules.booking.validation import (
    normalise_adults,
    normalise_currency_code,
    parse_budget_cap,
)
from app.shared.countries import country_alpha2


FLIGHTS = [FlightResult(price=100), FlightResult(price=300)]
HOTELS = [HotelResult(price_per_night=50), HotelResult(price_per_night=200)]


@pytest.mark.parametrize("cap", [None, 0, 0.0])
def test_none_or_zero_cap_means_no_cap_for_flights_and_hotels(cap):
    assert filter_flights_by_budget(FLIGHTS, max_budget=cap) == FLIGHTS
    assert filter_hotels_by_budget(HOTELS, max_price_per_night=cap) == HOTELS


def test_positive_cap_filters():
    assert filter_flights_by_budget(FLIGHTS, max_budget=150) == FLIGHTS[:1]
    assert filter_hotels_by_budget(HOTELS, max_price_per_night=100) == HOTELS[:1]


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "abc"])
def test_invalid_caps_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_budget_cap(bad, field_name="cap")


def test_currency_and_adults_normalisers():
    assert normalise_currency_code(" myr ") == "MYR"
    assert normalise_adults(2.0) == 2

    for bad in ("RM", "MYRR", None, 3):
        with pytest.raises(ValueError):
            normalise_currency_code(bad)

    for bad in (0, -1, 1.5, True, "x"):
        with pytest.raises(ValueError):
            normalise_adults(bad)


@pytest.mark.parametrize(
    ("name", "code"),
    [("UK", "GB"), ("Macau", "MO"), ("Malaysia", "MY"), (" United  Kingdom ", "GB"), ("JP", "JP")],
)
def test_country_alpha2(name, code):
    assert country_alpha2(name) == code


def test_unknown_country_is_none():
    assert country_alpha2("Narnia") is None
    assert country_alpha2("") is None
    assert country_alpha2(None) is None
