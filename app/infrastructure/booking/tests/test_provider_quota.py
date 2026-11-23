"""Flight and hotel searches never turn a used-up SerpAPI quota into an
empty list (which reads as "no flights/hotels"); other failures still do."""

import pytest

from app.infrastructure.booking.flight_provider import SerpApiFlightProvider
from app.infrastructure.booking.hotel_provider import SerpApiHotelProvider
from app.infrastructure.search.serpapi import SerpApiQuotaExhausted, SerpApiUnavailableError


class FailingClient:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def search(self, params):
        self.calls += 1
        raise self.error


class EmptyCache:
    def get(self, *, namespace, key):
        return None

    def set(self, *, namespace, key, payload, ttl_hours=1):
        return True


class Iata:
    def resolve_iata_code(self, place):
        return {"Malaysia": "KUL", "Japan": "HND"}[place]


class AnyLocation:
    def resolve_country_code(self, latitude, longitude):
        return "JP"


def flights(client):
    return SerpApiFlightProvider(client=client, iata_resolver=Iata(), cache=EmptyCache())


def hotels(client):
    return SerpApiHotelProvider(client=client, cache=EmptyCache(), location_verifier=AnyLocation())


FLIGHT = dict(origin="Malaysia", destination="Japan", departure_date="2026-11-01", currency="JPY")
HOTEL = dict(destination="Tokyo, Japan", check_in="2026-11-01", check_out="2026-11-03", currency="JPY")


def test_a_used_up_quota_is_raised_by_flight_search():
    with pytest.raises(SerpApiQuotaExhausted):
        flights(FailingClient(SerpApiQuotaExhausted("used up"))).search_flights(**FLIGHT)


def test_a_used_up_quota_is_raised_by_hotel_search():
    with pytest.raises(SerpApiQuotaExhausted):
        hotels(FailingClient(SerpApiQuotaExhausted("used up"))).search_hotels(**HOTEL)


def test_other_failures_still_mean_no_results():
    assert flights(FailingClient(SerpApiUnavailableError("down"))).search_flights(**FLIGHT) == []
    assert hotels(FailingClient(SerpApiUnavailableError("down"))).search_hotels(**HOTEL) == []
