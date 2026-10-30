"""Destination resolution with fake geocoder, country locator and city picker."""

import pytest

from app.modules.destination.exceptions import (
    DestinationResolutionInvalid,
    DestinationResolutionUnavailable,
)
from app.modules.destination.service import DestinationResolutionService
from app.shared.geo import GeocodedPlace


TRIP = dict(start_date="2026-11-01", end_date="2026-11-05", num_people=2)

# city → (lat, lng); the fake locator maps coordinates back to a country.
PLACES = {
    "tokyo, japan": (35.68, 139.69),
    "osaka, japan": (34.69, 135.50),
    "seoul, japan": (37.56, 126.97),
    "singapore, singapore": (1.35, 103.82),
}
COUNTRIES = {
    (35.68, 139.69): "JP",
    (34.69, 135.50): "JP",
    (37.56, 126.97): "KR",
    (1.35, 103.82): "SG",
}


class FakeGeocoder:
    def __init__(self, places=PLACES):
        self.places = places
        self.queries = []

    def geocode(self, query, *, proximity=None):
        self.queries.append(query)
        coords = self.places.get(query.casefold())
        if coords is None:
            return None
        return GeocodedPlace(latitude=coords[0], longitude=coords[1], place_name=query)


class FakeLocator:
    def __init__(self, countries=COUNTRIES):
        self.countries = countries

    def resolve_country_code(self, latitude, longitude):
        return self.countries.get((latitude, longitude))


class FakePicker:
    def __init__(self, proposals=(), error=None):
        self.proposals = list(proposals)
        self.error = error
        self.calls = []

    def propose_primary_city(self, *, country, start_date, end_date, num_people, rejected):
        self.calls.append(tuple(rejected))
        if self.error is not None:
            raise self.error
        return self.proposals.pop(0)


def service(*, geocoder=None, locator=None, picker=None):
    return DestinationResolutionService(
        geocoder=geocoder or FakeGeocoder(),
        country_locator=locator or FakeLocator(),
        city_picker=picker or FakePicker(),
    )


def test_user_cities_are_verified_cleaned_and_deduplicated():
    geocoder = FakeGeocoder()
    result = service(geocoder=geocoder).resolve(
        country=" Japan ",
        requested_cities=["  Tokyo ", "Osaka", "tokyo"],
        **TRIP,
    )

    assert result.cities == ("Tokyo", "Osaka")
    assert result.primary_city == "Tokyo"
    assert result.country_code == "JP"
    assert result.source == "user"
    assert geocoder.queries[0] == "Tokyo, Japan"


def test_user_city_in_another_country_is_rejected():
    with pytest.raises(DestinationResolutionInvalid, match="Seoul"):
        service().resolve(country="Japan", requested_cities=["Seoul"], **TRIP)


def test_blank_user_city_is_rejected():
    with pytest.raises(DestinationResolutionInvalid, match="non-empty"):
        service().resolve(country="Japan", requested_cities=["   "], **TRIP)


def test_unknown_country_is_rejected():
    with pytest.raises(DestinationResolutionInvalid, match="country"):
        service().resolve(country="Atlantis", requested_cities=[], **TRIP)


def test_geocoder_without_answer_is_unavailable():
    with pytest.raises(DestinationResolutionUnavailable, match="locality"):
        service(geocoder=FakeGeocoder(places={})).resolve(
            country="Japan", requested_cities=["Tokyo"], **TRIP
        )


def test_country_locator_without_answer_is_unavailable():
    with pytest.raises(DestinationResolutionUnavailable, match="country"):
        service(locator=FakeLocator(countries={})).resolve(
            country="Japan", requested_cities=["Tokyo"], **TRIP
        )


def test_city_state_skips_ai_and_geocoding():
    geocoder = FakeGeocoder()
    picker = FakePicker(error=AssertionError("must not be called"))

    result = service(geocoder=geocoder, picker=picker).resolve(
        country="Singapore", requested_cities=[], **TRIP
    )

    assert result.cities == ("Singapore",)
    assert result.source == "city_state"
    assert geocoder.queries == []
    assert picker.calls == []


def test_country_only_uses_verified_ai_city():
    picker = FakePicker(proposals=[" Tokyo "])

    result = service(picker=picker).resolve(
        country="Japan", requested_cities=[], **TRIP
    )

    assert result.cities == ("Tokyo",)
    assert result.source == "ai"
    assert picker.calls == [()]


def test_wrong_country_and_empty_ai_candidates_are_retried():
    picker = FakePicker(proposals=["Seoul", "", "Osaka"])

    result = service(picker=picker).resolve(
        country="Japan", requested_cities=[], **TRIP
    )

    assert result.cities == ("Osaka",)
    assert picker.calls == [(), ("Seoul",), ("Seoul", "empty candidate")]


def test_ai_gives_up_after_three_rejected_candidates():
    picker = FakePicker(proposals=["Seoul", "Seoul", "Seoul", "Tokyo"])

    with pytest.raises(DestinationResolutionUnavailable, match="could not be selected"):
        service(picker=picker).resolve(country="Japan", requested_cities=[], **TRIP)

    assert len(picker.calls) == 3


def test_ai_provider_failure_is_unavailable():
    picker = FakePicker(error=RuntimeError("quota"))

    with pytest.raises(DestinationResolutionUnavailable, match="AI destination"):
        service(picker=picker).resolve(country="Japan", requested_cities=[], **TRIP)
