"""Offline hotel contract tests. No credentials, network or Firebase needed."""

import pytest

from app.infrastructure.booking.hotel_provider import (
    SerpApiHotelProvider,
    _extract_rate,
    _parse_hotel_property,
)
from app.infrastructure.search.serpapi import SerpApiUnavailableError
from app.modules.booking.hotels import (
    calculate_total_nights,
    normalise_hotel_coordinates,
    verify_hotel_location,
)


class FakeClient:
    def __init__(self, properties=None, fail=False):
        self.calls = []
        self.properties = properties if properties is not None else []
        self.fail = fail

    def search(self, params):
        self.calls.append(dict(params))
        if self.fail:
            raise SerpApiUnavailableError("offline")
        return {"properties": self.properties}


class FakeCache:
    def __init__(self):
        self.content = None
        self.gets = []
        self.sets = []

    def get(self, *, namespace, key):
        self.gets.append((namespace, dict(key)))
        return self.content

    def set(self, *, namespace, key, payload, ttl_hours=1):
        self.sets.append((namespace, dict(key), payload, ttl_hours))
        self.content = payload
        return True


class FakeVerifier:
    def __init__(self, country="MY", names=("Kuala Lumpur",), raise_error=False):
        self.country = country
        self.names = names
        self.raise_error = raise_error
        self.country_calls = []
        self.locality_calls = []

    def resolve_country_code(self, latitude, longitude):
        self.country_calls.append((latitude, longitude))
        if self.raise_error:
            raise RuntimeError("offline")
        return self.country

    def resolve_locality_names(self, latitude, longitude, destination_country_code):
        self.locality_calls.append((latitude, longitude, destination_country_code))
        if self.raise_error:
            raise RuntimeError("offline")
        return self.names


def make_hotel(name="Sample hotel", rate=198.0, lat=3.139, lng=101.6869):
    return {
        "name": name,
        "rate_per_night": {"extracted_lowest": rate, "lowest": "RM 198"},
        "gps_coordinates": {"latitude": lat, "longitude": lng},
        "images": [{"thumbnail": "https://example.com/a.jpg"}],
        "amenities": ["Wi-Fi", "Pool"],
        "overall_rating": "4.6",
        "reviews": "1,234",
        "extracted_hotel_class": "4",
        "link": "https://example.com/book",
    }


def search(provider, **changes):
    kwargs = dict(
        destination="Kuala Lumpur, Malaysia",
        check_in="2026-12-01",
        check_out="2026-12-03",
        currency="MYR",
        adults=2,
    )
    kwargs.update(changes)
    return provider.search_hotels(**kwargs)


def test_night_calculation_and_invalid_dates():
    assert calculate_total_nights("2026-12-01", "2026-12-03") == 2
    assert calculate_total_nights("2026-12-01", "2026-12-01") == 0
    for start, end in [
        ("2026-12-03", "2026-12-01"),
        ("2026-02-30", "2026-03-01"),
        ("INVALID", "2026-12-01"),
        (None, "2026-12-01"),
    ]:
        with pytest.raises(ValueError):
            calculate_total_nights(start, end)


@pytest.mark.parametrize("value, expected", [
    ({"extracted_lowest": 1234.56}, 1234.56),
    ({"lowest": "¥12,345"}, 12345.0),
    ({"lowest": "MYR 1,234.50"}, 1234.50),
    ({"lowest": "$99 - $150"}, None),
    ({"lowest": "RM 1,2,3"}, None),
    ({"extracted_lowest": "NaN", "lowest": "RM 20"}, 20.0),
    ({"lowest": "RM 0"}, None),
    ({"extracted_lowest": float("inf")}, None),
    ({"extracted_lowest": True}, None),
])
def test_extract_rate(value, expected):
    assert _extract_rate(value) == expected


def test_reject_invalid_coords():
    for loc in [
        {"lat": 91, "lng": 101},
        {"lat": 3.1, "lng": -181},
        {"lat": float("nan"), "lng": 101},
        {"lat": True, "lng": 101},
        {"lat": 3.1},
        None,
    ]:
        assert normalise_hotel_coordinates(loc) is None


def test_exact_city_and_country_fail_closed():
    location = {"lat": 3.139, "lng": 101.6869}
    for verifier in [
        FakeVerifier(country="SG"),
        FakeVerifier(names=("Petaling Jaya",)),
        FakeVerifier(names=["Kuala Lumpur"]),
        FakeVerifier(raise_error=True),
    ]:
        assert verify_hotel_location(
            location, requested_city="Kuala Lumpur",
            destination_country_code="MY", verifier=verifier,
        ) is None

    result = verify_hotel_location(
        location, requested_city=" kuala   lumpur ",
        destination_country_code="my", verifier=FakeVerifier(),
    )
    assert result is not None
    assert result.model_dump() == {
        "lat": 3.139, "lng": 101.6869, "country_code": "MY"
    }


def test_search_returns_models_sorted_filtered_and_cached():
    cache, client, verifier = FakeCache(), FakeClient([
        make_hotel("Expensive", 350), make_hotel("Cheap", 180)
    ]), FakeVerifier()
    provider = SerpApiHotelProvider(client=client, cache=cache, location_verifier=verifier)
    assert callable(provider.search_hotels)
    found = search(provider, max_price_per_night=200)
    assert [h.hotel_name for h in found] == ["Cheap"]
    assert found[0].location.country_code == "MY"
    assert len(client.calls) == 1
    assert "api_key" not in client.calls[0]
    assert cache.sets[0][0] == "hotel_v2"
    assert len(cache.content) == 2  # cache UNFILTERED, sorted results
    assert [h["hotel_name"] for h in cache.content] == ["Cheap", "Expensive"]
    assert cache.sets[0][-1] == 1
    again = search(provider, max_price_per_night=None)
    assert len(again) == 2
    assert len(client.calls) == 1
    assert len(verifier.country_calls) == 4  # every cached hotel reverified


def test_invalid_cache_location_rejected_and_fresh_refetch():
    cache, client, verifier = FakeCache(), FakeClient([make_hotel()]), FakeVerifier()
    provider = SerpApiHotelProvider(client=client, cache=cache, location_verifier=verifier)
    search(provider)
    cache.content[0]["location"] = {"lat": 91, "lng": 101, "country_code": "MY"}
    assert len(search(provider)) == 1
    assert len(client.calls) == 2


def test_forged_cached_country_must_be_reverse_verified():
    cache, client, verifier = FakeCache(), FakeClient([make_hotel()]), FakeVerifier()
    provider = SerpApiHotelProvider(client=client, cache=cache, location_verifier=verifier)
    search(provider)
    verifier.country = "SG"
    assert search(provider) == []
    assert len(client.calls) == 2


def test_no_hotel_for_same_day_and_no_external_calls():
    cache, client = FakeCache(), FakeClient([make_hotel()])
    provider = SerpApiHotelProvider(client=client, cache=cache, location_verifier=FakeVerifier())
    assert search(provider, check_out="2026-12-01") == []
    assert client.calls == [] and cache.gets == []


def test_rejects_invalid_queries_before_network():
    provider = SerpApiHotelProvider(client=FakeClient(), cache=FakeCache(), location_verifier=FakeVerifier())
    for overrides in [
        {"check_out": "2026-11-01"},
        {"adults": 0},
        {"adults": True},
        {"currency": "no"},
        {"max_price_per_night": -1},
        {"max_price_per_night": float("nan")},
        {"max_price_per_night": True},
    ]:
        with pytest.raises(ValueError):
            search(provider, **overrides)


def test_drops_invalid_provider_rows_and_bad_links():
    invalid = make_hotel(name="Invalid", lat=200)
    bad_name = make_hotel(name="")
    good = make_hotel(name="Safe")
    good["link"] = "javascript:alert(1)"
    good["images"] = [{"thumbnail": "file:///etc/passwd"}]
    client = FakeClient([invalid, bad_name, good, {"name": "missing rate"}])
    provider = SerpApiHotelProvider(client=client, cache=FakeCache(), location_verifier=FakeVerifier())
    found = search(provider)
    assert len(found) == 1
    assert found[0].booking_url.startswith("https://www.google.com/travel/hotels?q=")
    assert found[0].image == ""
    assert found[0].reviews == 1234
    assert found[0].hotel_class == 4


def test_provider_unavailable_returns_empty_list():
    provider = SerpApiHotelProvider(client=FakeClient(fail=True), cache=FakeCache(), location_verifier=FakeVerifier())
    assert search(provider) == []


def test_parse_hotel_does_not_accept_non_mapping():
    assert _parse_hotel_property("not an object", destination="Kuala Lumpur, Malaysia") is None