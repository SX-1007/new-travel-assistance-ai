"""SerpAPI Google Maps adapter with a fake client and in-memory cache."""

import pytest

from app.infrastructure.places.serpapi_maps import (
    SerpApiMapsPlaceSearch,
    parse_candidate,
    parse_candidates,
)
from app.infrastructure.search.serpapi import SerpApiUnavailableError
from app.modules.places.exceptions import (
    PlaceSearchQuotaExhausted,
    PlaceSearchUnavailable,
)
from app.modules.places.policies import SearchQuota


SENSOJI = {
    "title": " Senso-ji ",
    "gps_coordinates": {"latitude": 35.7148, "longitude": 139.7967},
    "address": "2-3-1 Asakusa",
    "type": ["Buddhist temple", "Tourist attraction"],
    "rating": 4.5,
    "reviews": 1000,
    "images": [{"thumbnail": "https://img/1.jpg"}],
    "phone": "+81 3-3842-0181",
}


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else {"local_results": [SENSOJI]}
        self.error = error
        self.calls = []

    def search(self, params):
        self.calls.append(dict(params))
        if self.error:
            raise self.error
        return self.payload


class MemoryCache:
    def __init__(self):
        self.entries = {}

    def get(self, *, namespace, key):
        return self.entries.get((namespace, repr(sorted(key.items()))))

    def set(self, *, namespace, key, payload, ttl_hours):
        self.entries[(namespace, repr(sorted(key.items())))] = payload
        return True


def provider(client=None, cache=None):
    client = client or FakeClient()
    cache = cache or MemoryCache()
    return SerpApiMapsPlaceSearch(client=client, cache=cache), client, cache


def test_candidate_is_normalised():
    candidate = parse_candidate(SENSOJI)

    assert candidate.title == "Senso-ji"
    assert candidate.category == "Buddhist temple, Tourist attraction"
    assert candidate.thumbnail == "https://img/1.jpg"
    assert candidate.rating == 4.5 and candidate.reviews == 1000


@pytest.mark.parametrize(
    "raw",
    [
        {"title": "x"},
        {"title": "", "gps_coordinates": {"latitude": 1, "longitude": 1}},
        {"title": "x", "gps_coordinates": {"latitude": 95, "longitude": 1}},
        {"title": "x", "gps_coordinates": {"latitude": True, "longitude": 1}},
        "not a dict",
    ],
)
def test_unusable_candidates_are_dropped(raw):
    assert parse_candidate(raw) is None


def test_out_of_range_rating_is_dropped():
    assert parse_candidate({**SENSOJI, "rating": 7}).rating is None


def test_direct_place_result_is_used_when_no_list():
    assert [c.title for c in parse_candidates({"place_results": SENSOJI})] == ["Senso-ji"]


def test_search_sends_location_bias_and_caches():
    places, client, _ = provider()
    quota = SearchQuota(5)

    first = places.search("Senso-ji, Tokyo", near=(35.68, 139.76), zoom=11, quota=quota)
    second = places.search("senso-ji,  tokyo", near=(35.68, 139.76), zoom=11, quota=quota)

    assert first == second
    assert len(client.calls) == 1
    assert client.calls[0]["ll"] == "@35.68,139.76,11z"
    assert client.calls[0]["engine"] == "google_maps"
    assert quota.used == 1


def test_exhausted_quota_raises_before_calling_provider():
    places, client, _ = provider()

    with pytest.raises(PlaceSearchQuotaExhausted):
        places.search("Senso-ji", quota=SearchQuota(0))

    assert client.calls == []


def test_cache_hit_is_free_even_when_quota_is_spent():
    places, _, _ = provider()
    places.search("Senso-ji")

    assert places.search("Senso-ji", quota=SearchQuota(0))


def test_provider_failure_is_translated():
    places, _, _ = provider(client=FakeClient(error=SerpApiUnavailableError("down")))

    with pytest.raises(PlaceSearchUnavailable):
        places.search("Senso-ji")


def test_corrupt_cache_entry_is_ignored():
    places, client, cache = provider()
    places.search("Senso-ji")
    for key in cache.entries:
        cache.entries[key] = [{"title": "x", "latitude": 999, "longitude": 0}]

    places.search("Senso-ji")

    assert len(client.calls) == 2


def test_blank_query_makes_no_call():
    places, client, _ = provider()

    assert places.search("   ") == []
    assert client.calls == []
