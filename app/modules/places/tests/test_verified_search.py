"""Destination verification with fake search, geocoder and locator."""

import pytest

from app.modules.itinerary.models import MAX_ACTIVITY_CITY_DISTANCE_KM
from app.modules.places.exceptions import PlaceSearchUnavailable
from app.modules.places.models import PlaceCandidate
from app.modules.places.policies import (
    MAX_KM_FROM_CITY,
    SearchQuota,
    canonical_city,
    choose_locality,
    join_query,
)
from app.modules.places.service import VerifiedPlaceSearch
from app.shared.geo import GeocodedPlace


TOKYO = (35.6812, 139.7671)

SENSOJI = PlaceCandidate(title="Senso-ji", latitude=35.7148, longitude=139.7967, address="Asakusa")
DISNEY = PlaceCandidate(title="Tokyo Disneyland", latitude=35.6329, longitude=139.8804)
OSAKA = PlaceCandidate(title="Osaka Castle", latitude=34.6873, longitude=135.5262)
SEOUL = PlaceCandidate(title="Gyeongbokgung", latitude=37.5796, longitude=126.9770)

LOCALITIES = {
    (SENSOJI.latitude, SENSOJI.longitude): ("JP", ("Asakusa", "Tokyo")),
    (DISNEY.latitude, DISNEY.longitude): ("JP", ("Urayasu", "Chiba")),
    (OSAKA.latitude, OSAKA.longitude): ("JP", ("Osaka",)),
    (SEOUL.latitude, SEOUL.longitude): ("KR", ("Seoul",)),
}


class FakeSearch:
    def __init__(self, candidates=(), error=None):
        self.candidates = list(candidates)
        self.error = error
        self.calls = []

    def search(self, query, *, near=None, zoom=11, quota=None):
        self.calls.append((query, near, zoom))
        if self.error:
            raise self.error
        return list(self.candidates)


class FakeGeocoder:
    def __init__(self, place=TOKYO, error=None):
        self.place = place
        self.error = error

    def geocode(self, query, *, proximity=None):
        if self.error:
            raise self.error
        if self.place is None:
            return None
        return GeocodedPlace(latitude=self.place[0], longitude=self.place[1], place_name=query)


class FakeLocator:
    def __init__(self, error=None):
        self.error = error

    def resolve_country_code(self, latitude, longitude):
        if self.error:
            raise self.error
        return LOCALITIES.get((latitude, longitude), (None, ()))[0]

    def resolve_locality_names(self, latitude, longitude, destination_country_code):
        return LOCALITIES.get((latitude, longitude), (None, ()))[1]


def service(candidates=(), *, geocoder=None, locator=None, error=None):
    search = FakeSearch(candidates, error=error)
    return (
        VerifiedPlaceSearch(
            search=search,
            geocoder=geocoder or FakeGeocoder(),
            locator=locator or FakeLocator(),
        ),
        search,
    )


def tokyo_area(places):
    return places.area(country="Japan", city="Tokyo")


def test_area_resolves_country_and_anchor():
    places, _ = service()
    area = tokyo_area(places)

    assert area.country_code == "JP"
    assert area.anchor == TOKYO


def test_area_rejects_unknown_country_and_blank_city():
    places, _ = service()

    assert places.area(country="Atlantis", city="Tokyo") is None
    assert places.area(country="Japan", city="  ") is None


def test_anchor_failure_keeps_area_without_anchor():
    places, _ = service(geocoder=FakeGeocoder(error=RuntimeError("down")))

    assert tokyo_area(places).anchor is None


def test_exact_locality_is_verified_with_distance_evidence():
    places, search = service([SENSOJI])

    [place] = places.search("Senso-ji", area=tokyo_area(places))

    assert place.verified_locality == "Tokyo"
    assert 0 < place.distance_from_requested_city_km < 10
    assert search.calls[0][1] == TOKYO


def test_nearby_place_is_accepted_with_distance_proof():
    places, _ = service([DISNEY])

    [place] = places.search("Tokyo Disneyland", area=tokyo_area(places))

    assert place.verified_locality == "Urayasu"
    assert place.distance_from_requested_city_km <= MAX_KM_FROM_CITY
    assert place.location_payload()["distance_from_requested_city_km"] == (
        place.distance_from_requested_city_km
    )


def test_nearby_place_without_anchor_is_rejected():
    places, _ = service([DISNEY], geocoder=FakeGeocoder(place=None))

    assert places.search("Tokyo Disneyland", area=tokyo_area(places)) == []


def test_far_and_foreign_places_are_rejected():
    places, _ = service([OSAKA, SEOUL, SENSOJI])

    results = places.search("castle", area=tokyo_area(places), limit=3)

    assert [place.name for place in results] == ["Senso-ji"]


def test_locator_failure_rejects_candidate():
    places, _ = service([SENSOJI], locator=FakeLocator(error=RuntimeError("down")))

    assert places.search("Senso-ji", area=tokyo_area(places)) == []


def test_limit_and_blank_query():
    places, search = service([SENSOJI, SENSOJI])

    assert len(places.search("Senso-ji", area=tokyo_area(places), limit=1)) == 1
    assert places.search("  ", area=tokyo_area(places)) == []
    assert len(search.calls) == 1


def test_provider_failure_propagates():
    places, _ = service(error=PlaceSearchUnavailable("down"))

    with pytest.raises(PlaceSearchUnavailable):
        places.search("Senso-ji", area=tokyo_area(places))


# ── Pure rules ──────────────────────────────────────────────────────


def test_radius_never_exceeds_itinerary_contract():
    assert MAX_KM_FROM_CITY <= MAX_ACTIVITY_CITY_DISTANCE_KM


def test_canonical_city():
    assert canonical_city(" tokyo ", ["Tokyo", "Kyoto"]) == "Tokyo"
    assert canonical_city("Osaka", ["Tokyo", "Kyoto"]) is None
    assert canonical_city("", ["Tokyo"]) == "Tokyo"
    assert canonical_city("", ["Tokyo", "Kyoto"]) is None


def test_choose_locality():
    assert choose_locality(["Asakusa", "tokyo"], "Tokyo", nearby_allowed=False) == "tokyo"
    assert choose_locality(["Urayasu"], "Tokyo", nearby_allowed=True) == "Urayasu"
    assert choose_locality(["Urayasu"], "Tokyo", nearby_allowed=False) is None


def test_join_query_and_quota():
    assert join_query(" a ", "", None, "b") == "a, b"

    quota = SearchQuota(2)
    assert [quota.try_acquire() for _ in range(3)] == [True, True, False]
    assert quota.used == 2
