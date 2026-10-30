"""Nearby amenities with fake search, verification edges and routes."""

from app.modules.amenities.policies import MAX_RESULTS, amenity_queries
from app.modules.amenities.service import NearbyAmenityService
from app.modules.places.exceptions import PlaceSearchUnavailable
from app.modules.places.models import PlaceCandidate
from app.modules.places.service import VerifiedPlaceSearch
from app.shared.geo import GeocodedPlace


HOTEL = (35.6895, 139.6917)
ITINERARY = [
    {
        "day": 1,
        "hotel": {"hotel_name": "Park Hotel", "location": {"lat": HOTEL[0], "lng": HOTEL[1]}},
    }
]


def near(offset, title, **kwargs):
    return PlaceCandidate(title=title, latitude=HOTEL[0] + offset, longitude=HOTEL[1], **kwargs)


class FakeSearch:
    def __init__(self, by_zoom=None, error=None):
        self.by_zoom = by_zoom or {}
        self.error = error
        self.calls = []

    def search(self, query, *, near=None, zoom=11, quota=None):
        self.calls.append((query, near, zoom))
        if self.error:
            raise self.error
        return list(self.by_zoom.get(zoom, []))


class FakeGeocoder:
    def geocode(self, query, *, proximity=None):
        return GeocodedPlace(latitude=HOTEL[0], longitude=HOTEL[1], place_name=query)


class FakeLocator:
    def resolve_country_code(self, latitude, longitude):
        return "JP"

    def resolve_locality_names(self, latitude, longitude, destination_country_code):
        return ("Tokyo",)


class FakeRoutes:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def route_between(self, coordinates):
        self.calls.append(list(coordinates))
        if self.error:
            raise self.error
        return {"walking": {"distance_km": 0.4, "duration_mins": 5.0}}


def service(search=None, origin_search=None, routes=None):
    search = search or FakeSearch()
    places = VerifiedPlaceSearch(
        search=origin_search or search,
        geocoder=FakeGeocoder(),
        locator=FakeLocator(),
    )
    routes = routes or FakeRoutes()
    return NearbyAmenityService(search=search, places=places, routes=routes), search, routes


def find(svc, location="Park Hotel", category="pharmacy"):
    return svc.find_nearest(
        category=category,
        location_name=location,
        itinerary=ITINERARY,
        country="Japan",
        cities=["Tokyo"],
    )


def test_hotel_origin_ranks_by_distance_and_routes_nearest():
    search = FakeSearch({15: [near(0.01, "Far Pharmacy"), near(0.002, "Close Pharmacy", phone="123")]})
    svc, search, routes = service(search)

    result = find(svc)

    assert result["origin"] == {"name": "Park Hotel", "latitude": HOTEL[0], "longitude": HOTEL[1]}
    assert [r["name"] for r in result["results"]] == ["Close Pharmacy", "Far Pharmacy"]
    assert result["nearest"]["phone"] == "123"
    assert result["nearest"]["route_from_origin"]["profiles"]["walking"]["duration_mins"] == 5.0
    assert routes.calls[0][0] == (HOTEL[1], HOTEL[0])
    # Close search found something, so no widened search.
    assert {zoom for _, _, zoom in search.calls} == {15}


def test_widens_search_when_nothing_is_close():
    svc, search, _ = service(FakeSearch({12: [near(0.05, "Pharmacy")]}))

    result = find(svc)

    assert result["nearest"]["name"] == "Pharmacy"
    # Japan adds the local-language query, so two queries per zoom level.
    assert [(query, zoom) for query, _, zoom in search.calls] == [
        ("pharmacy", 15),
        ("薬局", 15),
        ("pharmacy", 12),
        ("薬局", 12),
    ]


def test_far_results_are_ignored_and_results_capped():
    many = [near(0.001 * i, f"P{i}") for i in range(1, 10)] + [near(2.0, "Way too far")]
    svc, _, _ = service(FakeSearch({15: many}))

    result = find(svc)

    assert len(result["results"]) == MAX_RESULTS
    assert "Way too far" not in [r["name"] for r in result["results"]]


def test_non_hotel_origin_must_be_verified():
    origin_search = FakeSearch({11: [near(0.001, "Shibuya Station")]})
    svc, _, _ = service(FakeSearch({15: [near(0.002, "Pharmacy")]}), origin_search=origin_search)

    result = find(svc, location="Shibuya Station")

    assert result["origin"]["name"] == "Shibuya Station"
    assert origin_search.calls[0][0] == "Shibuya Station, Tokyo, Japan"


def test_unverifiable_origin_is_an_error():
    foreign = PlaceCandidate(title="Elsewhere", latitude=48.85, longitude=2.35)
    svc, _, _ = service(origin_search=FakeSearch({11: [foreign]}))

    assert "error" in find(svc, location="Eiffel Tower")


def test_missing_location_and_category():
    svc, _, _ = service()

    assert "error" in find(svc, location="  ")
    assert find(svc, category="  ")["error"] == "No amenity category was provided."


def test_provider_and_route_failures_are_contained():
    svc, _, _ = service(FakeSearch(error=PlaceSearchUnavailable("down")))
    assert find(svc)["results"] == []

    svc, _, _ = service(FakeSearch({15: [near(0.002, "Pharmacy")]}), routes=FakeRoutes(error=RuntimeError()))
    assert find(svc)["nearest"]["route_from_origin"]["profiles"] == {}


def test_amenity_queries():
    assert amenity_queries("police", "JP") == ("police station", "交番", "警察署")
    assert amenity_queries("Police box", "FR") == ("police station",)
    assert amenity_queries("fireworks show", "JP") == ("fireworks show",)
    assert amenity_queries("fire", "JP") == ("fire station", "消防署")
    assert amenity_queries("  ", "JP") == ()
