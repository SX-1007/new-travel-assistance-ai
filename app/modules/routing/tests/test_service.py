"""Daily routing with fake Directions and geocoder providers."""

import copy
from types import SimpleNamespace

import pytest

from app.modules.itinerary.validators import _validate_map_day
from app.modules.routing.models import GeoJSONFeatureCollection, RouteInfo
from app.modules.routing.policies import (
    bind_route_endpoints,
    haversine_km,
    profiles_to_request,
)
from app.modules.routing.ports import DirectionsRoute
from app.modules.routing.service import DailyRouteService
from app.orchestration.workflows.daily_map import run_daily_map
from app.shared.geo import GeocodedPlace


HOTEL = (139.70, 35.69)
TEMPLE = (139.79, 35.71)
TOWER = (139.75, 35.66)
AIRPORT = (139.78, 35.55)


def day(number, *, flight=True, hotel=True, activities=None):
    plan = {"day": number, "date": f"2026-11-0{number}", "activities": []}
    if flight:
        plan["flight"] = [{"arrival_airport": {"id": "HND", "name": "Haneda Airport"}}]
    if hotel:
        plan["hotel"] = {
            "hotel_name": " Park Hotel ",
            "location": {"lat": HOTEL[1], "lng": HOTEL[0], "country_code": "JP"},
        }
    plan["activities"] = activities if activities is not None else [
        {
            "name": "Senso-ji",
            "type": "attraction",
            "order": 1,
            "location": {"latitude": TEMPLE[1], "longitude": TEMPLE[0]},
        },
        {
            "name": "Tokyo Tower",
            "type": "attraction",
            "order": 2,
            "location": {"latitude": TOWER[1], "longitude": TOWER[0]},
        },
    ]
    return plan


class FakeDirections:
    def __init__(self, *, fail=(), missing=()):
        self.fail = set(fail)
        self.missing = set(missing)
        self.calls = []

    def fetch_route(self, coordinates, profile):
        self.calls.append((tuple(coordinates), profile))
        if profile in self.fail:
            raise RuntimeError("provider down")
        if profile in self.missing:
            return None
        # Endpoints deliberately snapped away from the waypoints.
        snapped = [(c[0] + 0.001, c[1] + 0.001) for c in coordinates]
        return DirectionsRoute(
            distance_m=12345.0,
            duration_s=1800.0,
            coordinates=tuple(snapped),
        )


class FakeGeocoder:
    def __init__(self, error=None):
        self.error = error
        self.queries = []

    def geocode(self, query, *, proximity=None):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return GeocodedPlace(latitude=AIRPORT[1], longitude=AIRPORT[0], place_name=query)


def service(directions=None, geocoder=None):
    return DailyRouteService(
        directions=directions or FakeDirections(),
        geocoder=geocoder or FakeGeocoder(),
    )


def expected_points(plan):
    hotel = plan["hotel"]
    points = [("Park Hotel", HOTEL[0], HOTEL[1], "hotel", 0)]
    for activity in plan["activities"]:
        loc = activity["location"]
        points.append(
            (activity["name"], loc["longitude"], loc["latitude"], activity["type"], activity["order"])
        )
    assert hotel
    return points


def test_day_one_map_matches_public_contract_and_validator():
    plan = day(1)
    result = service().build_daily_maps([plan])

    routed = result.itinerary[0]
    geojson = result.daily_maps[1]

    GeoJSONFeatureCollection.model_validate(geojson)
    RouteInfo.model_validate(routed["route"])

    issues = []
    _validate_map_day(
        issues,
        result.daily_maps,
        1,
        expected_points(plan),
        routed["route"],
        expected_airport_name="Haneda Airport",
    )
    assert issues == []

    assert routed["route"]["ordered_stops"] == [
        "Haneda Airport",
        "Park Hotel",
        "Senso-ji",
        "Tokyo Tower",
    ]
    assert routed["route"]["profiles"]["driving"] == {"distance_km": 12.35, "duration_mins": 30.0}


def test_route_lines_start_and_end_at_waypoints():
    result = service().build_daily_maps([day(1)])
    geojson = result.daily_maps[1]
    lines = [f for f in geojson["features"] if f["geometry"]["type"] == "LineString"]

    # Airport → hotel → activities is over 30 km, so walking is capped out.
    assert {line["properties"]["profile"] for line in lines} == {"driving", "cycling"}
    assert set(result.itinerary[0]["route"]["profiles"]) == {"driving", "cycling"}
    for line in lines:
        assert line["geometry"]["coordinates"][0] == list(AIRPORT)
        assert line["geometry"]["coordinates"][-1] == list(TOWER)


def test_later_days_have_no_airport():
    geocoder = FakeGeocoder()
    result = service(geocoder=geocoder).build_daily_maps([day(1), day(2)])

    assert result.itinerary[1]["route"]["ordered_stops"][0] == "Park Hotel"
    assert geocoder.queries == ["Haneda Airport"]


def test_airport_geocode_failure_is_not_fatal():
    result = service(geocoder=FakeGeocoder(error=RuntimeError("down"))).build_daily_maps([day(1)])

    assert result.itinerary[0]["route"]["ordered_stops"][0] == "Park Hotel"


def test_one_failing_profile_keeps_the_others():
    directions = FakeDirections(fail={"walking"}, missing={"cycling"})
    result = service(directions=directions).build_daily_maps([day(2)])

    assert set(result.itinerary[0]["route"]["profiles"]) == {"driving"}
    lines = [f for f in result.daily_maps[2]["features"] if f["properties"]["type"] == "route"]
    assert [line["properties"]["profile"] for line in lines] == ["driving"]


def test_long_day_skips_walking_and_cycling():
    far = {
        "name": "Osaka Castle",
        "type": "attraction",
        "order": 1,
        "location": {"latitude": 34.687, "longitude": 135.526},
    }
    directions = FakeDirections()
    service(directions=directions).build_daily_maps([day(2, activities=[far])])

    assert [profile for _, profile in directions.calls] == ["driving"]


def test_no_routes_gives_points_only_and_no_route_summary():
    directions = FakeDirections(missing={"driving", "walking", "cycling"})
    result = service(directions=directions).build_daily_maps([day(2)])

    assert result.itinerary[0]["route"] is None
    assert all(f["geometry"]["type"] == "Point" for f in result.daily_maps[2]["features"])


def test_single_stop_day_makes_no_route_request():
    directions = FakeDirections()
    result = service(directions=directions).build_daily_maps(
        [day(3, flight=False, activities=[])]
    )

    assert directions.calls == []
    assert result.itinerary[0]["route"] is None
    assert len(result.daily_maps[3]["features"]) == 1


def test_broken_day_falls_back_without_affecting_other_days():
    broken = day(2, activities=5)
    broken["route"] = {"ordered_stops": ["stale"], "profiles": {}}

    result = service().build_daily_maps([day(1), broken, day(3, flight=False)])

    assert result.itinerary[1]["route"] is None
    assert result.daily_maps[2] == {"type": "FeatureCollection", "features": []}
    assert result.itinerary[0]["route"] is not None
    assert result.itinerary[2]["route"] is not None
    assert [d["day"] for d in result.itinerary] == [1, 2, 3]


def test_input_is_not_mutated():
    itinerary = [day(1), day(2)]
    snapshot = copy.deepcopy(itinerary)

    service().build_daily_maps(itinerary)

    assert itinerary == snapshot


def test_empty_itinerary():
    result = service().build_daily_maps([])
    assert result.itinerary == [] and result.daily_maps == {}


def test_invalid_worker_count_is_rejected():
    with pytest.raises(ValueError):
        DailyRouteService(directions=FakeDirections(), geocoder=FakeGeocoder(), max_workers=0)


def test_workflow_returns_only_state_updates():
    state = SimpleNamespace(draft_itinerary=[day(1)])
    update = run_daily_map(state, route_service=service())

    assert set(update) == {"draft_itinerary", "daily_map_info"}
    assert 1 in update["daily_map_info"]


# ── Pure rules ──────────────────────────────────────────────────────


def test_haversine_known_distance():
    # Tokyo Station → Osaka Station is roughly 400 km in a straight line.
    assert 390 < haversine_km((139.767, 35.681), (135.495, 34.702)) < 410


def test_profiles_to_request_respects_caps():
    near = [(139.70, 35.69), (139.79, 35.71)]
    far = [(139.70, 35.69), (135.50, 34.70)]

    assert profiles_to_request(near) == ("driving", "walking", "cycling")
    assert profiles_to_request(far) == ("driving",)
    assert profiles_to_request(near[:1]) == ()


def test_bind_route_endpoints():
    line = bind_route_endpoints([(1.1, 1.1), (2.0, 2.0), (3.1, 3.1)], [(1.0, 1.0), (3.0, 3.0)])
    assert line == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]


def test_days_filter_routes_only_those_days():
    directions = FakeDirections()
    itinerary = [day(1), day(2, flight=False), day(3, flight=False)]

    result = service(directions=directions).build_daily_maps(itinerary, days={2})

    assert sorted(result.daily_maps) == [2]
    assert result.itinerary[1]["route"] is not None
    assert "route" not in result.itinerary[0] and "route" not in result.itinerary[2]
    assert result.itinerary[0] == itinerary[0] and result.itinerary[0] is not itinerary[0]
    assert all(call[0][0] == HOTEL for call in directions.calls)


def test_empty_days_filter_makes_no_requests():
    directions = FakeDirections()
    result = service(directions=directions).build_daily_maps([day(1)], days=set())

    assert result.daily_maps == {} and directions.calls == []
