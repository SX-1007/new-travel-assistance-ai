"""Mapbox Directions client against a fake HTTP session."""

import pytest
import requests

from app.infrastructure.maps.directions import MapboxDirectionsClient
from app.modules.routing.ports import DirectionsRoute


POINTS = [(139.70, 35.69), (139.79, 35.71)]

ROUTE = {
    "routes": [
        {
            "distance": 9876.5,
            "duration": 1234.0,
            "geometry": {
                "type": "LineString",
                "coordinates": [[139.701, 35.691], [139.75, 35.70], [139.789, 35.709]],
            },
        }
    ]
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        raise AssertionError("injected session must not be closed")


def client(*responses):
    session = FakeSession(*responses)
    return MapboxDirectionsClient(access_token="token", session=session), session


def test_empty_token_is_rejected():
    with pytest.raises(ValueError):
        MapboxDirectionsClient(access_token=" ")


def test_route_is_parsed_and_url_is_built():
    directions, session = client(FakeResponse(payload=ROUTE))

    route = directions.fetch_route(POINTS, "walking")

    assert route == DirectionsRoute(
        distance_m=9876.5,
        duration_s=1234.0,
        coordinates=((139.701, 35.691), (139.75, 35.70), (139.789, 35.709)),
    )
    url, params = session.calls[0]
    assert url.endswith("/mapbox/walking/139.7,35.69;139.79,35.71")
    assert params["geometries"] == "geojson"


@pytest.mark.parametrize(
    "coordinates, profile",
    [
        (POINTS, "flying"),
        (POINTS[:1], "driving"),
        ([(999.0, 35.0), POINTS[1]], "driving"),
        ([POINTS[0]] * 26, "driving"),
    ],
)
def test_bad_input_makes_no_request(coordinates, profile):
    directions, session = client()

    assert directions.fetch_route(coordinates, profile) is None
    assert session.calls == []


@pytest.mark.parametrize(
    "response",
    [
        requests.Timeout("slow"),
        FakeResponse(status_code=422),
        FakeResponse(payload=ValueError("bad json")),
        FakeResponse(payload={"routes": []}),
        FakeResponse(payload={"routes": [{"distance": -1, "duration": 5, "geometry": ROUTE["routes"][0]["geometry"]}]}),
        FakeResponse(payload={"routes": [{"distance": 1, "duration": 5, "geometry": {"type": "Point", "coordinates": [1, 2]}}]}),
    ],
)
def test_failures_return_none(response):
    directions, _ = client(response)

    assert directions.fetch_route(POINTS, "driving") is None


def test_close_leaves_injected_session_open():
    directions, _ = client()
    directions.close()
