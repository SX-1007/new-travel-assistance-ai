"""Mapbox forward geocoder against a fake HTTP session."""

import pytest
import requests

from app.infrastructure.maps.mapbox import MapboxForwardGeocoder
from app.shared.geo import GeocodedPlace


TOKYO = {
    "features": [
        {
            "geometry": {"coordinates": [139.69, 35.68]},
            "properties": {"name": "Tokyo", "place_formatted": "Japan"},
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
        self.calls.append(params)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        raise AssertionError("injected session must not be closed")


def geocoder(*responses):
    session = FakeSession(*responses)
    return MapboxForwardGeocoder(access_token="token", session=session), session


def test_empty_token_is_rejected():
    with pytest.raises(ValueError):
        MapboxForwardGeocoder(access_token="  ")


def test_match_is_parsed_and_cached():
    geo, session = geocoder(FakeResponse(payload=TOKYO))

    first = geo.geocode("  Tokyo,   Japan ")
    second = geo.geocode("tokyo, japan")

    assert first == GeocodedPlace(latitude=35.68, longitude=139.69, place_name="Tokyo, Japan")
    assert second == first
    assert len(session.calls) == 1
    assert session.calls[0]["q"] == "Tokyo, Japan"
    assert "proximity" not in session.calls[0]


def test_proximity_is_sent_and_part_of_cache_key():
    geo, session = geocoder(FakeResponse(payload=TOKYO), FakeResponse(payload=TOKYO))

    geo.geocode("Tokyo", proximity=(139.0, 35.0))
    geo.geocode("Tokyo")

    assert session.calls[0]["proximity"] == "139.0,35.0"
    assert len(session.calls) == 2


def test_empty_answer_is_cached_as_negative():
    geo, session = geocoder(FakeResponse(payload={"features": []}))

    assert geo.geocode("Nowhere") is None
    assert geo.geocode("Nowhere") is None
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("down"),
        FakeResponse(status_code=503),
        FakeResponse(payload=ValueError("bad json")),
    ],
)
def test_transient_failures_are_not_cached(failure):
    geo, session = geocoder(failure, FakeResponse(payload=TOKYO))

    assert geo.geocode("Tokyo") is None
    assert geo.geocode("Tokyo") is not None
    assert len(session.calls) == 2


def test_permanent_http_failure_is_cached():
    geo, session = geocoder(FakeResponse(status_code=422))

    assert geo.geocode("Tokyo") is None
    assert geo.geocode("Tokyo") is None
    assert len(session.calls) == 1


def test_invalid_coordinates_are_rejected():
    payload = {"features": [{"geometry": {"coordinates": [999, 35]}, "properties": {}}]}
    geo, _ = geocoder(FakeResponse(payload=payload))

    assert geo.geocode("Tokyo") is None


def test_blank_query_makes_no_request():
    geo, session = geocoder()

    assert geo.geocode("   ") is None
    assert session.calls == []


def test_close_leaves_injected_session_open():
    geo, _ = geocoder()
    geo.close()
