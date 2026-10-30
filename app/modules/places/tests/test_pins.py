import pytest

from app.modules.places.exceptions import PlaceSearchUnavailable
from app.modules.places.models import PlaceCandidate
from app.modules.places.pins import PlacePinLocator


TOWER = PlaceCandidate(title="Tokyo Tower", latitude=35.6586, longitude=139.7454, address="4-2-8 Shibakoen")


class Search:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def search(self, query, *, near=None, zoom=11, quota=None):
        self.calls.append((query, near, zoom))
        outcome = self.results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_the_top_match_is_returned_and_cached_per_rounded_anchor():
    search = Search([TOWER, PlaceCandidate(title="Other", latitude=1.0, longitude=1.0)])
    locator = PlacePinLocator(search=search)

    assert locator.locate("  Tokyo   Tower ", near=(35.6581, 139.7414)) == TOWER
    # Same place, nearby anchor, different spacing/case: one provider call.
    assert locator.locate("tokyo tower", near=(35.6649, 139.7449)) == TOWER
    assert search.calls == [("Tokyo Tower", (35.66, 139.74), 13)]


def test_misses_are_cached_but_failures_are_not():
    search = Search([], PlaceSearchUnavailable("down"), [TOWER])
    locator = PlacePinLocator(search=search)

    assert locator.locate("Nowhere") is None
    assert locator.locate("Nowhere") is None
    with pytest.raises(PlaceSearchUnavailable):
        locator.locate("Tokyo Tower")
    assert locator.locate("Tokyo Tower") == TOWER
    assert len(search.calls) == 3


def test_a_blank_query_never_reaches_the_provider():
    search = Search()

    assert PlacePinLocator(search=search).locate("   ") is None
    assert search.calls == []
