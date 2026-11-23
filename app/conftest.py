"""Test-wide fixtures.

* "Today" is pinned. The trip rules refuse start dates in the past
  (``shared/trip_limits``), and the tests use fixed dates in late 2026, so
  they would start failing once the calendar passes them. A test that needs
  another day pins its own with ``pinned_today``.
* Request limits are off and plan slots are fresh in every test. Without
  this, a route test reaching the real container would read ``.env`` and
  count its requests in the real Upstash database. Tests of the limits
  pass their own through ``dependency_overrides``.
"""

from datetime import date

import pytest

from app.api.request_limits import PlanSlots, RequestLimits
from app.bootstrap import container
from app.shared.trip_limits import pinned_today


TEST_TODAY = date(2026, 10, 20)


@pytest.fixture(autouse=True)
def _pinned_trip_today():
    with pinned_today(TEST_TODAY):
        yield


@pytest.fixture(autouse=True)
def _no_real_request_limits(monkeypatch):
    monkeypatch.setattr(container, "get_request_limits", lambda: RequestLimits(None, fail_closed=False))
    slots = PlanSlots(max_running=container.MAX_PLANS_RUNNING)
    monkeypatch.setattr(container, "get_plan_slots", lambda: slots)
