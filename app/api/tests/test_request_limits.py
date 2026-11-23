"""Per-user request limits and plan slots, on their own and through the
real routes (429 + Retry-After over a limit, 409 for a second plan)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.api.providers import plan_slots, request_limits
from app.api.request_limits import (
    CHAT_MESSAGES,
    FORM_SUBMITS,
    MAP_PINS,
    PLANS,
    PlanSlots,
    RequestLimits,
)
from app.shared.exceptions import (
    PlanAlreadyRunning,
    PlanningCapacityFull,
    RateLimitExceeded,
    RateLimitUnavailable,
)


@dataclass
class Count:
    count: int
    seconds_left: int


class MemoryCounter:
    def __init__(self, *, seconds_left=1800):
        self.counts = {}
        self.seconds_left = seconds_left
        self.down = False

    def hit(self, rule, subject, *, window_seconds):
        if self.down:
            raise ConnectionError("store down")
        key = (rule, subject)
        self.counts[key] = self.counts.get(key, 0) + 1
        return Count(self.counts[key], self.seconds_left)


# ── RequestLimits ───────────────────────────────────────────────────


def test_requests_up_to_the_limit_pass_and_the_next_is_refused():
    limits = RequestLimits(MemoryCounter(seconds_left=5400), fail_closed=True)

    for _ in range(CHAT_MESSAGES.limit):
        limits.check(CHAT_MESSAGES, "u")

    with pytest.raises(RateLimitExceeded) as caught:
        limits.check(CHAT_MESSAGES, "u")

    assert caught.value.retry_after_seconds == 5400
    assert caught.value.message == (
        "You've reached the limit of 30 chat messages an hour. Please try again in 1 h 30 min."
    )
    # Another user has their own allowance.
    limits.check(CHAT_MESSAGES, "someone-else")


@pytest.mark.parametrize("rule", [FORM_SUBMITS, PLANS, CHAT_MESSAGES])
def test_production_refuses_plans_and_chat_when_the_store_is_down(rule):
    counter = MemoryCounter()
    counter.down = True

    with pytest.raises(RateLimitUnavailable):
        RequestLimits(counter, fail_closed=True).check(rule, "u")

    RequestLimits(counter, fail_closed=False).check(rule, "u")  # development


def test_map_pins_go_through_when_the_store_is_down_even_in_production():
    counter = MemoryCounter()
    counter.down = True

    RequestLimits(counter, fail_closed=True).check(MAP_PINS, "u")


def test_without_a_store_limits_are_off():
    limits = RequestLimits(None, fail_closed=False)

    assert not limits.enabled
    for _ in range(100):
        limits.check(PLANS, "u")


# ── PlanSlots ───────────────────────────────────────────────────────


def test_one_running_submission_per_user_and_a_cap_on_plans():
    async def scenario():
        slots = PlanSlots(max_running=2)

        async with slots.for_user("u"):
            with pytest.raises(PlanAlreadyRunning):
                async with slots.for_user("u"):
                    pass

            async with slots.for_user("v"):
                async with slots.planning(), slots.planning():
                    with pytest.raises(PlanningCapacityFull):
                        async with slots.planning():
                            pass

        # Everything is released afterwards, also after an error.
        with pytest.raises(RuntimeError):
            async with slots.for_user("u"), slots.planning():
                raise RuntimeError("boom")

        async with slots.for_user("u"), slots.planning(), slots.planning():
            pass

    asyncio.run(scenario())


# ── Through the routes ──────────────────────────────────────────────


def _world_with(limits, slots=None):
    from app.api.tests.test_routes import World

    world = World()
    api = world.client.app
    api.dependency_overrides[request_limits] = lambda: limits
    api.dependency_overrides[plan_slots] = lambda: slots or PlanSlots(max_running=2)
    return world


def test_the_form_answers_429_with_retry_after_over_its_limits():
    counter = MemoryCounter(seconds_left=7200)
    world = _world_with(RequestLimits(counter, fail_closed=True))

    # Every submission counts as a form submission; a planned one is also a plan.
    for _ in range(PLANS.limit):
        assert world.submit().json()["status"] == "success"

    over_plans = world.submit()
    assert over_plans.status_code == 429
    assert over_plans.headers["retry-after"] == "7200"
    assert "3 new trip plans a day" in over_plans.json()["detail"]

    # A budget check alone still counts against the form limit.
    for _ in range(FORM_SUBMITS.limit - PLANS.limit - 1):
        assert world.submit(total_budget=10).json()["status"] == "budget_confirmation_required"

    over_form = world.submit(total_budget=10)
    assert over_form.status_code == 429
    assert "10 trip form submissions a day" in over_form.json()["detail"]


def test_chat_answers_429_with_retry_after_over_its_limit():
    counter = MemoryCounter(seconds_left=600)
    world = _world_with(RequestLimits(counter, fail_closed=True))
    session_id = world.submit().json()["session_id"]
    counter.counts[("chat", "user-1")] = CHAT_MESSAGES.limit

    response = world.say(session_id, "Which day is the museum?")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "600"
    assert response.json()["detail"].endswith("Please try again in 10 min.")


def test_map_pins_answer_429_over_their_limit():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.errors import register_exception_handlers
    from app.api.providers import identity_verifier, place_pin_locator
    from app.api.routers import geocode
    from app.infrastructure.auth.dev_identity import InsecureDevIdentityVerifier

    class Locator:
        def locate(self, query, *, near=None):
            return None

    api = FastAPI()
    register_exception_handlers(api)
    api.include_router(geocode.router, prefix="/api/geocode")
    api.dependency_overrides[identity_verifier] = InsecureDevIdentityVerifier
    api.dependency_overrides[place_pin_locator] = Locator
    limits = RequestLimits(MemoryCounter(seconds_left=30), fail_closed=True)
    api.dependency_overrides[request_limits] = lambda: limits
    client = TestClient(api, raise_server_exceptions=False)
    headers = {"Authorization": "Bearer user-1"}

    statuses = [client.get("/api/geocode/?q=Senso-ji", headers=headers).status_code for _ in range(MAP_PINS.limit + 1)]

    assert statuses[:-1] == [200] * MAP_PINS.limit
    assert statuses[-1] == 429


def test_a_second_plan_from_the_same_user_while_one_runs_is_refused():
    slots = PlanSlots(max_running=2)
    world = _world_with(RequestLimits(None, fail_closed=False), slots)

    async def running_elsewhere():
        async with slots.for_user("user-1"):
            return world.submit()

    response = asyncio.run(running_elsewhere())

    assert response.status_code == 409
    assert "already have a trip being planned" in response.json()["detail"]


def test_a_full_planner_answers_503_without_using_up_a_plan():
    slots = PlanSlots(max_running=1)
    counter = MemoryCounter()
    world = _world_with(RequestLimits(counter, fail_closed=True), slots)

    async def planner_busy():
        async with slots.planning():
            return world.submit()

    response = asyncio.run(planner_busy())

    assert response.status_code == 503 and response.headers["retry-after"] == "60"
    assert ("plan", "user-1") not in counter.counts
