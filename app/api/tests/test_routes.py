"""The chat and form routes over the real compiled graph (in-memory
checkpointer, fake network and models), the real intake and presenters,
and the error mapping, through FastAPI's test client."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.api_core import exceptions as google_errors

from app.api.errors import ServiceNotReady, error_response_for, register_exception_handlers
from app.api.providers import chat_history_loader, graph_runtime, identity_verifier, new_trip_intake, output_review
from app.api.routers import chat, form
from app.infrastructure.auth.dev_identity import InsecureDevIdentityVerifier
from app.modules.budget.models import BudgetAssessment, BudgetAssessmentResult
from app.modules.destination.exceptions import DestinationResolutionUnavailable
from app.modules.destination.models import ResolvedDestination
from app.modules.output_review.service import OutputReviewService
from app.modules.profile.models import onboarding_profile
from app.orchestration.nodes.replies import CHANGE_NOT_SAVED
from app.orchestration.tests.node_fakes import assessment, tool_call
from app.orchestration.tests.test_graph import App, edit_call
from app.orchestration.trip_intake import NewTripIntake
from app.shared.exceptions import InvalidTurnRequest, SessionBusy


# The development verifier takes the bearer token as the user id; the
# Firebase check itself is tested in app/infrastructure/auth/tests.
USER = {"Authorization": "Bearer user-1"}
FORM = {
    "country": "Japan",
    "city": ["Tokyo"],
    "num_people": 1,
    "total_budget": 5000,
    "start_date": "2026-11-01",
    "end_date": "2026-11-03",
}
CHAT_SUCCESS_KEYS = {
    "status", "chat_reply", "draft_itinerary", "daily_map_info", "destination_country_code",
    "itinerary_modified", "total_budget", "currency", "budget_allocation", "budget_confirmation",
}
PLAN_FREE_KEYS = CHAT_SUCCESS_KEYS - {"destination_country_code"}


class Profiles:
    def __init__(self, profile=None):
        self.profile = {"name": "Ana", "home_country": "Malaysia", "home_state": "Selangor"} if profile is None else profile

    def get(self, user_id):
        return onboarding_profile(self.profile)


class Destinations:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def resolve(self, *, country, requested_cities, **trip):
        self.calls += 1
        if self.fail:
            raise DestinationResolutionUnavailable("no city")
        return ResolvedDestination(country=country, country_code="JP", cities=("Tokyo",), source="user")


class Budget:
    def __init__(self, *, persisted=True):
        self.saved = BudgetAssessment.model_validate(assessment())
        self.persisted = persisted

    def get_or_create(self, trip):
        return BudgetAssessmentResult(assessment=self.saved, persisted=self.persisted)

    def load_confirmed(self, trip, *, assessment_id):
        return self.saved if assessment_id == self.saved.assessment_id else None


class World:
    """One API over one in-memory graph."""

    def __init__(self, *, app=None, runtime=None, profiles=None, destinations=None, budget=None):
        self.app = app or App(writer=None)
        self.destinations = destinations or Destinations()
        self.budget = budget or Budget()
        self.reviewer = OutputReviewService(reviewer=object())
        self.intake = NewTripIntake(
            profiles=profiles or Profiles(),
            destinations=self.destinations,
            budget=self.budget,
        )
        api = FastAPI()
        register_exception_handlers(api)
        api.include_router(chat.router, prefix="/api/chat")
        api.include_router(form.router, prefix="/api/form")
        api.dependency_overrides[graph_runtime] = lambda: runtime or self.app.runtime
        api.dependency_overrides[new_trip_intake] = lambda: self.intake
        api.dependency_overrides[output_review] = lambda: self.reviewer
        api.dependency_overrides[chat_history_loader] = lambda: self.load_history
        api.dependency_overrides[identity_verifier] = InsecureDevIdentityVerifier
        self.client = TestClient(api, raise_server_exceptions=False)

    def load_history(self, user_id, limit):
        records = []
        for thread_id, (owner, _) in self.app.store.rows.items():
            saved = self.app.saver.get_tuple({"configurable": {"thread_id": thread_id}})
            if owner == user_id and saved is not None:
                records.append(
                    {
                        "session_id": thread_id,
                        "updated_at": saved.checkpoint["ts"],
                        "state": saved.checkpoint["channel_values"],
                    }
                )
        return records[:limit]

    def submit(self, **changes):
        return self.client.post("/api/form/submit", json={**FORM, **changes}, headers=USER)

    def say(self, session_id, message, headers=USER, **extra):
        return self.client.post(
            "/api/chat/message",
            json={"session_id": session_id, "user_message": message, **extra},
            headers=headers,
        )

    def history(self):
        return self.client.get("/api/chat/history", headers=USER)


class StubRuntime:
    def __init__(self, error):
        self.error = error

    async def invoke_chat(self, *args, **kwargs):
        raise self.error

    async def invoke_new_trip(self, *args, **kwargs):
        raise self.error


# ── Form ────────────────────────────────────────────────────────────


def test_a_sufficient_form_plans_a_trip_and_returns_its_session():
    world = World()

    response = world.submit()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["resolved_cities"] == ["Tokyo"] and body["destination_country_code"] == "JP"
    assert body["currency"] == "JPY" and [day["day"] for day in body["itinerary"]] == [1, 2, 3]
    assert set(body["daily_geojson_maps"]) == {"1", "2", "3"}
    assert world.app.store.rows[body["session_id"]] == ["user-1", "ready"]


def test_an_insufficient_budget_asks_for_confirmation_before_any_session():
    world = World(runtime=StubRuntime(AssertionError("the graph must not run")))

    body = world.submit(total_budget=10).json()

    assert body["status"] == "budget_confirmation_required"
    assert body["reason"] == "insufficient_budget" and body["stated_budget"] == 10
    assert body["resolved_cities"] == ["Tokyo"] and body["itinerary"] is None
    assert body["budget_assessment_id"] == world.budget.saved.assessment_id
    assert world.app.store.rows == {}


def test_a_recommendation_request_returns_the_grounded_minimum():
    body = World().submit(total_budget=None, request_budget_recommendation=True).json()

    assert body["status"] == "budget_confirmation_required"
    assert body["reason"] == "recommendation_requested" and body["stated_budget"] is None


def test_an_unsaved_assessment_cannot_be_offered_for_confirmation():
    body = World(budget=Budget(persisted=False)).submit(total_budget=10).json()

    assert body == {
        "status": "budget_check_unavailable",
        "reason": "assessment_cache_unavailable",
        "chat_reply": body["chat_reply"],
        "itinerary": None,
        "daily_geojson_maps": None,
    }


def test_a_confirmation_must_echo_its_cities_and_cover_the_shown_minimum():
    world = World()
    minimum = world.budget.saved.recommended_minimum_budget
    confirm = {"budget_assessment_id": world.budget.saved.assessment_id}

    no_city = world.submit(city=[], total_budget=minimum, **confirm).json()
    too_low = world.submit(total_budget=minimum - 1, **confirm).json()
    unknown = world.submit(total_budget=minimum, budget_assessment_id="other").json()

    for body in (no_city, too_low, unknown):
        assert body["status"] == "budget_check_unavailable"
        assert body["reason"] == "assessment_expired_or_invalid"
    assert world.destinations.calls == 2  # the city-less confirmation never resolved


def test_a_confirmed_recommendation_plans_the_trip():
    world = World()
    saved = world.budget.saved

    body = world.submit(total_budget=saved.recommended_minimum_budget, budget_assessment_id=saved.assessment_id).json()

    assert body["status"] == "success"


def test_form_input_and_onboarding_errors():
    assert World(profiles=Profiles({"name": "Ana"})).submit().status_code == 409
    assert World().submit(start_date="2026-11-1").status_code == 422
    assert World().submit(end_date="2026-10-01").status_code == 422

    body = World(destinations=Destinations(fail=True)).submit().json()
    assert body["status"] == "budget_check_unavailable"
    assert body["reason"] == "destination_resolution_unavailable"


# ── Chat ────────────────────────────────────────────────────────────


def planned_world(**kwargs):
    world = World(**kwargs)
    session_id = world.submit().json()["session_id"]
    return world, session_id


def test_a_chat_edit_returns_the_new_plan_flagged_as_modified():
    world, session = planned_world(app=App(agent=[edit_call({"day": 1, "action": "remove", "category": "activity", "index": 2})]))

    body = world.say(session, "remove activity 2 on day 1").json()

    assert set(body) == CHAT_SUCCESS_KEYS
    assert body["status"] == "success" and body["itinerary_modified"] is True
    assert [a["name"] for a in body["draft_itinerary"][0]["activities"]] == ["Senso-ji"]
    assert body["budget_confirmation"] is None


def test_a_question_returns_the_unchanged_plan_not_flagged():
    world, session = planned_world(app=App(agent=["Day 2 has Tokyo Tower."]))

    body = world.say(session, "what's on day 2?").json()

    assert body["status"] == "success" and body["itinerary_modified"] is False
    assert body["chat_reply"] == "Day 2 has Tokyo Tower."


def test_a_failed_change_keeps_the_graphs_own_explanation():
    world, session = planned_world(app=App(agent=[tool_call("update_trip_details", country="Korea")]))

    body = world.say(session, "actually make it Korea").json()

    assert set(body) == PLAN_FREE_KEYS | {"reason", "retryable"}
    assert body["status"] == "planning_unavailable" and body["retryable"] is True
    assert body["chat_reply"] == CHANGE_NOT_SAVED
    assert body["draft_itinerary"] == [] and body["itinerary_modified"] is False


def test_chat_errors_map_to_statuses():
    world, session = planned_world()

    assert world.say(session, "hi", headers={"Authorization": "Bearer someone-else"}).status_code == 404
    assert world.say("no-such-trip", "hi").status_code == 404
    assert world.say("x" * 129, "hi").status_code == 422
    assert world.say(session, "hi", headers={}).status_code == 401
    assert world.say(session, "ok", budget_action="accept_recommended").status_code == 422

    cases = {
        SessionBusy(): (409, "5"),
        asyncio.TimeoutError(): (504, None),
        InvalidTurnRequest("bad"): (422, None),
        ServiceNotReady(): (503, "5"),
        KeyError("secret detail"): (500, None),
    }
    for error, (code, retry_after) in cases.items():
        response = World(runtime=StubRuntime(error)).say(session, "hi")
        assert response.status_code == code
        assert response.headers.get("retry-after") == retry_after
        assert "secret" not in response.text and set(response.json()) == {"detail"}


def test_the_most_specific_error_rule_wins():
    assert error_response_for(google_errors.ResourceExhausted("quota")).status_code == 429
    assert error_response_for(google_errors.InternalServerError("boom")).status_code == 502


# ── History ─────────────────────────────────────────────────────────


def test_history_restores_messages_and_attaches_the_current_plan_once():
    world, session = planned_world(
        app=App(agent=[edit_call({"day": 1, "action": "remove", "category": "activity", "index": 2}), "Done."])
    )
    world.say(session, "remove activity 2 on day 1")

    response = world.history()

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"
    [restored] = response.json()["sessions"]
    assert restored["id"] == session and restored["title"] == "Tokyo, Japan"
    roles = [message["role"] for message in restored["messages"]]
    assert roles == ["user", "ai", "user", "ai"]
    with_plan = [message for message in restored["messages"] if "itinerary" in message]
    # Only the reply that published revision 2 carries the plan.
    assert with_plan == [restored["messages"][-1]]
    assert set(with_plan[0]) == {"role", "content", "itinerary", "maps", "budget", "destination_country_code"}
    assert with_plan[0]["budget"]["currency"] == "JPY"
    assert all(set(message) == {"role", "content"} for message in restored["messages"][:-1])
