"""Fakes shared by the graph-node tests (no network).

A committed trip is produced once by the real planner, router and validator
over the fake network edges of the end-to-end planning test, so edit and
promotion tests work on a plan that genuinely validates.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.modules.activities.service import ActivityPlanningService, ActivitySearchService
from app.modules.booking.service import BookingPlanningService
from app.modules.budget.assessment import assessment_matches_trip, build_assessment
from app.modules.budget.exceptions import BudgetAssessmentUnavailable
from app.modules.budget.models import BudgetAssessment, BudgetAssessmentResult, BudgetTrip
from app.modules.destination.models import ResolvedDestination
from app.modules.itinerary.service import ItineraryPlanningService
from app.modules.itinerary.tests.test_planning_end_to_end import (
    GOOD_IDEAS,
    FakeAirportGeocoder,
    FakeDirections,
    FakeFlights,
    FakeGeocoder,
    FakeHotels,
    FakeIdeas,
    FakeLocator,
    FakePlaceSearch,
    World,
)
from app.modules.itinerary.tests.test_planning_end_to_end import state as planning_state
from app.modules.output_review.models import OutputReviewDecision
from app.modules.output_review.service import OutputReviewService
from app.modules.places.service import VerifiedPlaceSearch
from app.modules.routing.service import DailyRouteService
from app.orchestration.nodes.context import ProfileLoader
from app.orchestration.nodes.planning import PlanningServices
from app.orchestration.nodes.post_tool import EditServices
from app.orchestration.nodes.replies import ReplyServices
from app.orchestration.state import AgentState


CONFIG = {"configurable": {"thread_id": "session-1", "request_id": "req-1", "__user_id": "user-1"}}

REJECT = OutputReviewDecision.reject("reply.unsupported_claim", feedback="Only state supported facts.")
UNAVAILABLE = RuntimeError("reviewer down")


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# ── Models ──────────────────────────────────────────────────────────


class FakeModel:
    """Returns scripted replies (an AIMessage, text, or an exception to raise)."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[list[Any]] = []
        self.tools: list[str] = []

    async def ainvoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        outcome = self.responses.pop(0) if self.responses else AIMessage(content="")

        if isinstance(outcome, Exception):
            raise outcome

        return outcome if isinstance(outcome, AIMessage) else AIMessage(content=outcome)

    def bind_tools(self, tools: Any) -> FakeModel:
        bound = FakeModel()
        bound.tools = [tool.name for tool in tools]
        return bound


def tool_call(name: str, call_id: str = "call-1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def tool_result(name: str, payload: Any, call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(content=json.dumps(payload), tool_call_id=call_id, name=name)


def tool_turn(request: str, name: str, payload: Any, call_id: str = "call-1", **args: Any) -> list[Any]:
    return [HumanMessage(content=request), tool_call(name, call_id, **args), tool_result(name, payload, call_id)]


# ── Review ──────────────────────────────────────────────────────────


class ScriptedReviewer:
    """Semantic reviewer port: scripted decisions, approval by default."""

    def __init__(self, *decisions: Any) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def review(self, payload: Any) -> OutputReviewDecision:
        self.calls += 1
        decision = self.decisions.pop(0) if self.decisions else OutputReviewDecision.approve()

        if isinstance(decision, Exception):
            raise decision

        return decision


def reply_services(*decisions: Any, writer: FakeModel | None = None) -> tuple[ReplyServices, ScriptedReviewer]:
    semantic = ScriptedReviewer(*decisions)
    return ReplyServices(reviewer=OutputReviewService(reviewer=semantic), writer=writer or FakeModel()), semantic


# ── Profiles ────────────────────────────────────────────────────────


class ProfileStore:
    def __init__(self, profile: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self.profile = profile or {"interests": ["temples"]}
        self.fail = fail
        self.calls = 0

    def fetch(self, user_id: str) -> dict[str, Any]:
        self.calls += 1

        if self.fail:
            raise RuntimeError("profile store down")

        return dict(self.profile)


def profiles(store: ProfileStore | None = None) -> tuple[ProfileLoader, ProfileStore]:
    store = store or ProfileStore()
    return ProfileLoader(store.fetch), store


# ── Budget and currency ─────────────────────────────────────────────


TRIP = BudgetTrip(
    origin="Malaysia",
    destination="Japan",
    destination_city="Tokyo",
    start_date="2026-11-01",
    end_date="2026-11-03",
    num_people=1,
)


def assessment(trip: BudgetTrip = TRIP) -> dict[str, Any]:
    return build_assessment(
        trip,
        base_currency="MYR",
        destination_currency="JPY",
        exchange_rate=32.0,
        outbound_flight={"price": 300.0, "airline": "AirTest"},
        return_flight={"price": 250.0, "airline": "AirTest"},
        hotel={"price_per_night": 100.0, "hotel_name": "Park Hotel"},
        now=datetime.now(timezone.utc) - timedelta(minutes=1),
    ).model_dump()


class FakeBudget:
    """BudgetAssessmentService stand-in: ``matches`` and the saved copy are scripted."""

    def __init__(self, *, matches: bool = True, saved: dict[str, Any] | None = None) -> None:
        self._matches = matches
        self.saved = saved
        self.loads = 0

    def matches(self, assessment: Any, trip: BudgetTrip) -> bool:
        return self._matches

    def load_confirmed(self, trip: BudgetTrip, *, assessment_id: str) -> Any:
        self.loads += 1
        return BudgetAssessment.model_validate(self.saved) if self.saved else None


class AssessingBudget:
    """BudgetAssessmentService stand-in that builds, saves and matches
    assessments per trip with the real rules (minimum = 8 x one-way flight
    price in destination currency, at ``rate``)."""

    def __init__(self, *, flight_price: float = 275.0, rate: float = 10.0, fail: bool = False, persist: bool = True) -> None:
        self.flight_price = flight_price
        self.rate = rate
        self.fail = fail
        self.persist = persist
        self.saved: dict[str, BudgetAssessment] = {}
        self.created: list[BudgetTrip] = []

    def get_or_create(self, trip: BudgetTrip) -> BudgetAssessmentResult:
        if self.fail:
            raise BudgetAssessmentUnavailable("provider down")

        for saved in self.saved.values():
            if assessment_matches_trip(saved, trip):
                return BudgetAssessmentResult(assessment=saved, persisted=True)

        built = build_assessment(
            trip,
            base_currency=FakeCurrency.CODES[trip.origin.casefold()],
            destination_currency=FakeCurrency.CODES[trip.destination.casefold()],
            exchange_rate=self.rate,
            outbound_flight={"price": self.flight_price, "airline": "AirTest"},
            return_flight={"price": self.flight_price, "airline": "AirTest"},
            hotel={"price_per_night": 10.0, "hotel_name": "Park Hotel"},
            now=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        self.created.append(trip)

        if self.persist:
            self.saved[built.assessment_id] = built

        return BudgetAssessmentResult(assessment=built, persisted=self.persist)

    def matches(self, assessment: Any, trip: BudgetTrip) -> bool:
        return assessment_matches_trip(assessment, trip)

    def load_confirmed(self, trip: BudgetTrip, *, assessment_id: str) -> BudgetAssessment | None:
        saved = self.saved.get(assessment_id)
        return saved if saved is not None and assessment_matches_trip(saved, trip) else None


class FakeDestinations:
    """DestinationResolutionService stand-in: the requested cities, or the
    country's main city when none is given; ``error`` is raised instead."""

    CODES = {"japan": "JP", "korea": "KR", "france": "FR", "singapore": "SG", "malaysia": "MY"}
    MAIN_CITY = {"japan": "Tokyo", "korea": "Seoul", "france": "Paris", "singapore": "Singapore"}

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def resolve(self, **request: Any) -> ResolvedDestination:
        self.calls.append(request)

        if self.error is not None:
            raise self.error

        country = request["country"].strip()
        cities = request["requested_cities"] or [self.MAIN_CITY[country.casefold()]]
        return ResolvedDestination(
            country=country,
            country_code=self.CODES[country.casefold()],
            cities=tuple(cities),
            source="user" if request["requested_cities"] else "ai",
        )


class FakeCurrency:
    CODES = {"malaysia": "MYR", "japan": "JPY", "korea": "KRW", "singapore": "SGD", "france": "EUR"}

    def __init__(self, *, fail: bool = False, rates: dict[tuple[str, str], float] | None = None) -> None:
        self.fail = fail
        self.rates = rates or {}
        self.calls = 0

    def get_currency_code(self, country: str) -> str:
        self.calls += 1

        if self.fail:
            raise RuntimeError("currency provider down")

        return self.CODES[country.casefold()]

    def get_currency_rate(self, base: str, target: str) -> float:
        if self.fail:
            raise RuntimeError("currency provider down")

        return self.rates.get((base, target), 10.0)


# ── Planning world ──────────────────────────────────────────────────


class Network:
    """The fake network edges, with call counters."""

    def __init__(self) -> None:
        self.flights = FakeFlights()
        self.hotels = FakeHotels()
        self.places = FakePlaceSearch()
        self.directions = FakeDirections()
        self.ideas = FakeIdeas(GOOD_IDEAS)

    def verified_places(self) -> VerifiedPlaceSearch:
        return VerifiedPlaceSearch(search=self.places, geocoder=FakeGeocoder(), locator=FakeLocator())

    def routes(self) -> DailyRouteService:
        return DailyRouteService(directions=self.directions, geocoder=FakeAirportGeocoder())

    def planning(self, *, currency: FakeCurrency | None = None, budget: FakeBudget | None = None) -> PlanningServices:
        return PlanningServices(
            planner=ItineraryPlanningService(),
            booking=BookingPlanningService(flights=self.flights, hotels=self.hotels),
            activities=ActivityPlanningService(ideas=self.ideas, places=self.verified_places()),
            routes=self.routes(),
            currency=currency or FakeCurrency(),
            budget=budget or FakeBudget(),
        )

    def edits(self, reviewer: OutputReviewService | None = None) -> EditServices:
        return EditServices(
            reviewer=reviewer or OutputReviewService(reviewer=ScriptedReviewer()),
            place_search=ActivitySearchService(places=self.verified_places()),
            routes=self.routes(),
        )


@functools.lru_cache(maxsize=1)
def _planned() -> dict[str, Any]:
    update = World(ideas=FakeIdeas(GOOD_IDEAS)).run(planning_state())
    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]
    return update["candidate_plan"]


def candidate() -> dict[str, Any]:
    """A validated candidate for the 3-day Tokyo trip (fresh copy)."""

    return copy.deepcopy(_planned())


def trip_state(**overrides: Any) -> AgentState:
    """The 3-day Tokyo trip before any plan is committed."""

    return planning_state(total_base_budget=600.0, **overrides)


def committed_state(*messages: Any, **overrides: Any) -> AgentState:
    """The Tokyo trip with the validated plan committed as revision 1."""

    plan = candidate()
    snapshot = {
        "plan_revision": 1,
        "total_base_budget": 600.0,
        "base_currency_code": "MYR",
        "dest_currency_code": "JPY",
        "total_convert_budget": 6000.0,
        "budget_allocation": dict(planning_state().budget_allocation),
        "draft_itinerary": copy.deepcopy(plan["itinerary"]),
        "daily_map_info": copy.deepcopy(plan["maps"]),
    }
    values: dict[str, Any] = dict(
        messages=list(messages),
        draft_itinerary=plan["itinerary"],
        daily_map_info=plan["maps"],
        plan_revision=1,
        accepted_plan_snapshot=snapshot,
    )
    values.update(overrides)
    return trip_state(**values)
