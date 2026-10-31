"""A chat trip change is checked like a new trip before any replan (B5, B6)."""

import pytest

from app.modules.budget.models import BudgetTrip
from app.modules.destination.exceptions import DestinationResolutionInvalid, DestinationResolutionUnavailable
from app.modules.output_review.service import OutputReviewService
from app.orchestration.nodes.replies import (
    CURRENCY_UNAVAILABLE_TEXT,
    DESTINATION_INVALID_TEXT,
    DESTINATION_UNAVAILABLE_TEXT,
    ORIGIN_INVALID_TEXT,
    SERVER_OWNED_KEY,
    TRIP_CHANGE_REJECTED,
)
from app.orchestration.nodes.trip_change import TripChangeServices, check_trip_change_node
from app.orchestration.tests.node_fakes import (
    CONFIG,
    AssessingBudget,
    FakeCurrency,
    FakeDestinations,
    ScriptedReviewer,
    committed_state,
    run,
)


def check(changes, *, budget=None, destinations=None, currency=None, **state):
    services = TripChangeServices(
        destinations=destinations or FakeDestinations(),
        budget=budget or AssessingBudget(),
        currency=currency or FakeCurrency(),
        reviewer=OutputReviewService(reviewer=ScriptedReviewer()),
    )
    update = run(check_trip_change_node(committed_state(plan_overrides=changes, **state), CONFIG, services=services))
    return update, services


def trip(**fields):
    values = dict(
        origin="Malaysia",
        destination="Japan",
        destination_city="Tokyo",
        start_date="2026-11-01",
        end_date="2026-11-03",
        num_people=1,
    )
    values.update(fields)
    return BudgetTrip(**values)


# ── Destination (B5) ────────────────────────────────────────────────


def test_a_new_country_gets_a_verified_city_and_its_own_budget_check():
    update, services = check({"country": "Korea"})

    # The Tokyo city list belongs to Japan: a new city is chosen and verified.
    assert services.destinations.calls[0]["requested_cities"] == []
    assert services.budget.created == [trip(destination="Korea", destination_city="Seoul")]
    assert update["post_tool_outcome"] == "replan_full"

    overrides = update["plan_overrides"]
    assert overrides["country"] == "Korea" and overrides["city"] == ["Seoul"]
    assert overrides["total_base_budget"] == 600.0
    assert overrides["budget_assessment"]["destination"] == "Korea"
    assert overrides["budget_assessment"]["destination_currency"] == "KRW"


def test_requested_cities_are_verified_inside_the_new_country():
    update, services = check({"country": "Korea", "city": ["Busan", "Seoul"]})

    assert services.destinations.calls[0]["requested_cities"] == ["Busan", "Seoul"]
    assert update["plan_overrides"]["city"] == ["Busan", "Seoul"]
    assert update["plan_overrides"]["budget_assessment"]["destination_city"] == "Busan"


def test_a_city_change_is_verified_and_keeps_the_itinerary_only_replan():
    update, services = check({"city": ["Osaka"]})

    assert services.destinations.calls[0]["country"] == "Japan"
    assert update["post_tool_outcome"] == "replan_itinerary"
    assert "country" not in update["plan_overrides"]
    # A new primary city is a new budget trip.
    assert services.budget.created == [trip(destination_city="Osaka")]


def test_adding_a_second_city_needs_no_new_budget_check():
    update, services = check({"city": ["Tokyo", "Yokohama"]})

    assert services.budget.created == []
    assert update["post_tool_outcome"] == "replan_itinerary"
    assert update["plan_overrides"] == {"city": ["Tokyo", "Yokohama"]}


def test_a_destination_outside_the_country_is_refused_with_the_plan_unchanged():
    update, services = check(
        {"city": ["Paris"]},
        destinations=FakeDestinations(error=DestinationResolutionInvalid("'Paris' could not be verified inside Japan.")),
    )

    assert update["post_tool_outcome"] == "reply_sent" and update["plan_overrides"] is None
    message = update["messages"][0]
    assert message.content == DESTINATION_INVALID_TEXT
    assert message.additional_kwargs[SERVER_OWNED_KEY] == TRIP_CHANGE_REJECTED
    # Bad input, not an outage: the response still carries the current plan.
    assert "planning_outcome" not in update or update["planning_outcome"] is None
    assert services.budget.created == []


def test_a_destination_check_outage_is_a_retryable_failure():
    update, _ = check({"country": "Korea"}, destinations=FakeDestinations(error=DestinationResolutionUnavailable("down")))

    assert update["messages"][0].content == DESTINATION_UNAVAILABLE_TEXT
    assert update["planning_outcome"] == "unavailable"
    assert update["planning_issue_codes"] == ["provider.destination.unavailable"]
    assert update["plan_overrides"] is None


# ── Origin (B6) ─────────────────────────────────────────────────────


def test_a_new_origin_converts_the_budget_to_the_new_home_currency():
    update, services = check(
        {"origin_country": "Singapore"},
        currency=FakeCurrency(rates={("MYR", "SGD"): 0.5}),
    )

    overrides = update["plan_overrides"]
    # MYR 600 is SGD 300, not SGD 600.
    assert overrides["total_base_budget"] == 300.0
    assert overrides["origin_country"] == "Singapore"
    assert overrides["budget_assessment"]["base_currency"] == "SGD"
    assert services.budget.created == [trip(origin="Singapore")]
    assert update["post_tool_outcome"] == "replan_full"


def test_an_origin_with_the_same_currency_keeps_the_amount(monkeypatch):
    monkeypatch.setitem(FakeCurrency.CODES, "brunei darussalam", "MYR")
    currency = FakeCurrency(rates={("MYR", "MYR"): 0.5})

    update, _ = check({"origin_country": "Brunei Darussalam"}, currency=currency)

    assert update["plan_overrides"]["total_base_budget"] == 600.0


def test_an_unrecognised_origin_is_refused_before_any_provider_call():
    currency = FakeCurrency()
    update, services = check({"origin_country": "Narnia"}, currency=currency)

    assert update["messages"][0].content == ORIGIN_INVALID_TEXT
    assert update["plan_overrides"] is None and update["post_tool_outcome"] == "reply_sent"
    assert currency.calls == 0 and services.budget.created == []


def test_a_currency_outage_never_keeps_the_amount_unconverted():
    update, services = check({"origin_country": "Singapore"}, currency=FakeCurrency(fail=True))

    assert update["messages"][0].content == CURRENCY_UNAVAILABLE_TEXT
    assert update["planning_outcome"] == "unavailable"
    assert update["planning_issue_codes"] == ["currency.unavailable"]
    assert update["plan_overrides"] is None and services.budget.created == []


# ── Budget gate ─────────────────────────────────────────────────────


def test_more_travellers_are_judged_against_the_new_trip():
    update, services = check({"num_people": 3})

    assert services.budget.created == [trip(num_people=3)]
    assert update["post_tool_outcome"] == "replan_itinerary"
    assert update["plan_overrides"]["num_people"] == 3
    assert update["plan_overrides"]["budget_assessment"]["num_people"] == 3
    # Same total: the category budgets the user set are kept.
    assert "budget_allocation" not in update["plan_overrides"]


def test_a_budget_too_low_for_the_new_trip_waits_with_its_recommendation():
    # Minimum: 8 x 1000 / 10 = MYR 800, above the MYR 600 budget.
    update, _ = check({"country": "Korea"}, budget=AssessingBudget(flight_price=1000.0))

    assert update["post_tool_outcome"] == "budget_gate"
    assert update["plan_overrides"] is None
    assert update["budget_gate_outcome"] == "budget_confirmation_required"
    assert update["budget_gate_reason"] == "insufficient_budget"
    assert update["pending_trip_change"] == {
        "updates": {"country": "Korea", "city": ["Seoul"]},
        "total_base_budget": 600.0,
    }

    pending = update["pending_budget_confirmation"]
    assert pending["stated_budget"] == 600.0 and pending["recommended_minimum_budget"] == 800.0
    assert pending["assessment"]["destination"] == "Korea"


def test_a_converted_budget_is_what_the_recommendation_judges():
    update, _ = check(
        {"origin_country": "Singapore"},
        budget=AssessingBudget(flight_price=1000.0),
        currency=FakeCurrency(rates={("MYR", "SGD"): 0.5}),
    )

    assert update["pending_trip_change"]["total_base_budget"] == 300.0
    assert update["pending_budget_confirmation"]["stated_budget"] == 300.0
    assert update["pending_budget_confirmation"]["base_currency"] == "SGD"


def test_an_unsaved_recommendation_is_not_offered():
    update, _ = check({"country": "Korea"}, budget=AssessingBudget(flight_price=1000.0, persist=False))

    assert update["budget_gate_outcome"] == "budget_check_unavailable"
    assert update["budget_gate_reason"] == "assessment_cache_unavailable"
    assert update.get("pending_trip_change") is None and update["plan_overrides"] is None


def test_a_budget_check_outage_leaves_the_plan_alone():
    update, _ = check({"num_people": 2}, budget=AssessingBudget(fail=True))

    assert update["post_tool_outcome"] == "budget_gate"
    assert update["budget_gate_outcome"] == "budget_check_unavailable"
    assert update["budget_gate_reason"] == "provider_data_unavailable"
    assert update["plan_overrides"] is None


def test_a_budget_stated_with_the_change_is_judged_and_applied_with_it():
    update, _ = check(
        {"num_people": 2},
        pending_budget_proposal={"mode": "amount", "total_base_budget": 900.0},
    )

    overrides = update["plan_overrides"]
    assert overrides["total_base_budget"] == 900.0
    # A new total gets fresh financials from the new assessment.
    assert overrides["total_convert_budget"] == 9000.0
    assert sum(overrides["budget_allocation"].values()) == pytest.approx(9000.0, abs=0.06)
    assert update["pending_budget_proposal"] is None


def test_a_recommendation_asked_with_the_change_is_shown_not_assumed():
    update, _ = check({"country": "Korea"}, pending_budget_proposal={"mode": "recommendation", "total_base_budget": None})

    assert update["budget_gate_outcome"] == "budget_confirmation_required"
    assert update["budget_gate_reason"] == "recommendation_requested"
    assert update["pending_budget_confirmation"]["stated_budget"] is None
    assert update["pending_trip_change"]["updates"]["country"] == "Korea"


@pytest.mark.parametrize(
    "changes, kwargs",
    [
        ({"country": "Korea"}, {}),
        ({"city": ["Paris"]}, {"destinations": FakeDestinations(error=DestinationResolutionInvalid("x"))}),
        ({"num_people": 2}, {"budget": AssessingBudget(fail=True)}),
        ({"country": "Korea"}, {"budget": AssessingBudget(flight_price=1000.0)}),
    ],
)
def test_a_stated_budget_is_consumed_whatever_the_outcome(changes, kwargs):
    update, _ = check(changes, pending_budget_proposal={"mode": "amount", "total_base_budget": 650.0}, **kwargs)

    # Left behind, the next turn would assess it against the old trip.
    assert update["pending_budget_proposal"] is None


def test_an_unexpected_error_fails_closed():
    class Broken(FakeDestinations):
        def resolve(self, **request):
            raise KeyError("boom")

    update, _ = check({"country": "Korea"}, destinations=Broken())

    assert update["post_tool_outcome"] == "reply_sent" and update["plan_overrides"] is None
    assert update["planning_outcome"] == "unavailable"
    assert update["planning_issue_codes"] == ["trip_change.failed"]
