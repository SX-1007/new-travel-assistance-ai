"""A trip change waiting on a recommendation is applied only with it."""

from app.modules.budget.models import BudgetAssessment
from app.orchestration.pending_trip import pending_trip_updates, pending_trip_values, with_pending_trip
from app.orchestration.tests.node_fakes import TRIP, assessment, committed_state


KOREA = assessment(TRIP.model_copy(update={"destination": "Korea", "destination_city": "Seoul"}))
WAITING = {"updates": {"country": "Korea", "city": ["Seoul"]}, "total_base_budget": 600.0}


def pending(**overrides):
    values = {"pending_trip_change": WAITING, "pending_budget_confirmation": {"assessment": KOREA}}
    values.update(overrides)
    return values


def test_the_waiting_change_brings_its_trip_total_and_currencies():
    overlay = pending_trip_values(pending())
    saved = BudgetAssessment.model_validate(KOREA)

    assert overlay["country"] == "Korea" and overlay["city"] == ["Seoul"]
    assert overlay["total_base_budget"] == 600.0
    assert overlay["budget_assessment"] == saved.model_dump()
    assert overlay["base_currency_code"] == saved.base_currency
    assert overlay["exchange_rate"] == {saved.base_currency: 1.0, saved.destination_currency: saved.exchange_rate}
    assert pending_trip_updates(pending()) == {"country": "Korea", "city": ["Seoul"]}


def test_nothing_applies_without_both_records_well_formed():
    for values in (
        {},
        pending(pending_budget_confirmation=None),
        pending(pending_trip_change=None),
        pending(pending_budget_confirmation={"assessment": {"bad": True}}),
        pending(pending_trip_change={**WAITING, "total_base_budget": -1}),
        pending(pending_trip_change={**WAITING, "updates": "Korea"}),
        # Only trip-detail fields may be laid over the session.
        pending(pending_trip_change={**WAITING, "updates": {"plan_revision": 9}}),
    ):
        assert pending_trip_values(values) == {}


def test_unknown_fields_in_the_change_are_ignored():
    values = pending(pending_trip_change={**WAITING, "updates": {"country": "Korea", "draft_itinerary": []}})

    assert pending_trip_updates(values) == {"country": "Korea"}


def test_the_state_view_is_a_copy():
    state = committed_state(**pending())

    view = with_pending_trip(state)

    assert view.country == "Korea" and state.country == "Japan"
    assert with_pending_trip(committed_state()) is not None
