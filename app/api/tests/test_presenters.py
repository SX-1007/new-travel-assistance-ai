"""Budget-gate rendering and history restore, on hand-built states."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, HumanMessage

from app.api.presenters.budget import (
    BUDGET_CHECK_UNAVAILABLE_MESSAGE,
    EXPIRED_BUDGET_CONFIRMATION_MESSAGE,
    chat_gate,
)
from app.api.presenters.chat import UNSAFE_HISTORY_MESSAGE, chat_turn_response, history_session
from app.modules.budget.chat_gate import pending_confirmation_payload
from app.modules.budget.models import BudgetAssessment
from app.modules.output_review.service import OutputReviewService
from app.orchestration.graph import TurnResult
from app.orchestration.tests.node_fakes import assessment


REVIEWER = OutputReviewService(reviewer=object())
TRIP_STATE = {
    "origin_country": "Malaysia",
    "country": "Japan",
    "city": ["Tokyo"],
    "num_people": 1,
    "start_date": "2026-11-01",
    "end_date": "2026-11-03",
}


def gate_state(*, reason="insufficient_budget", stated=10.0, **changes):
    saved = BudgetAssessment.model_validate(assessment())
    pending = pending_confirmation_payload(saved, reason=reason, stated_budget=stated)
    return {
        **TRIP_STATE,
        "budget_gate_outcome": "budget_confirmation_required",
        "pending_budget_confirmation": {**pending, **changes},
        "messages": [HumanMessage(content="make it 10"), AIMessage(content="Please confirm.")],
    }


def later(hours):
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def test_a_valid_pending_recommendation_renders_the_confirmation_card():
    response = chat_turn_response(TurnResult(gate_state(), plan_changed=False))

    assert response.status == "budget_confirmation_required"
    card = response.budget_confirmation
    assert card.resolved_cities == ["Tokyo"] and card.stated_budget == 10.0
    # The card's own text is filled (the old history left it empty, which
    # the client rejects).
    assert card.chat_reply == response.chat_reply and "below the provider-grounded minimum" in card.chat_reply


def test_an_inconsistent_or_expired_record_fails_closed():
    tampered = gate_state(recommended_minimum_budget=1.0)
    covered = gate_state(stated=1_000_000.0)
    contradictory = gate_state(reason="recommendation_requested", stated=10.0)

    for state in (tampered, covered, contradictory):
        assert chat_gate(state) == ("unavailable", None)

    assert chat_gate(gate_state(), now=later(24 * 30)) == ("unavailable", None)
    response = chat_turn_response(TurnResult(tampered, plan_changed=False))
    assert response.status == "budget_check_unavailable"
    assert response.chat_reply == BUDGET_CHECK_UNAVAILABLE_MESSAGE


def test_no_gate_but_a_leftover_record_is_unavailable():
    state = {**gate_state(), "budget_gate_outcome": None}

    assert chat_gate(state) == ("unavailable", None)
    assert chat_gate({**TRIP_STATE, "budget_gate_outcome": None}) == ("none", None)


def record(state, ts="2026-10-27T10:00:00+00:00", session_id="trip-1"):
    return {"session_id": session_id, "updated_at": ts, "state": state}


def test_history_shows_a_pending_card_or_says_it_expired():
    live = history_session(record(gate_state()), REVIEWER)
    expired = history_session(record(gate_state()), REVIEWER, now=later(24 * 30))

    assert live.messages[-1].budget_confirmation is not None
    assert live.messages[-1].content == live.messages[-1].budget_confirmation.chat_reply
    assert expired.messages[-1].budget_confirmation is None
    assert expired.messages[-1].content == EXPIRED_BUDGET_CONFIRMATION_MESSAGE


def test_history_hides_unsafe_replies_and_intermediate_tool_calls():
    state = {
        **TRIP_STATE,
        "messages": [
            HumanMessage(content="hi"),
            AIMessage(content="Let me look that up.", tool_calls=[{"name": "search_places", "args": {}, "id": "c1"}]),
            AIMessage(content='Here: {"draft_itinerary": []}'),
            HumanMessage(content="and?"),
            AIMessage(content="I have updated your itinerary."),
        ],
    }

    session = history_session(record(state), REVIEWER)

    assert [(m.role, m.content) for m in session.messages] == [
        ("user", "hi"),
        ("ai", UNSAFE_HISTORY_MESSAGE),
        ("user", "and?"),
        # No saved plan stands behind a success claim.
        ("ai", UNSAFE_HISTORY_MESSAGE),
    ]
    assert session.title == "Tokyo, Japan"


def test_history_skips_sessions_the_client_could_not_list():
    assert history_session(record(TRIP_STATE, ts=""), REVIEWER) is None
    assert history_session(record(TRIP_STATE, session_id=" "), REVIEWER) is None

    untitled = history_session(record({"messages": [HumanMessage(content="  plan   a trip ")]}), REVIEWER)
    assert untitled.title == "plan a trip"
