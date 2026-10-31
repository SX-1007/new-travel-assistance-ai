"""Budget gate nodes: assessment, resuming an accepted budget, the gate reply."""

import pytest

from app.modules.budget.chat_gate import ChatBudgetDecision
from app.modules.budget.models import BudgetAssessment
from app.modules.output_review.service import OutputReviewService
from app.orchestration.nodes import budget as budget_nodes
from app.orchestration.nodes.budget import (
    ACCEPTED_STAGE_READY,
    assess_chat_budget_node,
    budget_gate_response_node,
    resume_accepted_budget_node,
)
from app.orchestration.nodes.replies import BUDGET_CONFIRMATION_TEXT, BUDGET_UNAVAILABLE_TEXT
from app.orchestration.tests.node_fakes import (
    CONFIG,
    TRIP,
    FakeBudget,
    ScriptedReviewer,
    assessment,
    committed_state,
    run,
)


ASSESSMENT = assessment()
MINIMUM = BudgetAssessment.model_validate(ASSESSMENT).recommended_minimum_budget


def gate_returns(monkeypatch, decision):
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)

        if isinstance(decision, Exception):
            raise decision

        return decision

    monkeypatch.setattr(budget_nodes, "evaluate_chat_budget", evaluate)
    return calls


def accepted(amount):
    return ChatBudgetDecision(
        status="accepted",
        chat_reply="ok",
        accepted_total_base_budget=amount,
        assessment=BudgetAssessment.model_validate(ASSESSMENT),
    )


# ── assess_chat_budget ──────────────────────────────────────────────


def test_accepted_amount_becomes_private_financials_without_a_provider_call(monkeypatch):
    gate_returns(monkeypatch, accepted(5000.0))
    state = committed_state(pending_budget_proposal={"mode": "amount", "total_base_budget": 5000.0})

    update = run(assess_chat_budget_node(state, CONFIG, budget=FakeBudget()))

    assert update["budget_gate_outcome"] == "accepted"
    record = update["accepted_budget_decision"]
    assert record["stage"] == ACCEPTED_STAGE_READY and record["target_plan_revision"] == 2
    overrides = update["plan_overrides"]
    assert overrides["total_base_budget"] == 5000.0
    assert overrides["total_convert_budget"] == 5000.0 * 32.0
    assert round(sum(overrides["budget_allocation"].values()), 2) == 160000.0
    # Public financials stay as they are until promotion.
    assert "total_base_budget" not in update and "budget_allocation" not in update
    assert update["planning_outcome"] is None and update["pending_budget_proposal"] is None


def test_acceptance_that_does_not_match_the_proposal_fails_closed(monkeypatch):
    gate_returns(monkeypatch, accepted(5000.0))
    state = committed_state(pending_budget_proposal={"mode": "amount", "total_base_budget": 7000.0})

    update = run(assess_chat_budget_node(state, CONFIG, budget=FakeBudget()))

    assert update["budget_gate_outcome"] == "budget_check_unavailable"
    assert update["budget_gate_reason"] == "invalid_accepted_decision"
    assert update["plan_overrides"] is None and update["accepted_budget_decision"] is None


def test_confirming_the_recommendation_uses_the_pending_record(monkeypatch):
    calls = gate_returns(monkeypatch, accepted(MINIMUM))
    pending = {
        "budget_assessment_id": ASSESSMENT["assessment_id"],
        "recommended_minimum_budget": MINIMUM,
        "assessment": ASSESSMENT,
    }
    state = committed_state(
        chat_budget_action="accept_recommended",
        chat_budget_assessment_id=ASSESSMENT["assessment_id"],
        pending_budget_confirmation=pending,
    )

    update = run(assess_chat_budget_node(state, CONFIG, budget=FakeBudget()))

    assert calls[0]["proposal"]["mode"] == "confirm"
    assert update["budget_gate_outcome"] == "accepted"
    assert update["pending_budget_confirmation"] is None


def test_confirmation_required_keeps_the_new_recommendation(monkeypatch):
    pending = {"budget_assessment_id": "a-1", "recommended_minimum_budget": 9000.0}
    gate_returns(
        monkeypatch,
        ChatBudgetDecision(
            status="budget_confirmation_required",
            chat_reply="confirm",
            reason="insufficient_budget",
            pending_confirmation=pending,
        ),
    )
    state = committed_state(pending_budget_proposal={"mode": "amount", "total_base_budget": 10.0})

    update = run(assess_chat_budget_node(state, CONFIG, budget=FakeBudget()))

    assert update["budget_gate_outcome"] == "budget_confirmation_required"
    assert update["pending_budget_confirmation"] == pending
    assert update["plan_overrides"] is None


def test_a_gate_exception_keeps_the_pending_recommendation(monkeypatch):
    gate_returns(monkeypatch, RuntimeError("cache down"))
    pending = {"budget_assessment_id": "a-1"}
    state = committed_state(pending_budget_proposal={"mode": "recommendation"}, pending_budget_confirmation=pending)

    update = run(assess_chat_budget_node(state, CONFIG, budget=FakeBudget()))

    assert update["budget_gate_reason"] == "budget_gate_exception"
    assert update["pending_budget_confirmation"] == pending


# ── resume_accepted_budget ──────────────────────────────────────────


def stored_acceptance(**record_overrides):
    record = {
        # Any legacy stage is rebuilt from the record's assessment.
        "stage": "activities",
        "accepted_total_base_budget": 5000.0,
        "assessment_id": ASSESSMENT["assessment_id"],
        "assessment": ASSESSMENT,
        "target_plan_revision": 2,
        "candidate": {"draft_itinerary": []},
        **record_overrides,
    }
    return committed_state(budget_gate_outcome="accepted", accepted_budget_decision=record)


def test_resume_reverifies_the_saved_assessment_then_plans():
    store = FakeBudget(saved=ASSESSMENT)

    update = run(resume_accepted_budget_node(stored_acceptance(), CONFIG, budget=store))

    assert store.loads == 1
    assert update["accepted_budget_decision"]["stage"] == ACCEPTED_STAGE_READY
    assert "candidate" not in update["accepted_budget_decision"]
    assert update["plan_overrides"]["total_base_budget"] == 5000.0


def test_resume_keeps_a_trip_change_accepted_with_the_budget():
    state = stored_acceptance().model_copy(update={"plan_overrides": {"num_people": 2, "total_base_budget": 5000.0}})
    store = FakeBudget(saved=ASSESSMENT)

    update = run(resume_accepted_budget_node(state, CONFIG, budget=store))

    assert update["plan_overrides"]["num_people"] == 2
    assert update["plan_overrides"]["total_base_budget"] == 5000.0


def test_accepting_a_waiting_trip_change_carries_it_into_the_replan(monkeypatch):
    korea = assessment(TRIP.model_copy(update={"destination": "Korea", "destination_city": "Seoul"}))
    decision = ChatBudgetDecision(
        status="accepted",
        chat_reply="ok",
        accepted_total_base_budget=5000.0,
        assessment=BudgetAssessment.model_validate(korea),
    )
    calls = gate_returns(monkeypatch, decision)
    state = committed_state(
        pending_budget_proposal={"mode": "amount", "total_base_budget": 5000.0},
        pending_trip_change={"updates": {"country": "Korea", "city": ["Seoul"]}, "total_base_budget": 600.0},
        pending_budget_confirmation={"assessment": korea},
        budget_gate_outcome="budget_confirmation_required",
    )

    update = run(assess_chat_budget_node(state, CONFIG, budget=FakeBudget()))

    # Judged against the changed trip, not the published Tokyo one.
    assert calls[0]["destination"] == "Korea" and calls[0]["destination_city"] == "Seoul"
    assert calls[0]["current_assessment"] == BudgetAssessment.model_validate(korea).model_dump()
    overrides = update["plan_overrides"]
    assert overrides["country"] == "Korea" and overrides["city"] == ["Seoul"]
    assert overrides["total_base_budget"] == 5000.0
    assert update["pending_trip_change"] is None and update["pending_budget_confirmation"] is None


def test_a_refused_waiting_change_stays_with_its_recommendation(monkeypatch):
    gate_returns(monkeypatch, RuntimeError("down"))
    waiting = {"updates": {"num_people": 2}, "total_base_budget": 600.0}
    state = committed_state(
        pending_budget_proposal={"mode": "amount", "total_base_budget": 5000.0},
        pending_trip_change=waiting,
        pending_budget_confirmation={"assessment": ASSESSMENT},
    )

    update = run(assess_chat_budget_node(state, CONFIG, budget=FakeBudget()))

    assert update["budget_gate_outcome"] == "budget_check_unavailable"
    # Not in the update: the waiting change is left as it was.
    assert "pending_trip_change" not in update


@pytest.mark.parametrize(
    ("state", "store", "reason"),
    [
        (stored_acceptance(), FakeBudget(saved=None), "assessment_expired_or_invalid"),
        (stored_acceptance(target_plan_revision=5), FakeBudget(saved=ASSESSMENT), "invalid_accepted_decision"),
        (stored_acceptance(assessment_id="other"), FakeBudget(saved=ASSESSMENT), "invalid_accepted_decision"),
        (stored_acceptance(), FakeBudget(matches=False, saved=ASSESSMENT), "invalid_accepted_decision"),
    ],
)
def test_resume_refuses_anything_that_does_not_hold_up(state, store, reason):
    update = run(resume_accepted_budget_node(state, CONFIG, budget=store))

    assert update["budget_gate_outcome"] == "budget_check_unavailable"
    assert update["budget_gate_reason"] == reason
    assert update["plan_overrides"] is None and update["pending_budget_confirmation"] is None


# ── budget_gate_response ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("outcome", "text"),
    [("budget_confirmation_required", BUDGET_CONFIRMATION_TEXT), ("budget_check_unavailable", BUDGET_UNAVAILABLE_TEXT)],
)
def test_gate_reply_is_fixed_text_with_no_model_review(outcome, text):
    semantic = ScriptedReviewer()
    state = committed_state(budget_gate_outcome=outcome)

    update = run(budget_gate_response_node(state, CONFIG, reviewer=OutputReviewService(reviewer=semantic)))

    assert update["messages"][0].content == text == update["budget_gate_message"]
    assert semantic.calls == 0
