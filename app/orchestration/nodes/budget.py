"""Chat budget gate nodes: assess a proposal, resume an accepted one, reply.

Flow: post-tool processing records one coherent budget proposal →
``assess_chat_budget`` runs the deterministic gate → on acceptance the new
total, the confirmed assessment and the financials derived from it go into
``plan_overrides`` and the planning transaction runs (the current plan stays
public until promotion) → otherwise ``budget_gate_response`` replies.

Acceptance builds the financials at once from the assessment's own exchange
rate (no provider call). ``resume_accepted_budget`` handles a checkpoint left
with an accepted decision (an interrupted run, or a stage saved by an
earlier version): it re-verifies the assessment against the store before
planning.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app.modules.budget.allocation import initial_budget_allocation
from app.modules.budget.assessment import (
    finite_positive,
    is_budget_sufficient,
    parse_assessment,
    primary_destination_city,
)
from app.modules.budget.chat_gate import ChatBudgetDecision, evaluate_chat_budget
from app.modules.budget.currency import convert_amount
from app.modules.budget.models import BudgetAssessment, BudgetTrip
from app.modules.budget.service import BudgetAssessmentService
from app.modules.output_review.service import OutputReviewService
from app.orchestration.nodes.context import node_log_context
from app.orchestration.nodes.replies import (
    BUDGET_CONFIRMATION_TEXT,
    BUDGET_GATE,
    BUDGET_UNAVAILABLE_TEXT,
    server_reply,
)
from app.orchestration.nodes.transaction import planning_reset
from app.orchestration.state import AgentState
from app.orchestration.workflows.output_review import effective_plan_revision
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

# Stage of an accepted decision whose financials are already in
# ``plan_overrides``. Any other stage (saved by an earlier version) is
# rebuilt on resume.
ACCEPTED_STAGE_READY = "ready"


class _InvalidAcceptedDecision(ValueError):
    """The gate accepted without evidence that holds up locally."""


# ── Nodes ───────────────────────────────────────────────────────────


async def assess_chat_budget_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    budget: BudgetAssessmentService,
) -> dict[str, Any]:
    """Run the deterministic gate on this turn's budget proposal."""

    log_context = node_log_context(config)
    proposal = state.pending_budget_proposal

    if state.chat_budget_action == "accept_recommended":
        proposal = {
            "mode": "confirm",
            "total_base_budget": None,
            "assessment_id": state.chat_budget_assessment_id,
        }

    try:
        decision = ChatBudgetDecision.model_validate(
            await asyncio.to_thread(
                evaluate_chat_budget,
                proposal=dict(proposal or {}),
                pending_confirmation=state.pending_budget_confirmation,
                current_assessment=state.budget_assessment,
                origin=state.origin_country or "",
                destination=state.country or "",
                destination_city=primary_destination_city(state.city),
                start_date=state.start_date or "",
                end_date=state.end_date or "",
                num_people=state.num_people or 1,
                assessment_service=budget,
            )
        )
        update: dict[str, Any] = {
            "pending_budget_proposal": None,
            "budget_gate_outcome": decision.status,
            "budget_gate_reason": decision.reason,
            "budget_gate_message": None,
            "chat_budget_action": None,
            "chat_budget_assessment_id": None,
        }

        if decision.status == "accepted":
            amount, assessment = _verified_acceptance(state, proposal, decision, budget)
            update.update(_accepted_update(state, amount, assessment))
        else:
            update["plan_overrides"] = None
            update["pending_budget_confirmation"] = (
                decision.pending_confirmation
                if decision.pending_confirmation is not None
                else state.pending_budget_confirmation
            )

    except _InvalidAcceptedDecision:
        update = _assessment_failure(state, "invalid_accepted_decision")
    except Exception:
        update = _assessment_failure(state, "budget_gate_exception")

    _log(log_context, "budget.gate", update)
    return update


async def resume_accepted_budget_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    budget: BudgetAssessmentService,
) -> dict[str, Any]:
    """Re-verify an accepted decision found in the checkpoint, then plan it.

    A checkpoint marker is never authority on its own: the record must hold
    up locally and its assessment must still be the saved one.
    """

    log_context = node_log_context(config)
    parsed = _local_acceptance(state, budget)

    if parsed is None:
        update = _replan_failure("invalid_accepted_decision")
    else:
        amount, assessment, trip = parsed
        persisted = await asyncio.to_thread(
            _load_confirmed,
            budget,
            trip,
            assessment.assessment_id,
        )

        if persisted is None or persisted.model_dump() != assessment.model_dump():
            update = _replan_failure("assessment_expired_or_invalid")
        else:
            update = {
                "budget_gate_outcome": "accepted",
                **_accepted_update(state, amount, persisted),
            }

    _log(log_context, "budget.resume", update)
    return update


async def budget_gate_response_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    reviewer: OutputReviewService,
) -> dict[str, Any]:
    """Fixed reply for a budget that was not accepted (no model call).

    The chat router renders the recommendation itself from
    ``pending_budget_confirmation``.
    """

    text = (
        BUDGET_CONFIRMATION_TEXT
        if state.budget_gate_outcome == "budget_confirmation_required"
        else BUDGET_UNAVAILABLE_TEXT
    )
    message = server_reply(
        state,
        text,
        marker=BUDGET_GATE,
        reviewer=reviewer,
        log_context=node_log_context(config),
        fallback=BUDGET_UNAVAILABLE_TEXT,
    )

    return {
        "budget_gate_message": message.content,
        "messages": [message],
    }


# ── Acceptance ──────────────────────────────────────────────────────


def budget_decision_pending(state: AgentState) -> bool:
    """A recommendation awaits the user's answer: only the budget interpreter
    may act until it is resolved.

    A check that was merely unavailable, with nothing pending, does not lock
    the session: the trip can still be discussed normally.
    """

    return state.pending_budget_confirmation is not None


def budget_trip_from_state(state: AgentState) -> BudgetTrip | None:
    """The exact trip budget assessments belong to, or None if incomplete."""

    try:
        return BudgetTrip(
            origin=state.origin_country or "",
            destination=state.country or "",
            destination_city=primary_destination_city(state.city),
            start_date=state.start_date or "",
            end_date=state.end_date or "",
            num_people=state.num_people or 1,
        )
    except ValidationError:
        return None


def assessment_financials(total_base_budget: float, assessment: BudgetAssessment) -> dict[str, Any]:
    """Currency fields and a fresh allocation from an assessment's own rate."""

    converted = convert_amount(total_base_budget, assessment.exchange_rate)

    return {
        "base_currency_code": assessment.base_currency,
        "dest_currency_code": assessment.destination_currency,
        "exchange_rate": {
            assessment.base_currency: 1.0,
            assessment.destination_currency: assessment.exchange_rate,
        },
        "total_convert_budget": converted,
        "budget_allocation": initial_budget_allocation(converted),
        "currency_fetched_at": utc_timestamp(),
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _verified_acceptance(
    state: AgentState,
    proposal: Any,
    decision: ChatBudgetDecision,
    budget: BudgetAssessmentService,
) -> tuple[float, BudgetAssessment]:
    """Re-check an "accepted" decision against the proposal that produced it."""

    amount = finite_positive(decision.accepted_total_base_budget)
    assessment = decision.assessment
    trip = budget_trip_from_state(state)

    if (
        amount is None
        or assessment is None
        or trip is None
        or not isinstance(proposal, Mapping)
        or not budget.matches(assessment, trip)
        or not is_budget_sufficient(amount, assessment)
    ):
        raise _InvalidAcceptedDecision

    mode = proposal.get("mode")

    if mode == "amount":
        if finite_positive(proposal.get("total_base_budget")) != amount:
            raise _InvalidAcceptedDecision

    elif mode == "confirm":
        pending = state.pending_budget_confirmation

        if (
            not isinstance(pending, Mapping)
            or pending.get("budget_assessment_id") != assessment.assessment_id
            or proposal.get("assessment_id") not in {None, assessment.assessment_id}
            or finite_positive(pending.get("recommended_minimum_budget"))
            != assessment.recommended_minimum_budget
            or amount != assessment.recommended_minimum_budget
        ):
            raise _InvalidAcceptedDecision

    else:
        raise _InvalidAcceptedDecision

    return amount, assessment


def _local_acceptance(
    state: AgentState,
    budget: BudgetAssessmentService,
) -> tuple[float, BudgetAssessment, BudgetTrip] | None:
    """Parse a stored accepted decision without trusting it."""

    record = state.accepted_budget_decision

    if (
        state.budget_gate_outcome != "accepted"
        or state.pending_budget_confirmation is not None
        or not isinstance(record, Mapping)
    ):
        return None

    amount = finite_positive(record.get("accepted_total_base_budget"))
    assessment = parse_assessment(record.get("assessment"))
    target = record.get("target_plan_revision")
    trip = budget_trip_from_state(state)

    if (
        amount is None
        or assessment is None
        or trip is None
        or record.get("assessment_id") != assessment.assessment_id
        or isinstance(target, bool)
        or target != effective_plan_revision(state) + 1
        or not budget.matches(assessment, trip)
        or not is_budget_sufficient(amount, assessment)
    ):
        return None

    return amount, assessment, trip


def _accepted_update(state: AgentState, amount: float, assessment: BudgetAssessment) -> dict[str, Any]:
    dumped = assessment.model_dump()

    return {
        "accepted_budget_decision": {
            "stage": ACCEPTED_STAGE_READY,
            "accepted_total_base_budget": amount,
            "assessment_id": assessment.assessment_id,
            "assessment": dumped,
            "target_plan_revision": effective_plan_revision(state) + 1,
        },
        "plan_overrides": {
            "total_base_budget": amount,
            "budget_assessment": dumped,
            **assessment_financials(amount, assessment),
        },
        "pending_budget_confirmation": None,
        "post_tool_outcome": None,
        **planning_reset(),
    }


def _load_confirmed(
    budget: BudgetAssessmentService,
    trip: BudgetTrip,
    assessment_id: str,
) -> BudgetAssessment | None:
    try:
        return budget.load_confirmed(trip, assessment_id=assessment_id)
    except Exception:
        return None


# ── Failures ────────────────────────────────────────────────────────


def _gate_unavailable(reason: str) -> dict[str, Any]:
    return {
        "pending_budget_proposal": None,
        "accepted_budget_decision": None,
        "plan_overrides": None,
        "budget_gate_outcome": "budget_check_unavailable",
        "budget_gate_reason": reason,
        "budget_gate_message": BUDGET_UNAVAILABLE_TEXT,
        "chat_budget_action": None,
        "chat_budget_assessment_id": None,
    }


def _assessment_failure(state: AgentState, reason: str) -> dict[str, Any]:
    """The proposal could not be assessed; a pending recommendation stays."""

    return {
        **_gate_unavailable(reason),
        "pending_budget_confirmation": state.pending_budget_confirmation,
    }


def _replan_failure(reason: str) -> dict[str, Any]:
    return {**_gate_unavailable(reason), "pending_budget_confirmation": None}


def _log(log_context: LogContext, event: str, update: Mapping[str, Any]) -> None:
    outcome = update.get("budget_gate_outcome")

    log_event(
        logger,
        event,
        context=log_context,
        stage="budget_gate",
        outcome="unavailable" if outcome == "budget_check_unavailable" else str(outcome or "unknown"),
        reason=update.get("budget_gate_reason"),
    )


__all__ = [
    "ACCEPTED_STAGE_READY",
    "assess_chat_budget_node",
    "assessment_financials",
    "budget_decision_pending",
    "budget_gate_response_node",
    "budget_trip_from_state",
    "resume_accepted_budget_node",
    "utc_timestamp",
]
