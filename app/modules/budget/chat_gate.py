"""Deterministic budget decisions for ongoing travel-planning chats.

Given a budget proposal from the chat (an amount, a request for a
recommendation, or a confirmation of the pending recommendation), return a
decision. Nothing here changes itinerary or session state; the orchestration
layer applies the decision.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Final, Literal, Optional

from pydantic import BaseModel, ValidationError

from app.modules.budget.assessment import (
    finite_positive,
    is_budget_sufficient,
    parse_assessment,
)
from app.modules.budget.exceptions import BudgetAssessmentUnavailable
from app.modules.budget.models import BudgetAssessment, BudgetTrip
from app.modules.budget.service import BudgetAssessmentService


ConfirmationReason = Literal["insufficient_budget", "recommendation_requested"]


class ChatBudgetDecision(BaseModel):
    """A non-mutating decision for one chat budget proposal."""

    status: Literal[
        "accepted", "budget_confirmation_required", "budget_check_unavailable"
    ]
    chat_reply: str
    reason: Optional[str] = None
    accepted_total_base_budget: Optional[float] = None
    assessment: Optional[BudgetAssessment] = None
    pending_confirmation: Optional[dict[str, Any]] = None


def pending_confirmation_payload(
    assessment: BudgetAssessment,
    *,
    reason: ConfirmationReason,
    stated_budget: float | None,
) -> dict[str, Any]:
    """Server-owned record of a recommendation awaiting explicit confirmation."""
    evidence = assessment.evidence
    return {
        "budget_assessment_id": assessment.assessment_id,
        "reason": reason,
        "stated_budget": stated_budget,
        "recommended_minimum_budget": assessment.recommended_minimum_budget,
        "base_currency": assessment.base_currency,
        "destination_currency": assessment.destination_currency,
        "expires_at": assessment.expires_at,
        "evidence": {
            "outbound_flight_price": evidence.outbound_flight_price,
            "return_flight_price": evidence.return_flight_price,
            "hotel_price_per_night": evidence.hotel_price_per_night,
            "hotel_nights": evidence.hotel_nights,
        },
        "assessment": assessment.model_dump(),
    }


def _unavailable(reason: str) -> ChatBudgetDecision:
    return ChatBudgetDecision(
        status="budget_check_unavailable",
        chat_reply="Unable to verify the budget using provider-backed data.",
        reason=reason,
    )


def _accepted(amount: float, assessment: BudgetAssessment) -> ChatBudgetDecision:
    return ChatBudgetDecision(
        status="accepted",
        chat_reply="Budget accepted.",
        accepted_total_base_budget=amount,
        assessment=assessment,
    )


def _confirmation_required(
    assessment: BudgetAssessment,
    *,
    reason: ConfirmationReason,
    stated_budget: float | None,
) -> ChatBudgetDecision:
    return ChatBudgetDecision(
        status="budget_confirmation_required",
        chat_reply="Please confirm the provider-backed recommended minimum budget.",
        reason=reason,
        assessment=assessment,
        pending_confirmation=pending_confirmation_payload(
            assessment,
            reason=reason,
            stated_budget=stated_budget,
        ),
    )


def _verified_against_store(
    candidate: BudgetAssessment,
    trip: BudgetTrip,
    service: BudgetAssessmentService,
) -> BudgetAssessment | None:
    """Accept state-held evidence only if the saved copy is identical."""
    # The service's clock, not the wall clock: expiry must be judged the same
    # way the service judged it when the assessment was saved.
    if not service.matches(candidate, trip):
        return None
    persisted = service.load_confirmed(trip, assessment_id=candidate.assessment_id)
    if persisted is None or persisted.model_dump() != candidate.model_dump():
        return None
    return persisted


def _confirmation_assessment(
    proposal: dict[str, Any],
    pending_confirmation: dict[str, Any],
    trip: BudgetTrip,
    service: BudgetAssessmentService,
) -> BudgetAssessment | None:
    pending = parse_assessment(pending_confirmation.get("assessment"))
    if pending is None:
        return None
    if pending_confirmation.get("budget_assessment_id") != pending.assessment_id:
        return None
    supplied_id = proposal.get("assessment_id")
    if supplied_id is not None and supplied_id != pending.assessment_id:
        return None
    return _verified_against_store(pending, trip, service)


def budget_change_proposal(
    proposed_total_base_budget: Any = None,
    *,
    request_recommendation: bool = False,
) -> dict[str, Any]:
    """Build the proposal ``evaluate_chat_budget`` accepts for an amount or a
    recommendation request. Raises ValueError for an invalid combination."""
    if request_recommendation:
        if proposed_total_base_budget is not None:
            raise ValueError("Recommendation mode cannot include an amount.")
        return {"mode": "recommendation", "total_base_budget": None}

    if proposed_total_base_budget is None:
        raise ValueError("Provide a positive budget or request a recommendation.")

    amount = finite_positive(proposed_total_base_budget)
    if amount is None:
        raise ValueError(
            "proposed_total_base_budget must be finite and greater than 0."
        )
    return {"mode": "amount", "total_base_budget": amount}


def confirm_recommendation_proposal() -> dict[str, Any]:
    """Build the proposal that accepts the pending server-owned recommendation."""
    return {"mode": "confirm", "total_base_budget": None}


# "5000", "5,000", "RM5000", "5.5k", "1.2m"; not a digit run inside a
# longer number.
_STATED_NUMBER: Final = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s?([kKmM])(?![a-zA-Z]))?"
)
_SUFFIX: Final = {"k": 1_000.0, "m": 1_000_000.0}
# Model rounding when it restates a converted amount.
_MATCH_TOLERANCE: Final = 0.005


def stated_amounts(text: str) -> tuple[float, ...]:
    """Every positive number the user wrote, with k/m suffixes applied."""

    amounts: list[float] = []

    for whole, fraction, suffix in _STATED_NUMBER.findall(text or ""):
        value = float(whole.replace(",", "") + (fraction or "")) * _SUFFIX.get(suffix.lower(), 1.0)

        if value > 0:
            amounts.append(value)

    return tuple(amounts)


def amount_is_stated(
    home_amount: float,
    stated: Iterable[float],
    *,
    rate: float,
    current_home_total: float | None = None,
) -> bool:
    """Whether a proposed home-currency total comes from the user's words.

    Each stated number may be in the home currency or (``rate``: destination
    units per home unit) the destination currency, and may be the new total
    itself or a change to ``current_home_total`` ("add 1000").
    """

    bases = [value for number in stated for value in (number, number / rate if rate > 0 else number)]
    candidates = list(bases)

    if current_home_total and current_home_total > 0:
        candidates += [current_home_total + value for value in bases]
        candidates += [current_home_total - value for value in bases]

    return any(
        candidate > 0 and abs(home_amount - candidate) <= max(0.01, _MATCH_TOLERANCE * candidate)
        for candidate in candidates
    )


def evaluate_chat_budget(
    *,
    proposal: dict[str, Any],
    pending_confirmation: dict[str, Any] | None,
    current_assessment: dict[str, Any] | None,
    origin: Any,
    destination: Any,
    destination_city: Any,
    start_date: Any,
    end_date: Any,
    num_people: Any,
    assessment_service: BudgetAssessmentService,
) -> ChatBudgetDecision:
    """Evaluate a chat proposal without changing itinerary or session state."""
    if not isinstance(proposal, dict):
        return _unavailable("invalid_budget_proposal")

    try:
        trip = BudgetTrip(
            origin=origin,
            destination=destination,
            destination_city=destination_city,
            start_date=start_date,
            end_date=end_date,
            num_people=num_people,
        )
    except ValidationError:
        return _unavailable("invalid_trip_details")

    mode = proposal.get("mode")

    if mode == "confirm":
        # A confirmation is valid only while a server-owned recommendation
        # is actually pending. This also distinguishes a stale/duplicate
        # confirmation from a genuinely expired assessment.
        if not isinstance(pending_confirmation, dict):
            return _unavailable("no_pending_budget_confirmation")

        assessment = _confirmation_assessment(
            proposal, pending_confirmation, trip, assessment_service
        )
        if assessment is None:
            return _unavailable("assessment_expired_or_invalid")

        return _accepted(assessment.recommended_minimum_budget, assessment)

    if mode not in {"amount", "recommendation"}:
        return _unavailable("invalid_budget_proposal")

    entered_budget = None
    if mode == "amount":
        entered_budget = finite_positive(proposal.get("total_base_budget"))
        if entered_budget is None:
            return _unavailable("invalid_budget_proposal")

    current = parse_assessment(current_assessment)
    assessment = (
        _verified_against_store(current, trip, assessment_service)
        if current is not None
        else None
    )
    persisted = assessment is not None

    if assessment is None:
        try:
            result = assessment_service.get_or_create(trip)
        except (BudgetAssessmentUnavailable, ValidationError):
            return _unavailable("provider_data_unavailable")
        assessment = result.assessment
        persisted = result.persisted

    if mode == "recommendation":
        if not persisted:
            return _unavailable("assessment_cache_unavailable")
        return _confirmation_required(
            assessment,
            reason="recommendation_requested",
            stated_budget=None,
        )

    if is_budget_sufficient(entered_budget, assessment):
        return _accepted(entered_budget, assessment)

    if not persisted:
        return _unavailable("assessment_cache_unavailable")

    return _confirmation_required(
        assessment,
        reason="insufficient_budget",
        stated_budget=entered_budget,
    )


__all__ = [
    "amount_is_stated",
    "stated_amounts",
    "ChatBudgetDecision",
    "pending_confirmation_payload",
    "budget_change_proposal",
    "confirm_recommendation_proposal",
    "evaluate_chat_budget",
]
