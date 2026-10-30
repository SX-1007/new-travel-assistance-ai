"""Budget-gate responses built only from server-owned assessments.

The chat graph stores a pending recommendation in
``pending_budget_confirmation`` (``modules/budget/chat_gate``); this module
checks that record against its embedded assessment and the session's trip
and renders the confirmation card and its text. Nothing a model wrote is
shown here.

The old router also re-read every pending assessment from Firestore, on
the event loop, for each live turn and each restored session. Accepting a
recommendation re-verifies it against the store anyway (the chat gate), so
display now checks it locally: same trip, arithmetic, and not expired.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Final, Literal

from pydantic import ValidationError

from app.api.schemas.budget import BudgetConfirmationResponse, BudgetEvidenceResponse
from app.modules.budget.assessment import (
    assessment_matches_trip,
    finite_positive,
    is_budget_sufficient,
    parse_assessment,
    primary_destination_city,
)
from app.modules.budget.models import BudgetAssessment, BudgetTrip
from app.orchestration.trip_intake import BudgetConfirmationRequired


BUDGET_CHECK_UNAVAILABLE_MESSAGE: Final = (
    "Unable to verify the budget right now. Your existing trip plan has not been changed."
)
EXPIRED_BUDGET_CONFIRMATION_MESSAGE: Final = (
    "This budget check expired or no longer matches the trip. Request a refreshed "
    "recommendation before planning."
)

GateKind = Literal["none", "confirmation", "unavailable"]


def form_confirmation(outcome: BudgetConfirmationRequired) -> BudgetConfirmationResponse:
    """The trip form's confirmation card for a fresh assessment."""

    return _confirmation(
        outcome.assessment,
        reason=outcome.reason,
        stated_budget=outcome.stated_budget,
        cities=list(outcome.cities),
        chat_reply=(
            "Here is the grounded minimum you requested. Confirm it or enter your "
            "own budget before itinerary planning."
            if outcome.reason == "recommendation_requested"
            else "Your entered budget is below the grounded minimum. Confirm the "
            "recommendation or enter another amount before itinerary planning."
        ),
    )


def chat_gate(
    state: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[GateKind, BudgetConfirmationResponse | None]:
    """What the session's budget gate shows, judged from its fields.

    ``none``: no gate is active. ``confirmation``: a valid recommendation
    awaits the user. ``unavailable``: the check failed, or the gate fields
    are inconsistent or expired (fail closed).
    """

    outcome = state.get("budget_gate_outcome")
    pending = state.get("pending_budget_confirmation")

    if outcome is None or outcome == "accepted":
        return ("none", None) if pending is None else ("unavailable", None)

    if outcome == "budget_confirmation_required":
        confirmation = pending_confirmation(pending, state, now=now)
        return ("confirmation", confirmation) if confirmation is not None else ("unavailable", None)

    return "unavailable", None


def pending_confirmation(
    raw: Any,
    state: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> BudgetConfirmationResponse | None:
    """The confirmation card for a stored pending record, or None when the
    record is inconsistent, expired or for another trip."""

    if not isinstance(raw, Mapping):
        return None

    assessment = parse_assessment(raw.get("assessment"))
    trip = _trip(state)

    if assessment is None or trip is None or not assessment_matches_trip(assessment, trip, now=now):
        return None

    evidence = assessment.evidence
    expected = {
        "budget_assessment_id": assessment.assessment_id,
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
    }

    if any(raw.get(key) != value for key, value in expected.items()):
        return None

    reason = raw.get("reason")
    stated = raw.get("stated_budget")

    if reason == "recommendation_requested":
        if stated is not None:
            return None
        stated_budget = None

    elif reason == "insufficient_budget":
        stated_budget = finite_positive(stated)

        # A stated amount that covers the minimum needs no confirmation.
        if stated_budget is None or is_budget_sufficient(stated_budget, assessment):
            return None

    else:
        return None

    cities = [str(city).strip() for city in state.get("city") or () if str(city).strip()]

    try:
        card = _confirmation(assessment, reason=reason, stated_budget=stated_budget, cities=cities, chat_reply="")
    except ValidationError:
        return None

    return card.model_copy(update={"chat_reply": confirmation_message(card)})


def confirmation_message(card: BudgetConfirmationResponse) -> str:
    """Server text for a confirmation, from the validated card only."""

    minimum = f"{card.base_currency} {card.recommended_minimum_budget:,.2f}"

    if card.stated_budget is None:
        return f"The provider-grounded minimum is {minimum}. Use the recommended budget to continue planning."

    return (
        f"{card.base_currency} {card.stated_budget:,.2f} is below the provider-grounded minimum "
        f"of {minimum}. Use the recommended budget to continue planning."
    )


def _confirmation(
    assessment: BudgetAssessment,
    *,
    reason: Literal["insufficient_budget", "recommendation_requested"],
    stated_budget: float | None,
    cities: list[str],
    chat_reply: str,
) -> BudgetConfirmationResponse:
    evidence = assessment.evidence

    return BudgetConfirmationResponse(
        status="budget_confirmation_required",
        reason=reason,
        chat_reply=chat_reply,
        budget_assessment_id=assessment.assessment_id,
        resolved_cities=cities,
        stated_budget=stated_budget,
        recommended_minimum_budget=assessment.recommended_minimum_budget,
        base_currency=assessment.base_currency,
        destination_currency=assessment.destination_currency,
        expires_at=assessment.expires_at,
        evidence=BudgetEvidenceResponse(
            outbound_flight_price=evidence.outbound_flight_price,
            return_flight_price=evidence.return_flight_price,
            hotel_price_per_night=evidence.hotel_price_per_night,
            hotel_nights=evidence.hotel_nights,
        ),
    )


def _trip(state: Mapping[str, Any]) -> BudgetTrip | None:
    try:
        return BudgetTrip(
            origin=state.get("origin_country") or "",
            destination=state.get("country") or "",
            destination_city=primary_destination_city(state.get("city")),
            start_date=state.get("start_date") or "",
            end_date=state.get("end_date") or "",
            num_people=state.get("num_people") or 1,
        )
    except ValidationError:
        return None


__all__ = [
    "BUDGET_CHECK_UNAVAILABLE_MESSAGE",
    "EXPIRED_BUDGET_CONFIRMATION_MESSAGE",
    "chat_gate",
    "confirmation_message",
    "form_confirmation",
    "pending_confirmation",
]
