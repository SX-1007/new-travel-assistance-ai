"""AgentState adapters for booking planning."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.modules.booking.models import (
    BookingTrip,
    SkeletonResult,
)
from app.modules.booking.selection import (
    accommodation_budget,
    transportation_budget,
)
from app.modules.booking.service import (
    BookingPlanningService,
)
from app.modules.budget.assessment import (
    assessment_matches_trip,
    parse_assessment,
    primary_destination_city,
)
from app.modules.budget.models import (
    BudgetTrip,
)
from app.orchestration.state import (
    AgentState,
)
from app.shared.countries import (
    country_alpha2,
)


def booking_trip_from_state(
    state: AgentState,
) -> BookingTrip:
    """
    Build the validated booking trip, or raise
    ValueError naming what is missing.
    """

    if not state.dest_currency_code:
        # Never guess a currency: prices would be compared against a budget
        # held in a different currency.
        raise ValueError(
            "dest_currency_code is required before booking"
        )

    country_code = country_alpha2(state.country)

    if country_code is None:
        raise ValueError(
            "destination country could not be verified"
        )

    try:
        return BookingTrip(
            origin_country=state.origin_country or "",
            destination_country=state.country or "",
            destination_country_code=country_code,
            city=primary_destination_city(state.city),
            start_date=state.start_date or "",
            end_date=state.end_date or "",
            currency=state.dest_currency_code.strip().upper(),
            adults=state.num_people or 1,
            transportation_budget=transportation_budget(
                state.budget_allocation
            ),
            accommodation_budget=accommodation_budget(
                state.budget_allocation
            ),
        )

    except ValidationError as exc:
        raise ValueError(
            "trip details are incomplete for booking"
        ) from exc


def _matching_evidence(
    state: AgentState,
    trip: BookingTrip,
) -> dict[str, Any] | None:
    """Budget-check evidence, only when it belongs to this exact trip."""

    assessment = parse_assessment(
        state.budget_assessment
    )

    if assessment is None:
        return None

    try:
        budget_trip = BudgetTrip(
            origin=trip.origin_country,
            destination=trip.destination_country,
            destination_city=trip.city,
            start_date=trip.start_date,
            end_date=trip.end_date,
            num_people=trip.adults,
        )

    except ValidationError:
        return None

    if not assessment_matches_trip(
        assessment,
        budget_trip,
    ):
        return None

    return assessment.evidence.model_dump()


def plan_booking_skeleton(
    state: AgentState,
    *,
    planner: BookingPlanningService,
    reuse_evidence: bool = True,
) -> SkeletonResult:
    """
    Build the day skeleton for this state's trip.

    With ``reuse_evidence=False`` the budget check's
    flights and hotel are ignored and a fresh search runs.
    """

    trip = booking_trip_from_state(
        state
    )

    return planner.plan_skeleton(
        trip,
        reuse=(
            _matching_evidence(
                state,
                trip,
            )
            if reuse_evidence
            else None
        ),
    )


def run_booking_planning(
    state: AgentState,
    *,
    planner: BookingPlanningService,
) -> dict[str, Any]:
    """
    Build the day skeleton and return only
    the state fields to update.
    """

    result = plan_booking_skeleton(
        state,
        planner=planner,
    )

    return {
        "draft_itinerary":
            result.itinerary,
    }


__all__ = [
    "booking_trip_from_state",
    "plan_booking_skeleton",
    "run_booking_planning",
]
