"""New-trip intake: everything the trip form needs before the graph may run.

    onboarding profile (origin) → verified planning cities → budget gate
        → ReadyToPlan(initial graph state)
        | BudgetConfirmationRequired | BudgetCheckUnavailable

No session id is created and no graph node runs until the entered budget
covers the provider-grounded minimum. A confirmation round trip must echo
the cities and assessment id from the previous response.

A profile-store outage raises ``ProfileUnavailable`` (503); it is never read
as an empty profile, which would wrongly send the user back to onboarding.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.messages import HumanMessage

from app.modules.budget.assessment import is_budget_sufficient
from app.modules.budget.exceptions import BudgetAssessmentUnavailable
from app.modules.budget.models import BudgetAssessment, BudgetTrip
from app.modules.budget.service import BudgetAssessmentService
from app.modules.destination.exceptions import DestinationResolutionUnavailable
from app.modules.destination.service import DestinationResolutionService
from app.modules.profile.models import OnboardingProfile
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

UnavailableReason = Literal[
    "provider_data_unavailable",
    "assessment_cache_unavailable",
    "assessment_expired_or_invalid",
    "destination_resolution_unavailable",
]
ConfirmationReason = Literal["insufficient_budget", "recommendation_requested"]


class OnboardingIncomplete(Exception):
    """The user has no name and home country/state on file yet."""


class ProfileReader(Protocol):
    def get(self, user_id: str) -> OnboardingProfile | None: ...


@dataclass(frozen=True, slots=True)
class TripForm:
    """The validated trip form (the API schema maps onto this)."""

    country: str
    cities: tuple[str, ...]
    num_people: int
    start_date: str
    end_date: str
    # None: the user asked for a recommendation instead of an amount.
    total_budget: float | None
    # Set only when the user confirms a previously shown recommendation.
    budget_assessment_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReadyToPlan:
    initial_state: dict[str, Any]
    cities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BudgetConfirmationRequired:
    assessment: BudgetAssessment
    reason: ConfirmationReason
    stated_budget: float | None
    cities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BudgetCheckUnavailable:
    reason: UnavailableReason


IntakeOutcome = ReadyToPlan | BudgetConfirmationRequired | BudgetCheckUnavailable


class NewTripIntake:
    def __init__(
        self,
        *,
        profiles: ProfileReader,
        destinations: DestinationResolutionService,
        budget: BudgetAssessmentService,
    ) -> None:
        self._profiles = profiles
        self._destinations = destinations
        self._budget = budget

    async def prepare(
        self,
        user_id: str,
        form: TripForm,
        *,
        log_context: LogContext | None = None,
    ) -> IntakeOutcome:
        """Decide whether ``form`` may be planned now.

        Raises ``OnboardingIncomplete`` and ``DestinationResolutionInvalid``
        (bad user input); every provider failure is an outcome instead.
        """

        origin_country, origin_state = await self._origin(user_id)

        # A confirmation must echo the server-resolved cities: re-resolving
        # here could pick another city and break the assessment's identity.
        if form.budget_assessment_id and not form.cities:
            return BudgetCheckUnavailable("assessment_expired_or_invalid")

        try:
            destination = await asyncio.to_thread(
                self._destinations.resolve,
                country=form.country,
                requested_cities=list(form.cities),
                start_date=form.start_date,
                end_date=form.end_date,
                num_people=form.num_people,
            )
        except DestinationResolutionUnavailable:
            _log(log_context, "destination_resolution", "destination.resolution.unavailable")
            return BudgetCheckUnavailable("destination_resolution_unavailable")

        cities = destination.cities
        trip = BudgetTrip(
            origin=origin_country,
            destination=destination.country,
            destination_city=destination.primary_city,
            start_date=form.start_date,
            end_date=form.end_date,
            num_people=form.num_people,
        )
        amount = form.total_budget

        if form.budget_assessment_id:
            assessment = await asyncio.to_thread(
                self._budget.load_confirmed,
                trip,
                assessment_id=form.budget_assessment_id,
            )

            # The confirmed amount must still cover the minimum that was shown.
            if (
                assessment is None
                or amount is None
                or amount < assessment.recommended_minimum_budget
                or not is_budget_sufficient(amount, assessment)
            ):
                return BudgetCheckUnavailable("assessment_expired_or_invalid")

        else:
            try:
                result = await asyncio.to_thread(self._budget.get_or_create, trip)
            except BudgetAssessmentUnavailable:
                _log(log_context, "budget_gate", "provider.data.unavailable")
                return BudgetCheckUnavailable("provider_data_unavailable")

            assessment = result.assessment

            if not is_budget_sufficient(amount, assessment):
                # Only a saved assessment can be confirmed in a later request.
                if not result.persisted:
                    return BudgetCheckUnavailable("assessment_cache_unavailable")

                return BudgetConfirmationRequired(
                    assessment=assessment,
                    reason="recommendation_requested" if amount is None else "insufficient_budget",
                    stated_budget=amount,
                    cities=cities,
                )

        assert amount is not None  # sufficient implies an amount

        return ReadyToPlan(
            initial_state={
                "messages": [
                    HumanMessage(
                        content=f"Please analyse all the information and plan a perfect trip to {destination.country}"
                    )
                ],
                "origin_country": origin_country,
                "origin_state": origin_state,
                "country": destination.country,
                "city": list(cities),
                "num_people": form.num_people,
                "total_base_budget": float(amount),
                "start_date": form.start_date,
                "end_date": form.end_date,
                "budget_assessment": assessment.model_dump(),
            },
            cities=cities,
        )

    async def _origin(self, user_id: str) -> tuple[str, str]:
        profile = await asyncio.to_thread(self._profiles.get, user_id)

        if profile is None:
            raise OnboardingIncomplete

        return profile.home_country, profile.home_state


def _log(log_context: LogContext | None, stage: str, code: str) -> None:
    log_event(
        logger,
        "trip_intake.unavailable",
        context=log_context,
        stage=stage,
        outcome="unavailable",
        issue_codes=[code],
    )


__all__ = [
    "BudgetCheckUnavailable",
    "BudgetConfirmationRequired",
    "IntakeOutcome",
    "NewTripIntake",
    "OnboardingIncomplete",
    "ProfileReader",
    "ReadyToPlan",
    "TripForm",
]
