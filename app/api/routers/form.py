"""POST /api/form/submit: check the budget, then plan a brand-new trip.

The intake (profile origin, verified cities, budget gate) decides whether
planning may start; only then is a session id created and the graph run.
Exceptions from the graph are mapped to HTTP statuses in ``api/errors.py``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import CurrentUser, RequestId
from app.api.presenters.budget import form_confirmation
from app.api.presenters.plan import (
    PLANNING_UNAVAILABLE_MESSAGE,
    AcceptedPlan,
    accepted_plan,
    planning_unavailable_reason,
    promoted_reply,
)
from app.api.providers import Runtime, TripIntake
from app.api.schemas.budget import BudgetCheckUnavailableResponse
from app.api.schemas.common import PlanningUnavailableResponse
from app.api.schemas.trip import (
    FinalResponse,
    InitialFormRequest,
    TripSubmissionResponse,
)
from app.modules.destination.exceptions import DestinationResolutionInvalid
from app.orchestration.trip_intake import (
    BudgetCheckUnavailable,
    BudgetConfirmationRequired,
    OnboardingIncomplete,
    TripForm,
)
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)
router = APIRouter(tags=["form"])

_UNAVAILABLE_REPLIES = {
    "provider_data_unavailable": (
        "Current flight, hotel, or exchange-rate data is unavailable. Retry the budget check before planning."
    ),
    "assessment_cache_unavailable": (
        "The grounded prices could not be saved for confirmation. Retry the budget check before planning."
    ),
    "assessment_expired_or_invalid": (
        "This budget check expired or no longer matches the trip. Refresh it before planning."
    ),
    "destination_resolution_unavailable": (
        "I could not safely verify a planning city for this country right now. "
        "Please retry before itinerary planning."
    ),
}


@router.post(
    "/submit",
    response_model=TripSubmissionResponse,
    summary="Submit the initial trip form and start a new planning session",
)
async def submit_trip_form(
    form: InitialFormRequest,
    user_id: CurrentUser,
    request_id: RequestId,
    intake: TripIntake,
    runtime: Runtime,
) -> TripSubmissionResponse:
    log_context = LogContext(request_id=request_id)

    try:
        outcome = await intake.prepare(
            user_id,
            TripForm(
                country=form.country,
                cities=tuple(form.city),
                num_people=form.num_people,
                start_date=form.start_date,
                end_date=form.end_date,
                total_budget=form.total_budget,
                budget_assessment_id=form.budget_assessment_id,
            ),
            log_context=log_context,
        )
    except OnboardingIncomplete as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Onboarding incomplete: submit your profile (name, home country and "
            "state/province) via /api/profile before planning a trip.",
        ) from exc
    except DestinationResolutionInvalid as exc:
        # Our own fixed messages about the user's input; safe to return.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    if isinstance(outcome, BudgetCheckUnavailable):
        return BudgetCheckUnavailableResponse(
            status="budget_check_unavailable",
            reason=outcome.reason,
            chat_reply=_UNAVAILABLE_REPLIES[outcome.reason],
        )

    if isinstance(outcome, BudgetConfirmationRequired):
        return form_confirmation(outcome)

    # The budget covers the minimum: only now does a session exist.
    session_id = str(uuid.uuid4())
    log_context = LogContext(request_id=request_id, session_id=session_id)
    log_event(logger, "form.submit", context=log_context, stage="form", outcome="started")

    result = await runtime.invoke_new_trip(
        outcome.initial_state,
        thread_id=session_id,
        user_id=user_id,
        request_id=request_id,
    )
    state = result.state
    plan: AcceptedPlan | None = None
    reply: str | None = None

    if state.get("planning_outcome") != "unavailable":
        plan = await asyncio.to_thread(accepted_plan, state)
        reply = promoted_reply(state.get("messages") or (), plan.plan_revision) if plan else None

    if plan is None or reply is None:
        log_event(
            logger,
            "form.submit",
            context=log_context,
            stage="form",
            outcome="unavailable",
            issue_codes=["planning.unavailable"],
        )
        return PlanningUnavailableResponse(
            status="planning_unavailable",
            reason=planning_unavailable_reason(state),
            chat_reply=PLANNING_UNAVAILABLE_MESSAGE,
        )

    log_event(
        logger,
        "form.submit",
        context=log_context.with_country(plan.destination_country_code),
        stage="form",
        outcome="completed",
    )

    return FinalResponse(
        status="success",
        chat_reply=reply,
        itinerary=plan.draft_itinerary,
        daily_geojson_maps=plan.daily_map_info,
        destination_country_code=plan.destination_country_code,
        resolved_cities=list(outcome.cities),
        total_budget=plan.total_convert_budget,
        currency=plan.dest_currency_code,
        budget_allocation=plan.budget_allocation,
        session_id=session_id,
    )
