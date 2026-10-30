"""AgentState adapter for the planning transaction.

Binds the booking, activity and routing services to one trip as the
transaction's three stages, runs the transaction off the event loop, and
returns only the state fields to update.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.modules.activities.service import ActivityPlanningService
from app.modules.booking.service import BookingPlanningService
from app.shared.observability import (
    LogContext,
    log_event,
)
from app.modules.itinerary.ports import (
    ActivityDraft,
    PlanningInputError,
    RouteDraft,
    SkeletonDraft,
)
from app.modules.itinerary.service import (
    PLANNING_MAX_ATTEMPTS,
    ItineraryPlanningService,
    PlanningResult,
)
from app.modules.routing.service import DailyRouteService
from app.orchestration.planning_requirements import build_trip_requirements
from app.orchestration.state import AgentState
from app.orchestration.workflows.activities import activity_request_from_state
from app.orchestration.workflows.booking import (
    booking_trip_from_state,
    plan_booking_skeleton,
)


logger = logging.getLogger(__name__)

# A retry of these can't change the outcome, so later graph turns must not
# start the transaction again either.
_FINAL_STOP_REASONS = frozenset(
    {"invalid_input", "booking_unrecoverable", "no_progress"}
)


class TripPlanningStages:
    """``PlanningStages`` for one trip held in ``AgentState``."""

    def __init__(
        self,
        state: AgentState,
        *,
        booking: BookingPlanningService,
        activities: ActivityPlanningService,
        routes: DailyRouteService,
        user_profile: Mapping[str, Any] | None = None,
    ) -> None:
        self._state = state
        self._booking = booking
        self._activities = activities
        self._routes = routes
        self._user_profile = user_profile

    def skeleton(self, *, reuse_evidence: bool) -> SkeletonDraft:
        try:
            booking_trip_from_state(self._state)
        except ValueError as exc:
            raise PlanningInputError(str(exc)) from exc

        result = plan_booking_skeleton(
            self._state,
            planner=self._booking,
            reuse_evidence=reuse_evidence,
        )

        return SkeletonDraft(
            itinerary=result.itinerary,
            reused_evidence=result.source == "assessment",
            missing=result.missing,
        )

    def activities(
        self,
        itinerary: Sequence[dict[str, Any]],
        *,
        days: frozenset[int] | None,
    ) -> ActivityDraft:
        if days is not None and not days:
            return ActivityDraft(itinerary=list(itinerary))

        result = self._activities.plan(
            activity_request_from_state(
                self._state,
                itinerary=itinerary,
                user_profile=self._user_profile,
                replan_days=days or (),
            )
        )

        return ActivityDraft(
            itinerary=result.itinerary,
            missing_days=result.missing_days,
        )

    def routes(
        self,
        itinerary: Sequence[dict[str, Any]],
        *,
        days: frozenset[int] | None,
    ) -> RouteDraft:
        result = self._routes.build_daily_maps(itinerary, days=days)

        return RouteDraft(
            itinerary=result.itinerary,
            maps=result.daily_maps,
        )


def log_context_from_config(config: Mapping[str, Any] | None) -> LogContext:
    """Request and session ids from a LangGraph ``RunnableConfig``."""

    values = config if isinstance(config, Mapping) else {}
    configurable = values.get("configurable")
    configurable = configurable if isinstance(configurable, Mapping) else {}
    metadata = values.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    return LogContext(
        request_id=configurable.get("request_id") or metadata.get("request_id"),
        session_id=(
            configurable.get("thread_id")
            or configurable.get("session_id")
            or metadata.get("thread_id")
        ),
    )


async def run_planning_transaction(
    state: AgentState,
    *,
    planner: ItineraryPlanningService,
    booking: BookingPlanningService,
    activities: ActivityPlanningService,
    routes: DailyRouteService,
    user_profile: Mapping[str, Any] | None = None,
    log_context: LogContext | None = None,
) -> dict[str, Any]:
    """Build a validated private candidate; the public plan is never touched.

    ``state.planning_attempts`` counts attempts across graph turns, so a
    re-entry only gets the attempts that are left.
    """

    remaining = PLANNING_MAX_ATTEMPTS - state.planning_attempts

    if remaining < 1:
        return {
            "planning_outcome": "unavailable",
            "candidate_plan": None,
        }

    try:
        requirements = build_trip_requirements(state)

    except ValueError:
        log_event(
            logger,
            "planning.complete",
            context=log_context,
            stage="requirements",
            outcome="unavailable",
            issue_codes=["planning.input.invalid"],
            reason="invalid_input",
        )
        return {
            "planning_outcome": "unavailable",
            "candidate_plan": None,
            "planning_attempts": PLANNING_MAX_ATTEMPTS,
            "planning_issue_codes": [
                *state.planning_issue_codes,
                "planning.input.invalid",
            ],
        }

    stages = TripPlanningStages(
        state,
        booking=booking,
        activities=activities,
        routes=routes,
        user_profile=user_profile,
    )

    try:
        result = await asyncio.to_thread(
            planner.build,
            requirements,
            stages,
            max_attempts=remaining,
            log_context=log_context,
        )

    except asyncio.CancelledError:
        # The worker thread finishes its current provider call; its result
        # is discarded and nothing reaches the state.
        log_event(
            logger,
            "planning.complete",
            context=(log_context or LogContext()).with_country(
                requirements.destination_country_code
            ),
            stage="transaction",
            outcome="interrupted",
            issue_codes=["transaction.cancelled"],
        )
        raise

    return planning_state_update(state, result)


def planning_state_update(
    state: AgentState,
    result: PlanningResult,
) -> dict[str, Any]:
    """Project a transaction result onto the private planning state fields."""

    attempts = state.planning_attempts + result.attempts

    candidate_plan = (
        {
            "itinerary": result.itinerary,
            "maps": result.maps,
            "attempt": attempts,
            "validation": result.report.model_dump(mode="json"),
        }
        if result.validated
        else None
    )

    return {
        "planning_outcome": result.status,
        "candidate_plan": candidate_plan,
        "planning_attempts": (
            PLANNING_MAX_ATTEMPTS
            if result.stop_reason in _FINAL_STOP_REASONS
            else attempts
        ),
        "planning_issue_codes": [
            *state.planning_issue_codes,
            *(issue.code for issue in result.issues),
        ],
    }


__all__ = [
    "TripPlanningStages",
    "log_context_from_config",
    "planning_state_update",
    "run_planning_transaction",
]
