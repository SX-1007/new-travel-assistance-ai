"""AgentState adapter for the planning transaction.

Binds the booking, activity and routing services to one trip as the
transaction's three stages, runs the transaction off the event loop, and
returns only the state fields to update.

One planning run has ONE limit on paid place searches, shared by all of
its attempts. Its count is kept in ``planning_searches_used`` next to
``planning_attempts``, so a run continued in a later graph turn goes on
from the same count; ``planning_reset`` starts both again for a new run.

Cancelling the run (a request time limit, shutdown) stops the worker
thread too: a ``CancellationToken`` is checked before every stage and every
paid place search, and the coroutine waits (boundedly) for the thread to
stop, so whoever holds a planning slot holds it until the spending ends.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from datetime import date
from typing import Any, Final

from app.modules.activities.policies import preferences_from_profile
from app.modules.activities.service import ActivityPlanningService
from app.modules.booking.service import BookingPlanningService
from app.shared.cancellation import CancellationToken
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
from app.modules.places.policies import SearchQuota
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
    {"invalid_input", "booking_unrecoverable", "no_progress", "provider_quota_exhausted"}
)

# How long a cancelled run waits for its worker thread to reach its next
# check: one provider call with its retries (SerpAPI: 15 s per try). A
# slower stop is logged as ``transaction.stop_overdue`` and not waited for.
STOP_GRACE_SECONDS: Final = 60.0


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
        search_quota: SearchQuota | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        self._state = state
        self._booking = booking
        self._activities = activities
        self._routes = routes
        self._user_profile = user_profile
        self.cancellation = cancellation or CancellationToken()
        self.search_quota = search_quota or plan_search_quota(
            state, activities, user_profile, cancellation=self.cancellation
        )

    def skeleton(self, *, reuse_evidence: bool) -> SkeletonDraft:
        self.cancellation.raise_if_cancelled()

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

        self.cancellation.raise_if_cancelled()
        result = self._activities.plan(
            activity_request_from_state(
                self._state,
                itinerary=itinerary,
                user_profile=self._user_profile,
                replan_days=days or (),
            ),
            quota=self.search_quota,
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
        self.cancellation.raise_if_cancelled()
        result = self._routes.build_daily_maps(itinerary, days=days)

        return RouteDraft(
            itinerary=result.itinerary,
            maps=result.daily_maps,
        )


def plan_search_quota(
    state: AgentState,
    activities: ActivityPlanningService,
    user_profile: Mapping[str, Any] | None,
    *,
    cancellation: CancellationToken | None = None,
) -> SearchQuota:
    """The run's search limit, counting searches it already used."""

    return activities.search_quota(
        days=_trip_days(state),
        pacing=preferences_from_profile(user_profile).pacing,
        used=state.planning_searches_used,
        cancellation=cancellation,
    )


def _trip_days(state: AgentState) -> int:
    try:
        start = date.fromisoformat(state.start_date or "")
        end = date.fromisoformat(state.end_date or "")
    except ValueError:
        return 1

    return max((end - start).days + 1, 1)


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
    executor: Executor | None = None,
    stop_grace_seconds: float = STOP_GRACE_SECONDS,
) -> dict[str, Any]:
    """Build a validated private candidate; the public plan is never touched.

    ``state.planning_attempts`` counts attempts across graph turns, so a
    re-entry only gets the attempts that are left. The transaction runs on
    ``executor`` (planning's own threads) or, without one, the loop's
    default pool. If this coroutine is cancelled, the worker is told to stop
    and awaited for up to ``stop_grace_seconds`` before the cancellation
    goes on.
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

    # As asyncio.to_thread does, but on the given pool: the context goes
    # along, so log fields and tracing stay attached.
    build = functools.partial(
        contextvars.copy_context().run,
        planner.build,
        requirements,
        stages,
        max_attempts=remaining,
        log_context=log_context,
    )
    worker = asyncio.get_running_loop().run_in_executor(executor, build)

    try:
        # Shielded: cancelling this coroutine must not detach the worker
        # future, which is still needed to learn when the thread stops.
        result = await asyncio.shield(worker)

    except asyncio.CancelledError:
        stopped = await _stop_worker(worker, stages.cancellation, stop_grace_seconds)
        log_event(
            logger,
            "planning.complete",
            context=(log_context or LogContext()).with_country(
                requirements.destination_country_code
            ),
            stage="transaction",
            outcome="interrupted",
            issue_codes=["transaction.cancelled" if stopped else "transaction.stop_overdue"],
        )
        raise

    return planning_state_update(state, result, searches_used=stages.search_quota.used)


async def _stop_worker(
    worker: asyncio.Future[Any],
    cancellation: CancellationToken,
    grace_seconds: float,
) -> bool:
    """Tell the worker to stop and wait for it; True once it has stopped.

    The worker raises ``OperationCancelled`` at its next check. Whatever it
    ends with is discarded: nothing reaches the state.
    """

    cancellation.cancel()

    if not worker.done():
        await asyncio.wait({worker}, timeout=grace_seconds)

    if not worker.done():
        return False

    if not worker.cancelled():
        # Retrieved, or asyncio logs "exception was never retrieved".
        worker.exception()

    return True


def planning_state_update(
    state: AgentState,
    result: PlanningResult,
    *,
    searches_used: int | None = None,
) -> dict[str, Any]:
    """Project a transaction result onto the private planning state fields.

    ``searches_used`` is the run's total so far (``None``: unchanged).
    """

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
        "planning_searches_used": (
            state.planning_searches_used if searches_used is None else searches_used
        ),
    }


__all__ = [
    "TripPlanningStages",
    "log_context_from_config",
    "plan_search_quota",
    "planning_state_update",
    "run_planning_transaction",
]
