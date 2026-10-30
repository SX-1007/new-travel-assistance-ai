"""AgentState adapter for activity planning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.modules.activities.models import (
    ActivityPlanRequest,
)
from app.modules.activities.policies import (
    preferences_from_profile,
)
from app.modules.activities.service import (
    ActivityPlanningService,
)
from app.orchestration.state import (
    AgentState,
)


def activity_request_from_state(
    state: AgentState,
    *,
    itinerary: Sequence[Mapping[str, Any]],
    user_profile: Mapping[str, Any] | None = None,
    replan_days: Iterable[int] = (),
) -> ActivityPlanRequest:
    """The activity-planning request for this state's trip."""

    allocation = state.budget_allocation or {}

    return ActivityPlanRequest(
        itinerary=itinerary,
        country=state.country or "",
        cities=tuple(state.city),
        num_people=state.num_people or 1,
        activity_budget=allocation.get("activity", 0.0),
        food_budget=allocation.get("food", 0.0),
        preferences=preferences_from_profile(user_profile),
        replan_days=tuple(sorted(replan_days)),
    )


def run_activity_planning(
    state: AgentState,
    *,
    planner: ActivityPlanningService,
    user_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fill the draft itinerary with verified
    activities and return only the state
    fields to update.
    """

    if not state.draft_itinerary:
        return {}

    result = planner.plan(
        activity_request_from_state(
            state,
            itinerary=state.draft_itinerary,
            user_profile=user_profile,
        )
    )

    return {
        "draft_itinerary":
            result.itinerary,
    }


__all__ = [
    "activity_request_from_state",
    "run_activity_planning",
]
