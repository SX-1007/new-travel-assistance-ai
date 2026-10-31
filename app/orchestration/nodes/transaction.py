"""The private planning transaction's state: overrides, resets and kinds.

A change that needs a new plan (new dates, a new destination, an accepted
budget, a category budget) is held in ``plan_overrides`` while the plan is
built and reviewed. The nodes plan, review and validate against
``transaction_view(state)``; only promotion copies the overrides into the
public fields. So a failed re-plan leaves the trip exactly as it was: never,
say, a 5-day date range over a 3-day itinerary.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Final

from app.orchestration.messages import current_turn_used_tool
from app.orchestration.state import AgentState
from app.orchestration.tools.trip import TRIP_DETAIL_FIELDS
from app.orchestration.workflows.output_review import CANDIDATE_FINANCIAL_FIELDS


# Fields a transaction may change; anything else in the overrides is ignored.
OVERRIDABLE_FIELDS: Final = frozenset(
    {*TRIP_DETAIL_FIELDS, *CANDIDATE_FINANCIAL_FIELDS, "budget_assessment"}
)


class PlanKind(str, Enum):
    """What the candidate being reviewed is, for its reply and fallback."""

    NEW = "new"
    EDIT = "edit"
    BUDGET = "budget"
    REQUIREMENTS = "requirements"
    ALLOCATION = "allocation"


def transaction_view(state: AgentState) -> AgentState:
    """``state`` as the running transaction sees it (overrides applied).

    A shallow copy: the nodes only read it, so nothing (least of all the
    message history) is deep-copied.
    """

    update = committed_overrides(state)
    return state.model_copy(update=update) if update else state


def committed_overrides(state: AgentState) -> dict[str, Any]:
    overrides = state.plan_overrides

    if not isinstance(overrides, Mapping):
        return {}

    return {key: value for key, value in overrides.items() if key in OVERRIDABLE_FIELDS}


def plan_kind(state: AgentState) -> PlanKind:
    if current_turn_used_tool(state.messages, "edit_itinerary"):
        return PlanKind.EDIT

    if state.budget_gate_outcome == "accepted" and isinstance(state.accepted_budget_decision, Mapping):
        return PlanKind.BUDGET

    overrides = committed_overrides(state)

    if overrides.keys() & set(TRIP_DETAIL_FIELDS):
        return PlanKind.REQUIREMENTS

    if "budget_allocation" in overrides:
        return PlanKind.ALLOCATION

    return PlanKind.NEW


def planning_reset() -> dict[str, Any]:
    """A cleared transaction (overrides are kept: they are its input)."""

    return {
        "planning_outcome": None,
        "candidate_plan": None,
        "planning_attempts": 0,
        "planning_issue_codes": [],
        "output_review_attempts": 0,
        "output_review_issue_codes": [],
    }


def finished_transaction() -> dict[str, Any]:
    """Everything a finished (committed or abandoned) transaction leaves behind."""

    return {
        "plan_overrides": None,
        "post_tool_outcome": None,
        "accepted_budget_decision": None,
        "pending_budget_proposal": None,
        "pending_budget_confirmation": None,
        "pending_itinerary_mutation": None,
        "budget_gate_outcome": None,
        "budget_gate_reason": None,
        "budget_gate_message": None,
        "chat_budget_action": None,
        "chat_budget_assessment_id": None,
    }


__all__ = [
    "OVERRIDABLE_FIELDS",
    "PlanKind",
    "committed_overrides",
    "finished_transaction",
    "plan_kind",
    "planning_reset",
    "transaction_view",
]
