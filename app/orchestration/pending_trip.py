"""A chat trip change waiting on a budget recommendation.

When a trip change needs a bigger budget, ``check_trip_change`` keeps it in
``pending_trip_change`` beside the usual ``pending_budget_confirmation``.
Until the user answers, everything about that recommendation (the card, the
decision tools, accepting it) is judged against the trip with the change
applied, never the published one.

Works on plain mappings and on ``AgentState``, so the API presenter and the
graph read a pending change the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.modules.budget.assessment import finite_positive, parse_assessment
from app.orchestration.state import AgentState
from app.orchestration.tools.trip import TRIP_DETAIL_FIELDS


def pending_trip_updates(values: Mapping[str, Any] | AgentState) -> dict[str, Any]:
    """The waiting trip-detail changes, or {} when none is pending."""

    overlay = pending_trip_values(values)
    return {key: value for key, value in overlay.items() if key in TRIP_DETAIL_FIELDS}


def pending_trip_values(values: Mapping[str, Any] | AgentState) -> dict[str, Any]:
    """Fields to lay over the session while its trip change waits.

    The trip changes, the total in the new home currency, and the pending
    assessment with its currencies. {} unless both records are present and
    well formed: a change is never applied without its recommendation.
    """

    pending = _field(values, "pending_trip_change")
    confirmation = _field(values, "pending_budget_confirmation")

    if not isinstance(pending, Mapping) or not isinstance(confirmation, Mapping):
        return {}

    raw_updates = pending.get("updates")
    updates = (
        {key: value for key, value in raw_updates.items() if key in TRIP_DETAIL_FIELDS}
        if isinstance(raw_updates, Mapping)
        else {}
    )
    total = finite_positive(pending.get("total_base_budget"))
    assessment = parse_assessment(confirmation.get("assessment"))

    if not updates or total is None or assessment is None:
        return {}

    return {
        **updates,
        "total_base_budget": total,
        "budget_assessment": assessment.model_dump(),
        "base_currency_code": assessment.base_currency,
        "dest_currency_code": assessment.destination_currency,
        "exchange_rate": {
            assessment.base_currency: 1.0,
            assessment.destination_currency: assessment.exchange_rate,
        },
    }


def with_pending_trip(state: AgentState) -> AgentState:
    """``state`` as the pending recommendation sees it (a shallow copy)."""

    overlay = pending_trip_values(state)
    return state.model_copy(update=overlay) if overlay else state


def _field(values: Mapping[str, Any] | AgentState, name: str) -> Any:
    return values.get(name) if isinstance(values, Mapping) else getattr(values, name, None)


__all__ = [
    "pending_trip_updates",
    "pending_trip_values",
    "with_pending_trip",
]
