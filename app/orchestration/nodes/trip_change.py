"""Check a chat trip change the way the trip form checks a new trip.

Post-tool processing puts the accepted ``update_trip_details`` changes in
``plan_overrides`` (and a budget stated with them in
``pending_budget_proposal``). This node runs before any re-plan:

1. A new origin must be a recognised country (the onboarding rule).
2. A new destination country or city list is resolved again: the cities are
   verified inside the country, and a new country without cities gets a
   verified primary city, as on the form.
3. A new origin converts the total into the new home currency.
4. When the trip the budget belongs to changed (origin, destination, primary
   city, dates or travellers) or a budget was stated, the chat budget gate
   judges the total against that trip's provider-grounded minimum.

Then: covered → re-plan with the checked changes and the new assessment
(``replan_full`` for a new destination or origin, else ``replan_itinerary``);
too low → the change waits in ``pending_trip_change`` beside the usual
recommendation card; anything else → a fixed reply. The published plan is
untouched until a re-plan is promoted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.runnables import RunnableConfig

from app.modules.budget.assessment import finite_positive, is_budget_sufficient
from app.modules.budget.chat_gate import ChatBudgetDecision, evaluate_chat_budget
from app.modules.budget.currency import convert_amount
from app.modules.budget.models import BudgetTrip
from app.modules.budget.ports import CurrencyProviderPort
from app.modules.budget.service import BudgetAssessmentService
from app.modules.destination.exceptions import DestinationResolutionInvalid, DestinationResolutionUnavailable
from app.modules.destination.service import DestinationResolutionService
from app.modules.output_review.service import OutputReviewService
from app.orchestration.nodes.budget import assessment_financials, budget_trip_from_state
from app.orchestration.nodes.context import node_log_context
from app.orchestration.nodes.replies import (
    CHANGE_NOT_SAVED,
    CURRENCY_UNAVAILABLE_TEXT,
    DESTINATION_INVALID_TEXT,
    DESTINATION_UNAVAILABLE_TEXT,
    ORIGIN_INVALID_TEXT,
    TRIP_CHANGE_REJECTED,
    server_reply,
)
from app.orchestration.nodes.transaction import committed_overrides, planning_reset
from app.orchestration.state import AgentState
from app.orchestration.tools.trip import TRIP_DETAIL_FIELDS
from app.shared.countries import country_alpha2
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

_FULL_REPLAN_FIELDS: Final = frozenset({"country", "origin_country"})


@dataclass(frozen=True, slots=True)
class TripChangeServices:
    """What checking a trip change may call."""

    destinations: DestinationResolutionService
    budget: BudgetAssessmentService
    currency: CurrencyProviderPort
    reviewer: OutputReviewService


class _Refused(Exception):
    """The change can't be planned; ``text`` tells the user why."""

    def __init__(self, text: str, code: str, *, unavailable: bool = False) -> None:
        super().__init__(code)
        self.text = text
        self.code = code
        # A provider outage (retryable), not a problem with what was asked.
        self.unavailable = unavailable


@dataclass(frozen=True, slots=True)
class _Checked:
    changes: dict[str, Any]
    # In the (new) home currency.
    total: float
    trip: BudgetTrip
    # None: the budget's trip is unchanged and no budget was stated.
    decision: ChatBudgetDecision | None


async def check_trip_change_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    services: TripChangeServices,
) -> dict[str, Any]:
    log_context = node_log_context(config)
    changes = {key: value for key, value in committed_overrides(state).items() if key in TRIP_DETAIL_FIELDS}

    if not changes:
        return {**_cleared(), "plan_overrides": None, "post_tool_outcome": "agent"}

    try:
        checked = await asyncio.to_thread(_check, state, changes, state.pending_budget_proposal, services)
    except _Refused as refusal:
        _log(log_context, "unavailable" if refusal.unavailable else "rejected", refusal.code)
        return _refused(state, refusal, services.reviewer, log_context)
    except Exception:
        refusal = _Refused(CHANGE_NOT_SAVED, "trip_change.failed", unavailable=True)
        _log(log_context, "failed", refusal.code)
        return _refused(state, refusal, services.reviewer, log_context)

    route = "replan_full" if checked.changes.keys() & _FULL_REPLAN_FIELDS else "replan_itinerary"
    decision = checked.decision

    if decision is None:
        _log(log_context, "validated")
        return {**_cleared(), "plan_overrides": checked.changes, "post_tool_outcome": route}

    if decision.status == "accepted":
        return _accepted(state, checked, decision, route, services.budget, log_context)

    if decision.status == "budget_confirmation_required" and decision.pending_confirmation is not None:
        _log(log_context, "confirmation_required", f"budget.{decision.reason or 'insufficient'}")
        return {
            **_cleared(),
            "plan_overrides": None,
            "pending_trip_change": {"updates": checked.changes, "total_base_budget": checked.total},
            "pending_budget_confirmation": decision.pending_confirmation,
            "budget_gate_outcome": "budget_confirmation_required",
            "budget_gate_reason": decision.reason,
            "budget_gate_message": None,
            "post_tool_outcome": "budget_gate",
        }

    return _budget_unavailable(decision.reason or "budget_check_unavailable", log_context)


# ── Checks ──────────────────────────────────────────────────────────


def _check(
    state: AgentState,
    changes: dict[str, Any],
    proposal: Any,
    services: TripChangeServices,
) -> _Checked:
    changes = dict(changes)
    origin = changes.get("origin_country")

    # The same rule as onboarding: the origin decides the home currency.
    if origin is not None and country_alpha2(origin) is None:
        raise _Refused(ORIGIN_INVALID_TEXT, "trip_change.origin_invalid")

    if changes.keys() & {"country", "city"}:
        changes.update(_resolved_destination(state, changes, services.destinations))

    trip = budget_trip_from_state(state.model_copy(update=changes))

    if trip is None:
        raise _Refused(CHANGE_NOT_SAVED, "trip_change.incomplete_trip")

    total = _stated_total(state, proposal)

    if origin is not None:
        total = _in_new_home_currency(state, origin, total, services.currency)

    if trip == budget_trip_from_state(state) and proposal is None:
        return _Checked(changes=changes, total=total, trip=trip, decision=None)

    try:
        decision = evaluate_chat_budget(
            proposal=(
                {"mode": "recommendation", "total_base_budget": None}
                if _mode(proposal) == "recommendation"
                else {"mode": "amount", "total_base_budget": total}
            ),
            pending_confirmation=None,
            current_assessment=state.budget_assessment,
            origin=trip.origin,
            destination=trip.destination,
            destination_city=trip.destination_city,
            start_date=trip.start_date,
            end_date=trip.end_date,
            num_people=trip.num_people,
            assessment_service=services.budget,
        )
    except Exception:
        decision = ChatBudgetDecision(
            status="budget_check_unavailable",
            chat_reply="",
            reason="budget_gate_exception",
        )

    return _Checked(changes=changes, total=total, trip=trip, decision=decision)


def _resolved_destination(
    state: AgentState,
    changes: Mapping[str, Any],
    destinations: DestinationResolutionService,
) -> dict[str, Any]:
    if "city" in changes:
        requested = list(changes["city"])
    elif "country" in changes:
        # The current cities belong to the old country: pick a verified one.
        requested = []
    else:
        requested = list(state.city)

    try:
        destination = destinations.resolve(
            country=changes.get("country", state.country or ""),
            requested_cities=requested,
            start_date=changes.get("start_date", state.start_date or ""),
            end_date=changes.get("end_date", state.end_date or ""),
            num_people=changes.get("num_people", state.num_people or 1),
        )
    except DestinationResolutionInvalid as exc:
        raise _Refused(DESTINATION_INVALID_TEXT, "destination.resolution.invalid") from exc
    except DestinationResolutionUnavailable as exc:
        raise _Refused(DESTINATION_UNAVAILABLE_TEXT, "provider.destination.unavailable", unavailable=True) from exc

    resolved: dict[str, Any] = {"city": list(destination.cities)}

    # Only a changed country is written back: a city change keeps the
    # itinerary-only re-plan.
    if "country" in changes:
        resolved["country"] = destination.country

    return resolved


def _stated_total(state: AgentState, proposal: Any) -> float:
    """A total stated with the change, else the current one (home currency)."""

    if _mode(proposal) == "amount":
        amount = finite_positive(proposal.get("total_base_budget"))

        if amount is not None:
            return amount

    return float(state.total_base_budget or 0.0)


def _in_new_home_currency(
    state: AgentState,
    origin: str,
    total: float,
    currency: CurrencyProviderPort,
) -> float:
    """The total converted from the current home currency to the new origin's."""

    try:
        current = state.base_currency_code or currency.get_currency_code(state.origin_country or "")
        new = currency.get_currency_code(origin)

        if current.strip().upper() == new.strip().upper():
            return total

        return convert_amount(total, currency.get_currency_rate(current, new))

    except Exception as exc:
        raise _Refused(CURRENCY_UNAVAILABLE_TEXT, "currency.unavailable", unavailable=True) from exc


def _mode(proposal: Any) -> Any:
    return proposal.get("mode") if isinstance(proposal, Mapping) else None


# ── Outcomes ────────────────────────────────────────────────────────


def _accepted(
    state: AgentState,
    checked: _Checked,
    decision: ChatBudgetDecision,
    route: str,
    budget: BudgetAssessmentService,
    log_context: LogContext,
) -> dict[str, Any]:
    amount = finite_positive(decision.accepted_total_base_budget)
    assessment = decision.assessment

    # The gate's acceptance is re-checked against what was asked of it.
    if (
        amount is None
        or amount != checked.total
        or assessment is None
        or not budget.matches(assessment, checked.trip)
        or not is_budget_sufficient(amount, assessment)
    ):
        return _budget_unavailable("invalid_accepted_decision", log_context)

    overrides: dict[str, Any] = {
        **checked.changes,
        "total_base_budget": amount,
        "budget_assessment": assessment.model_dump(),
    }

    # A full re-plan rebuilds the financials from this assessment itself; an
    # itinerary-only one keeps them unless the total changed.
    if route == "replan_itinerary" and amount != float(state.total_base_budget or 0.0):
        overrides.update(assessment_financials(amount, assessment))

    _log(log_context, "validated")
    return {**_cleared(), "plan_overrides": overrides, "post_tool_outcome": route}


def _budget_unavailable(reason: str, log_context: LogContext) -> dict[str, Any]:
    _log(log_context, "unavailable", f"budget.{reason}")
    return {
        **_cleared(),
        "plan_overrides": None,
        "budget_gate_outcome": "budget_check_unavailable",
        "budget_gate_reason": reason,
        "budget_gate_message": None,
        "post_tool_outcome": "budget_gate",
    }


def _refused(
    state: AgentState,
    refusal: _Refused,
    reviewer: OutputReviewService,
    log_context: LogContext,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "messages": [
            server_reply(
                state,
                refusal.text,
                marker=TRIP_CHANGE_REJECTED,
                reviewer=reviewer,
                log_context=log_context,
                fallback=CHANGE_NOT_SAVED,
            )
        ],
        **_cleared(),
        "plan_overrides": None,
        "post_tool_outcome": "reply_sent",
    }

    if refusal.unavailable:
        update["planning_outcome"] = "unavailable"
        update["planning_issue_codes"] = [refusal.code]

    return update


def _cleared() -> dict[str, Any]:
    # A budget stated with the change is consumed here, whatever the outcome.
    return {**planning_reset(), "pending_budget_proposal": None}


def _log(log_context: LogContext, outcome: str, *codes: str) -> None:
    log_event(logger, "planning.trip_change", context=log_context, stage="trip_change", outcome=outcome, issue_codes=list(codes))


__all__ = [
    "TripChangeServices",
    "check_trip_change_node",
]
