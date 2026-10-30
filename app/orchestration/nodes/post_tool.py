"""Apply the latest tool batch to state, privately, and decide what follows.

``ToolNode`` only appends tool messages. This node reads the newest batch
(each result bound to the call that produced it) and records its decision in
``post_tool_outcome`` for the router:

1. Budget tools win: one coherent proposal goes to the gate
   (``assess_budget``); malformed or conflicting ones fail closed
   (``budget_gate``). While a recommendation is pending, other tools are
   ignored.
2. Trip-detail changes go into ``plan_overrides`` and trigger a replan
   (``replan_full`` for a new destination or origin, else
   ``replan_itinerary``). Edits in the same batch are dropped.
3. Itinerary edits and category budgets become a validated private
   candidate (``review_candidate``), or a fixed reply (``reply_sent``).
4. Anything else returns to the agent.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.runnables import RunnableConfig

from app.modules.activities.service import ActivitySearchService
from app.modules.budget.allocation import normalise_budget_allocation
from app.modules.budget.assessment import finite_positive
from app.modules.itinerary.editing import EditOutcome, apply_edits, reassess_flight_budget
from app.modules.itinerary.validators import validate_itinerary_candidate
from app.modules.output_review.service import OutputReviewService
from app.modules.routing.service import DailyRouteService
from app.orchestration.edit_intent import (
    DAY_CLARIFICATION_MARKER,
    day_authorizer,
    pending_mutation_record,
    recover_place_options,
    trusted_booking_options,
    trusted_place_options,
)
from app.orchestration.messages import latest_tool_batch, latest_user_request
from app.orchestration.nodes.budget import budget_decision_pending
from app.orchestration.nodes.context import node_log_context
from app.orchestration.nodes.replies import (
    EDIT_FAILED_VALIDATION,
    EDIT_NOT_APPLICABLE,
    EDIT_REJECTED,
    server_reply,
)
from app.orchestration.nodes.transaction import planning_reset
from app.orchestration.planning_requirements import build_trip_requirements
from app.orchestration.state import AgentState
from app.orchestration.tools.trip import validated_trip_updates
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

_BUDGET_TOOLS: Final = frozenset({"propose_budget_change", "confirm_recommended_budget"})
_FULL_REPLAN_FIELDS: Final = frozenset({"country", "origin_country"})
# Server-side place lookups per batch when the model skipped search_places.
_MAX_PLACE_RECOVERIES: Final = 3
# Cent rounding across six categories.
_ALLOCATION_TOLERANCE: Final = 0.06

ToolCall = tuple[str, dict[str, Any], Any]


@dataclass(frozen=True, slots=True)
class EditServices:
    """What applying an edit may call: the reviewer for fixed replies, the
    place search for recovery and the router for the changed days' maps."""

    reviewer: OutputReviewService
    place_search: ActivitySearchService
    routes: DailyRouteService


async def post_tool_processing_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    edits: EditServices,
) -> dict[str, Any]:
    log_context = node_log_context(config)
    batch = latest_tool_batch(state.messages)

    status, proposal = resolve_budget_batch(batch)

    if status == "coherent":
        return {
            "pending_budget_proposal": proposal,
            "budget_gate_outcome": None,
            "budget_gate_reason": None,
            "budget_gate_message": None,
            "post_tool_outcome": "assess_budget",
        }

    if status != "absent":
        _log(log_context, "budget_tools", "rejected", status)
        return {
            "pending_budget_proposal": None,
            "budget_gate_outcome": "budget_check_unavailable",
            "budget_gate_reason": status,
            "budget_gate_message": None,
            "chat_budget_action": None,
            "chat_budget_assessment_id": None,
            "post_tool_outcome": "budget_gate",
        }

    # While a recommendation is pending, only the restricted interpreter may
    # decide; any other tool output is ignored.
    if budget_decision_pending(state):
        return {"post_tool_outcome": "budget_gate"}

    changes = _trip_changes(state, batch)

    if changes:
        return {
            **planning_reset(),
            "plan_overrides": changes,
            "post_tool_outcome": "replan_full" if changes.keys() & _FULL_REPLAN_FIELDS else "replan_itinerary",
        }

    requested_edits = [
        edit
        for name, _, result in batch
        if name == "edit_itinerary" and isinstance(result, Mapping)
        for edit in (result.get("edits") or [])
    ]
    allocation = _allocation_change(state, batch, log_context)

    if not requested_edits and allocation is None:
        return {"post_tool_outcome": "agent"}

    return await _incremental_candidate(state, requested_edits, allocation, edits, log_context)


# ── Budget tools ────────────────────────────────────────────────────


def resolve_budget_batch(batch: Sequence[ToolCall]) -> tuple[str, dict[str, Any] | None]:
    """("absent" | "coherent" | a failure reason, the one proposal)."""

    results = [(name, result) for name, _, result in batch if name in _BUDGET_TOOLS]

    if not results:
        return "absent", None

    proposals = [_budget_proposal(name, result) for name, result in results]

    if any(proposal is None for proposal in proposals):
        return "invalid_budget_tool_result", None

    if any(proposal != proposals[0] for proposal in proposals[1:]):
        return "conflicting_budget_tool_results", None

    return "coherent", proposals[0]


def _budget_proposal(name: str, result: Any) -> dict[str, Any] | None:
    if not isinstance(result, Mapping) or result.get("status") != "budget_proposal":
        return None

    proposal = result.get("proposal")
    mode = proposal.get("mode") if isinstance(proposal, Mapping) else None

    if name == "confirm_recommended_budget":
        return {"mode": "confirm", "total_base_budget": None} if mode == "confirm" else None

    if mode == "recommendation":
        return {"mode": "recommendation", "total_base_budget": None}

    amount = finite_positive(proposal.get("total_base_budget")) if mode == "amount" else None
    return {"mode": "amount", "total_base_budget": amount} if amount is not None else None


# ── Requirements and allocation ─────────────────────────────────────


def _trip_changes(state: AgentState, batch: Sequence[ToolCall]) -> dict[str, Any]:
    """Accepted ``update_trip_details`` results, re-validated against state."""

    changes: dict[str, Any] = {}

    for name, _, result in batch:
        if name == "update_trip_details" and isinstance(result, Mapping) and result.get("status") == "accepted":
            changes.update(validated_trip_updates(state, result.get("updates")))

    return changes


def _allocation_change(
    state: AgentState,
    batch: Sequence[ToolCall],
    log_context: LogContext,
) -> dict[str, float] | None:
    """The newest category budget, if valid and different from the current one.

    The old node trusted the tool message's allocation as-is.
    """

    raw = None

    for name, _, result in batch:
        if name == "update_budget_category" and isinstance(result, Mapping) and "budget_allocation" in result:
            raw = result["budget_allocation"]

    if raw is None:
        return None

    try:
        allocation = normalise_budget_allocation(raw if isinstance(raw, Mapping) else None)
    except ValueError:
        allocation = None

    total = float(state.total_convert_budget or 0.0)

    if allocation is None or abs(sum(allocation.values()) - total) > _ALLOCATION_TOLERANCE:
        _log(log_context, "allocation", "rejected", "allocation.invalid")
        return None

    try:
        current = normalise_budget_allocation({k: v for k, v in state.budget_allocation.items() if k != "flight"})
    except ValueError:
        current = None

    return None if allocation == current else allocation


# ── Incremental candidate ───────────────────────────────────────────


async def _incremental_candidate(
    state: AgentState,
    requested_edits: list[Any],
    allocation: dict[str, float] | None,
    services: EditServices,
    log_context: LogContext,
) -> dict[str, Any]:
    transportation = float((allocation or state.budget_allocation).get("transportation") or 0.0)

    if requested_edits:
        outcome = await _apply(state, requested_edits, transportation, services)

        if not outcome.applied:
            return _edit_refused(state, requested_edits, outcome, services.reviewer, log_context)

        itinerary, affected = outcome.itinerary, outcome.affected_days
    else:
        # A new transportation share can change which flights are over budget.
        itinerary, affected = copy.deepcopy(state.draft_itinerary), ()
        reassess_flight_budget(itinerary, transportation)

    if itinerary == state.draft_itinerary and allocation is None:
        return {"post_tool_outcome": "agent"}

    # Checkpoints may restore day keys as strings; "1" and 1 would collide.
    maps = {_day_key(day): day_map for day, day_map in state.daily_map_info.items()}
    view = state.model_copy(update={"budget_allocation": allocation}) if allocation else state

    try:
        if affected:
            # Only the changed days are routed again (the old node re-routed
            # every day of the trip, one paid directions call per day).
            routed = await asyncio.to_thread(services.routes.build_daily_maps, itinerary, days=affected)
            itinerary = routed.itinerary
            maps.update(routed.daily_maps)

        report = validate_itinerary_candidate(build_trip_requirements(view), itinerary, maps)

    except Exception as exc:
        code = f"planning.incremental.{type(exc).__name__.casefold()}"
        _log(log_context, "incremental_candidate", "failed", code)
        return _rejected(state, EDIT_NOT_APPLICABLE, [code], services.reviewer, log_context)

    if not report.qualified:
        codes = [f"planning.incremental.{issue.code}" for issue in report.issues]
        _log(log_context, "incremental_candidate", "rejected", *codes)
        return _rejected(state, EDIT_FAILED_VALIDATION, codes, services.reviewer, log_context)

    _log(log_context, "incremental_candidate", "validated")

    return {
        "planning_outcome": "validated",
        "candidate_plan": {
            "itinerary": itinerary,
            "maps": maps,
            "attempt": 1,
            "validation": report.model_dump(mode="json"),
        },
        "planning_attempts": 1,
        "planning_issue_codes": [],
        "output_review_attempts": 0,
        "output_review_issue_codes": [],
        "plan_overrides": {"budget_allocation": allocation} if allocation else None,
        "post_tool_outcome": "review_candidate",
    }


async def _apply(
    state: AgentState,
    requested_edits: list[Any],
    transportation: float,
    services: EditServices,
) -> EditOutcome:
    places = trusted_place_options(state.messages)
    bookings = trusted_booking_options(state.messages)
    is_day_authorized = day_authorizer(state)

    def apply(trusted: Sequence[Mapping[str, Any]]) -> EditOutcome:
        return apply_edits(
            state.draft_itinerary,
            requested_edits,
            trusted_places=trusted,
            trusted_bookings=bookings,
            is_day_authorized=is_day_authorized,
            transportation_budget=transportation,
        )

    outcome = apply(places)

    # Tool order is not a trust boundary: if the model skipped search_places,
    # look the place up server-side once, only after everything else passed.
    if outcome.code == "untrusted_place":
        recovered = await asyncio.to_thread(_recover_places, state, requested_edits, places, services.place_search)

        if recovered:
            outcome = apply([*places, *recovered])

    return outcome


def _recover_places(
    state: AgentState,
    requested_edits: list[Any],
    trusted: list[dict[str, Any]],
    search: ActivitySearchService,
) -> list[dict[str, Any]]:
    known = {_name_key(option.get("name")) for option in trusted}
    recovered: list[dict[str, Any]] = []
    lookups = 0

    for edit in requested_edits:
        details = edit.get("new_details") if isinstance(edit, Mapping) else None
        name = _name_key(details.get("name")) if isinstance(details, Mapping) else ""

        if not name or name in known or lookups >= _MAX_PLACE_RECOVERIES:
            continue

        lookups += 1
        known.add(name)
        recovered.extend(recover_place_options(edit, state, search=search))

    return recovered


def _day_key(day: Any) -> Any:
    return int(day) if isinstance(day, str) and day.isascii() and day.isdigit() and day == str(int(day)) else day


def _name_key(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _edit_refused(
    state: AgentState,
    requested_edits: list[Any],
    outcome: EditOutcome,
    reviewer: OutputReviewService,
    log_context: LogContext,
) -> dict[str, Any]:
    _log(log_context, "edit", "rejected", f"edit.{outcome.code or 'refused'}")

    if outcome.code == "day_not_authorized":
        message = server_reply(
            state,
            outcome.error or "",
            marker=DAY_CLARIFICATION_MARKER,
            reviewer=reviewer,
            log_context=log_context,
        )
        return {
            "messages": [message],
            "pending_itinerary_mutation": pending_mutation_record(
                requested_edits,
                source_request=latest_user_request(state.messages),
            ),
            **planning_reset(),
            "post_tool_outcome": "reply_sent",
        }

    return _rejected(state, outcome.error or "", [f"edit.{outcome.code or 'refused'}"], reviewer, log_context)


def _rejected(
    state: AgentState,
    text: str,
    issue_codes: list[str],
    reviewer: OutputReviewService,
    log_context: LogContext,
) -> dict[str, Any]:
    """The edit was not saved; the reply is final and the plan unchanged."""

    return {
        "messages": [
            server_reply(
                state,
                text,
                marker=EDIT_REJECTED,
                reviewer=reviewer,
                log_context=log_context,
                fallback=EDIT_NOT_APPLICABLE,
            )
        ],
        **planning_reset(),
        "planning_issue_codes": issue_codes,
        "pending_itinerary_mutation": None,
        "plan_overrides": None,
        "post_tool_outcome": "reply_sent",
    }


def _log(log_context: LogContext, stage: str, outcome: str, *codes: str) -> None:
    log_event(logger, "planning.post_tool", context=log_context, stage=stage, outcome=outcome, issue_codes=list(codes))


__all__ = [
    "EditServices",
    "post_tool_processing_node",
    "resolve_budget_batch",
]
