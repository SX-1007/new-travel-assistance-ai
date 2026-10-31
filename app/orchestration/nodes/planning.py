"""Planning nodes: currency setup, the validated transaction, its reply, promotion.

Flow:

    process_initial_form ─┐
    prepare_planning_transaction ─┴→ execute_validated_transaction
        → generate_reviewed_response → promote_candidate (if a candidate)

* ``execute_validated_transaction`` runs the whole bounded retry loop itself
  (``workflows/planning.py``); a failed transaction is final.
* Everything plans against ``transaction_view(state)``, so pending
  requirement and budget changes stay private until ``promote_candidate``
  commits them together with the plan.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.modules.activities.service import ActivityPlanningService
from app.modules.booking.service import BookingPlanningService
from app.modules.budget.allocation import initial_budget_allocation
from app.modules.budget.assessment import parse_assessment
from app.modules.budget.ports import CurrencyProviderPort
from app.modules.budget.service import BudgetAssessmentService
from app.modules.itinerary.history import append_previous_itinerary
from app.modules.itinerary.service import PLANNING_MAX_ATTEMPTS, ItineraryPlanningService
from app.modules.itinerary.validators import validate_itinerary_candidate
from app.modules.output_review.models import OutputReviewDecision
from app.modules.output_review.service import OutputReviewService
from app.modules.routing.service import DailyRouteService
from app.orchestration.nodes.budget import (
    assessment_financials,
    budget_trip_from_state,
)
from app.orchestration.nodes.context import (
    ProfileLoader,
    node_log_context,
    user_id_from_config,
)
from app.orchestration.nodes.replies import (
    EDIT_REJECTED,
    PLANNING_UNAVAILABLE,
    QUALIFIED_FALLBACKS,
    ReplyServices,
    plan_instruction,
    planning_failure_text,
    server_reply,
    write_reply,
)
from app.orchestration.nodes.transaction import (
    OVERRIDABLE_FIELDS,
    PlanKind,
    committed_overrides,
    finished_transaction,
    plan_kind,
    planning_reset,
    transaction_view,
)
from app.orchestration.planning_requirements import build_trip_requirements
from app.orchestration.prompts import build_prompt_messages
from app.orchestration.state import AgentState
from app.orchestration.workflows.currency import run_currency_pipeline
from app.orchestration.workflows.output_review import (
    effective_plan_revision,
    generate_reviewed_reply,
    review_context_from_state,
)
from app.orchestration.workflows.planning import run_planning_transaction
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlanningServices:
    """Everything the transaction and its currency setup call out to."""

    planner: ItineraryPlanningService
    booking: BookingPlanningService
    activities: ActivityPlanningService
    routes: DailyRouteService
    currency: CurrencyProviderPort
    budget: BudgetAssessmentService


# ── Setup ───────────────────────────────────────────────────────────


async def process_initial_form_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    planning: PlanningServices,
) -> dict[str, Any]:
    """Currency and allocation for a new trip or a new destination/origin.

    The preflight assessment's rate is reused when it belongs to this exact
    trip; otherwise the currency provider is asked. For a trip that already
    has a plan the results go into ``plan_overrides``. The transaction is
    cleared here, so the graph can go straight to the transaction.
    """

    log_context = node_log_context(config)
    view = transaction_view(state)

    try:
        financials = await asyncio.to_thread(_trip_financials, view, planning)

    except Exception:
        # A provider failure stops planning (never a 1:1 rate), but as a
        # normal "unavailable" outcome with a reply, not a failed run.
        log_event(
            logger,
            "planning.initial_form",
            context=log_context,
            stage="initial_form",
            outcome="unavailable",
            issue_codes=["currency.unavailable"],
        )
        return {
            **planning_reset(),
            "planning_outcome": "unavailable",
            "planning_attempts": PLANNING_MAX_ATTEMPTS,
            "planning_issue_codes": ["currency.unavailable"],
            "post_tool_outcome": None,
        }

    log_event(logger, "planning.initial_form", context=log_context, stage="initial_form", outcome="completed")

    if effective_plan_revision(state) > 0:
        financial_update: dict[str, Any] = {
            "plan_overrides": {**committed_overrides(state), **financials},
        }
    else:
        financial_update = financials

    return {
        **financial_update,
        **planning_reset(),
        "post_tool_outcome": None,
        # The amount now enters planning; no stale gate state may linger.
        "pending_budget_proposal": None,
        "budget_gate_outcome": None,
        "budget_gate_reason": None,
        "budget_gate_message": None,
    }


def prepare_planning_transaction_node(state: AgentState) -> dict[str, Any]:
    """Checkpoint a cleared transaction before any provider call."""

    del state
    return {**planning_reset(), "post_tool_outcome": None}


def _trip_financials(state: AgentState, planning: PlanningServices) -> dict[str, Any]:
    trip = budget_trip_from_state(state)
    assessment = state.budget_assessment

    if trip is not None and assessment and planning.budget.matches(assessment, trip):
        parsed = parse_assessment(assessment)

        if parsed is not None:
            return assessment_financials(float(state.total_base_budget or 0.0), parsed)

    financials = run_currency_pipeline(state, currency_provider=planning.currency)
    financials["budget_allocation"] = initial_budget_allocation(financials["total_convert_budget"])
    return financials


# ── Transaction ─────────────────────────────────────────────────────


async def execute_validated_transaction_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    planning: PlanningServices,
    profiles: ProfileLoader,
) -> dict[str, Any]:
    """Build a validated private candidate; the public plan is untouched."""

    log_context = node_log_context(config)
    profile = await profiles.load(user_id_from_config(config), log_context=log_context)

    return await run_planning_transaction(
        transaction_view(state),
        planner=planning.planner,
        booking=planning.booking,
        activities=planning.activities,
        routes=planning.routes,
        user_profile=profile,
        log_context=log_context,
    )


# ── Reply ───────────────────────────────────────────────────────────


async def generate_reviewed_response_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    replies: ReplyServices,
    profiles: ProfileLoader,
) -> dict[str, Any]:
    """Write and privately review the reply for a qualified candidate.

    If no draft passes (or the reviewer is down), the fixed fallback for the
    plan kind is used instead of discarding a validated plan. Without a
    candidate the transaction failed: reply with the fixed failure text.
    """

    log_context = node_log_context(config)
    candidate = state.candidate_plan
    kind = plan_kind(state)

    if state.planning_outcome != "validated" or not isinstance(candidate, Mapping):
        return _transaction_failed(state, kind, list(state.planning_issue_codes), replies.reviewer, log_context)

    view = transaction_view(state)
    profile = await profiles.load(user_id_from_config(config), log_context=log_context)
    prompt = [*build_prompt_messages(view, profile), HumanMessage(content=plan_instruction(kind, view))]

    result = await generate_reviewed_reply(
        generate=lambda previous: write_reply(replies.writer, prompt, previous, config),
        build_context=lambda text, attempt: review_context_from_state(
            view,
            text,
            candidate=candidate,
            user_profile=profile,
            attempt=attempt,
        ),
        reviewer=replies.reviewer,
        log_context=log_context,
    )

    attempts = state.output_review_attempts + result.attempts
    issue_codes = [*state.output_review_issue_codes, *result.issue_codes]
    reply, decision, source = result.reply, result.decision, "reviewed"

    if reply is None:
        reply = QUALIFIED_FALLBACKS[kind]
        decision = replies.reviewer.check(
            review_context_from_state(view, reply, candidate=candidate, stage="fallback_review")
        )
        source = "server_fallback"

        if not decision.approved:
            return _transaction_failed(state, kind, list(decision.issue_codes), replies.reviewer, log_context)

    return {
        "candidate_plan": {
            **candidate,
            "reply": reply,
            "review": decision.model_dump(mode="json"),
            "reply_source": source,
        },
        "output_review_attempts": attempts,
        "output_review_issue_codes": issue_codes,
    }


# ── Promotion ───────────────────────────────────────────────────────


async def promote_candidate_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    reviewer: OutputReviewService,
) -> dict[str, Any]:
    """Re-check and atomically publish one candidate with its overrides.

    Fail-closed: the stored review must be an approval, the reply must pass
    the deterministic rules again and the plan must re-validate against the
    requirements it will be published under. No second paid review.
    """

    started = time.perf_counter()
    log_context = node_log_context(config)
    kind = plan_kind(state)
    candidate = state.candidate_plan

    def log(outcome: str, codes: Sequence[str] = ()) -> None:
        log_event(
            logger,
            "planning.promotion",
            context=log_context,
            stage="promotion",
            outcome=outcome,
            attempt=candidate.get("attempt", 0) if isinstance(candidate, Mapping) else 0,
            issue_codes=list(codes),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    fields = _committed_fields(state, candidate) if isinstance(candidate, Mapping) else {}
    view = state.model_copy(update=fields) if fields else state
    issue_codes = _promotion_issues(state, view, candidate, reviewer)

    if issue_codes:
        log("rejected", issue_codes)
        return _transaction_failed(state, kind, issue_codes, reviewer, log_context, stage="promotion")

    assert isinstance(candidate, Mapping)
    revision = effective_plan_revision(state) + 1
    itinerary = candidate["itinerary"]
    maps = candidate["maps"]

    committed: dict[str, Any] = {
        **fields,
        "draft_itinerary": itinerary,
        "daily_map_info": maps,
        "plan_revision": revision,
        "itinerary_history": append_previous_itinerary(
            state.itinerary_history,
            _previous_snapshot(state),
        ),
    }

    log("committed")

    return {
        **committed,
        "accepted_plan_snapshot": {
            "plan_revision": revision,
            "total_base_budget": float(view.total_base_budget or 0.0),
            "base_currency_code": view.base_currency_code,
            "dest_currency_code": view.dest_currency_code,
            "total_convert_budget": float(view.total_convert_budget or 0.0),
            "budget_allocation": dict(view.budget_allocation),
            # Own copies, so the public fields and the snapshot never share
            # nested objects.
            "draft_itinerary": copy.deepcopy(itinerary),
            "daily_map_info": copy.deepcopy(maps),
        },
        "messages": [AIMessage(content=candidate["reply"], additional_kwargs={"plan_revision": revision})],
        **planning_reset(),
        **finished_transaction(),
    }


def _promotion_issues(
    state: AgentState,
    view: AgentState,
    candidate: Any,
    reviewer: OutputReviewService,
) -> list[str]:
    """Why the candidate may not be published; empty when it may."""

    if not isinstance(candidate, Mapping):
        return ["promotion.candidate.missing"]

    reply = candidate.get("reply")

    try:
        prior = OutputReviewDecision.model_validate(candidate.get("review"))
    except Exception:
        return ["promotion.candidate.invalid"]

    if (
        not prior.approved
        or not isinstance(reply, str)
        or not reply.strip()
        or not isinstance(candidate.get("itinerary"), list)
        or not isinstance(candidate.get("maps"), Mapping)
    ):
        return ["promotion.candidate.invalid"]

    check = reviewer.check(
        review_context_from_state(
            view,
            reply,
            candidate=candidate,
            stage="promotion_review",
            attempt=state.output_review_attempts + 1,
        )
    )

    if not check.approved:
        return list(check.issue_codes)

    try:
        requirements = build_trip_requirements(view)
    except ValueError:
        return ["promotion.requirements.invalid"]

    report = validate_itinerary_candidate(requirements, candidate["itinerary"], candidate["maps"])
    return [issue.code for issue in report.issues]


def _committed_fields(state: AgentState, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Overrides to publish; a candidate saved by an earlier version may carry its own."""

    legacy = candidate.get("financials")
    legacy = {key: value for key, value in legacy.items() if key in OVERRIDABLE_FIELDS} if isinstance(legacy, Mapping) else {}
    return {**committed_overrides(state), **legacy}


def _previous_snapshot(state: AgentState) -> Mapping[str, Any] | None:
    if isinstance(state.accepted_plan_snapshot, Mapping):
        return state.accepted_plan_snapshot

    if state.draft_itinerary:
        return {"plan_revision": effective_plan_revision(state), "draft_itinerary": state.draft_itinerary}

    return None


# ── Failure ─────────────────────────────────────────────────────────


def _transaction_failed(
    state: AgentState,
    kind: PlanKind,
    issue_codes: list[str],
    reviewer: OutputReviewService,
    log_context: LogContext,
    *,
    stage: str = "response",
) -> dict[str, Any]:
    """Final failure: fixed reply, overrides dropped, public plan unchanged."""

    marker = EDIT_REJECTED if kind is PlanKind.EDIT else PLANNING_UNAVAILABLE
    message = server_reply(
        state,
        planning_failure_text(state, kind),
        marker=marker,
        reviewer=reviewer,
        log_context=log_context,
    )

    log_event(
        logger,
        "planning.unavailable",
        context=log_context,
        stage=stage,
        outcome="unavailable",
        issue_codes=issue_codes,
        reason=kind.value,
    )

    prefix = "planning.incremental." if kind is PlanKind.EDIT else ""

    return {
        "messages": [message],
        "planning_outcome": "unavailable",
        "candidate_plan": None,
        "planning_issue_codes": [
            *state.planning_issue_codes,
            *(f"{prefix}{code}" for code in issue_codes if code not in state.planning_issue_codes),
        ],
        **finished_transaction(),
    }


__all__ = [
    "PlanningServices",
    "execute_validated_transaction_node",
    "generate_reviewed_response_node",
    "prepare_planning_transaction_node",
    "process_initial_form_node",
    "promote_candidate_node",
]
