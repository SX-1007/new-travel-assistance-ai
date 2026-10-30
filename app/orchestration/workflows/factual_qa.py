"""AgentState adapter for factual Q&A: node bodies and routing decisions.

Factual evidence is turn-scoped: a later turn never reuses an earlier
turn's live facts (the graph resets these fields at the start of a turn).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import ValidationError

from app.modules.factual_qa.models import (
    EvidenceItem,
    EvidenceValidation,
    FactualRoute,
    TripContext,
)
from app.modules.factual_qa.policies import next_after_retrieval, next_after_routing
from app.modules.factual_qa.service import FactualQAService
from app.modules.itinerary.projection import compact_itinerary
from app.orchestration.messages import latest_user_request
from app.orchestration.state import AgentState
from app.orchestration.workflows.output_review import current_itinerary, effective_plan_revision
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)

_ROUTE_INVALID_REPLY = (
    "I couldn't safely determine how to verify that factual request, so I won't guess."
)


def empty_factual_state() -> dict[str, Any]:
    """Factual fields reset at the start of every chat turn."""

    return {
        "factual_route": None,
        "factual_evidence": [],
        "factual_validation": None,
        "factual_issue_codes": [],
    }


def trip_context_from_state(state: AgentState) -> TripContext:
    return TripContext(
        origin_country=state.origin_country,
        destination_country=state.country,
        cities=tuple(state.city),
        start_date=state.start_date,
        end_date=state.end_date,
        num_people=state.num_people,
        plan_revision=effective_plan_revision(state),
        itinerary=tuple(compact_itinerary(current_itinerary(state))),
    )


async def run_factual_routing(
    state: AgentState,
    *,
    service: FactualQAService,
    log_context: LogContext | None = None,
) -> dict[str, Any]:
    message = latest_user_request(state.messages)

    if not message:
        return empty_factual_state()

    route = await asyncio.to_thread(service.route, message, trip_context_from_state(state))

    log_event(
        logger,
        "factual.route",
        context=log_context,
        stage="factual_router",
        outcome=route.request_type.value,
        reason=route.classified_by,
    )

    return {**empty_factual_state(), "factual_route": route.model_dump(mode="json")}


async def run_factual_retrieval(
    state: AgentState,
    *,
    service: FactualQAService,
    log_context: LogContext | None = None,
) -> dict[str, Any]:
    route = _route(state)

    if route is None:
        invalid = EvidenceValidation(approved=False, issue_codes=["fact.route_invalid"])
        return {
            "factual_evidence": [],
            "factual_validation": invalid.model_dump(mode="json"),
            "factual_issue_codes": list(invalid.issue_codes),
        }

    result = await asyncio.to_thread(service.retrieve, route, trip_context_from_state(state))

    log_event(
        logger,
        "factual.retrieval",
        context=log_context,
        stage="factual_retrieval",
        outcome="approved" if result.validation.approved else "rejected",
        issue_codes=result.validation.issue_codes,
    )

    return {
        "factual_evidence": [item.model_dump(mode="json") for item in result.evidence],
        "factual_validation": result.validation.model_dump(mode="json"),
        "factual_issue_codes": list(result.validation.issue_codes),
    }


async def run_factual_answer(
    state: AgentState,
    *,
    service: FactualQAService,
    log_context: LogContext | None = None,
) -> dict[str, Any]:
    route = _route(state)

    if route is None:
        return {
            "messages": [
                AIMessage(
                    content=_ROUTE_INVALID_REPLY,
                    additional_kwargs={"server_owned": "factual_qa_unavailable"},
                )
            ],
            "factual_issue_codes": ["fact.route_invalid"],
        }

    answer = await asyncio.to_thread(
        service.answer,
        route,
        trip_context_from_state(state),
        _evidence(state),
        _validation(state, route),
    )

    log_event(
        logger,
        "factual.answer",
        context=log_context,
        stage="factual_answer",
        outcome="approved" if answer.grounded else "safe_fallback",
        issue_codes=answer.issue_codes,
    )

    return {
        "messages": [
            AIMessage(content=answer.text, additional_kwargs={"server_owned": "factual_qa"})
        ],
        "factual_issue_codes": list(answer.issue_codes),
    }


def mixed_intent_message(state: AgentState, *, service: FactualQAService) -> SystemMessage | None:
    """Verified facts for the general agent on a mixed request, if any."""

    route = _route(state)

    if route is None or state.factual_validation is None:
        return None

    brief = service.mixed_intent_brief(route, _evidence(state), _validation(state, route))
    return SystemMessage(content=brief) if brief else None


def route_after_factual_router(state: AgentState) -> Literal["retrieval", "answer", "agent"]:
    route = _route(state)
    return "agent" if route is None else next_after_routing(route)


def route_after_factual_retrieval(state: AgentState) -> Literal["answer", "agent"]:
    route = _route(state)

    if route is None or state.factual_validation is None:
        return "answer"

    return next_after_retrieval(route, _validation(state, route))


# ── Internals ───────────────────────────────────────────────────────


def _route(state: AgentState) -> FactualRoute | None:
    if not state.factual_route:
        return None

    try:
        return FactualRoute.model_validate(state.factual_route)
    except ValidationError:
        return None


def _evidence(state: AgentState) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []

    for item in state.factual_evidence:
        try:
            evidence.append(EvidenceItem.model_validate(item))
        except ValidationError:
            continue

    return evidence


def _validation(state: AgentState, route: FactualRoute) -> EvidenceValidation:
    if state.factual_validation is None:
        # Never retrieved: required evidence is missing, not approved.
        return EvidenceValidation(
            approved=not route.requires_external_evidence,
            issue_codes=[] if not route.requires_external_evidence else ["fact.evidence_missing"],
        )

    try:
        return EvidenceValidation.model_validate(state.factual_validation)
    except ValidationError:
        return EvidenceValidation(approved=False, issue_codes=["fact.validation_invalid"])


__all__ = [
    "empty_factual_state",
    "mixed_intent_message",
    "route_after_factual_retrieval",
    "route_after_factual_router",
    "run_factual_answer",
    "run_factual_retrieval",
    "run_factual_routing",
    "trip_context_from_state",
]
