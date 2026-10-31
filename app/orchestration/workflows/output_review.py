"""AgentState adapter for output review, and the bounded rewrite loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from app.modules.itinerary.projection import compact_itinerary
from app.modules.output_review.models import OutputReviewContext, OutputReviewDecision
from app.modules.output_review.policies import is_read_only_itinerary_history_request
from app.modules.output_review.service import OutputReviewService
from app.orchestration.messages import (
    compact_payload,
    current_turn_tool_results,
    latest_user_request,
)
from app.orchestration.state import AgentState
from app.shared.observability import LogContext


OUTPUT_REVIEW_MAX_ATTEMPTS: Final = 3

# Accepted-budget replans carry their own financials on the candidate; the
# review must judge the reply against those, not the old plan's.
CANDIDATE_FINANCIAL_FIELDS: Final = frozenset(
    {
        "base_currency_code",
        "dest_currency_code",
        "exchange_rate",
        "total_convert_budget",
        "budget_allocation",
        "currency_fetched_at",
        "total_base_budget",
    }
)

_MAX_TOOL_RESULT_CHARS: Final = 6_000


@dataclass(frozen=True, slots=True)
class ReviewedReply:
    """Outcome of the rewrite loop; ``reply`` is None when nothing passed."""

    reply: str | None
    decision: OutputReviewDecision
    attempts: int
    issue_codes: tuple[str, ...]


def review_context_from_state(
    state: AgentState,
    reply: str,
    *,
    candidate: Mapping[str, Any] | None = None,
    user_profile: Mapping[str, Any] | None = None,
    stage: str = "response_review",
    attempt: int = 0,
) -> OutputReviewContext:
    """Everything the reviewer may rely on, already compact."""

    request = latest_user_request(state.messages)
    financials = candidate.get("financials") if isinstance(candidate, Mapping) else None

    return OutputReviewContext(
        latest_user_request=request,
        proposed_reply=reply,
        trusted_requirements=trusted_requirements(
            state,
            user_profile=user_profile,
            include_history=is_read_only_itinerary_history_request(request),
            overrides=financials if isinstance(financials, Mapping) else None,
        ),
        tool_evidence=tool_evidence(state),
        candidate_summary=(
            {"itinerary": compact_itinerary(candidate.get("itinerary"))}
            if isinstance(candidate, Mapping)
            else None
        ),
        stage=stage,
        attempt=attempt,
    )


def trusted_requirements(
    state: AgentState,
    *,
    user_profile: Mapping[str, Any] | None = None,
    include_history: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Server-owned trip facts; older revisions only for history questions."""

    revision = effective_plan_revision(state)

    requirements: dict[str, Any] = {
        "origin_country": state.origin_country,
        "country": state.country,
        "city": list(state.city),
        "num_people": state.num_people,
        "start_date": state.start_date,
        "end_date": state.end_date,
        "base_currency_code": state.base_currency_code,
        "dest_currency_code": state.dest_currency_code,
        "total_base_budget": state.total_base_budget,
        "total_convert_budget": state.total_convert_budget,
        "budget_allocation": dict(state.budget_allocation),
        "long_term_user_preferences": dict(user_profile or {}),
        "current_plan_revision": revision,
        "current_itinerary": compact_itinerary(current_itinerary(state)),
    }

    for key, value in (overrides or {}).items():
        if key in CANDIDATE_FINANCIAL_FIELDS and key in requirements:
            requirements[key] = value

    if include_history:
        requirements["itinerary_history"] = [
            {
                "revision": entry.get("plan_revision", entry.get("revision")),
                "itinerary": compact_itinerary(entry.get("draft_itinerary")),
            }
            for entry in state.itinerary_history
            if isinstance(entry, Mapping)
        ]

    return requirements


def effective_plan_revision(state: AgentState) -> int:
    """Plans saved without a revision (by earlier versions) count as revision 1."""

    if state.plan_revision > 0:
        return state.plan_revision

    return 1 if state.draft_itinerary or state.daily_map_info else 0


def current_itinerary(state: AgentState) -> list[dict[str, Any]]:
    """The live itinerary: the accepted snapshot when it is this revision."""

    revision = effective_plan_revision(state)
    snapshot = state.accepted_plan_snapshot if isinstance(state.accepted_plan_snapshot, dict) else {}
    snapshot_revision = snapshot.get("plan_revision")
    snapshot_itinerary = snapshot.get("draft_itinerary")

    if (
        snapshot_revision == revision
        and not isinstance(snapshot_revision, bool)
        and isinstance(snapshot_itinerary, list)
    ):
        return snapshot_itinerary

    return state.draft_itinerary


def tool_evidence(state: AgentState) -> list[dict[str, Any]]:
    """This turn's tool results, without images and map geometry."""

    return [
        {"name": name, "result": compact_payload(result, max_chars=_MAX_TOOL_RESULT_CHARS)}
        for name, result in current_turn_tool_results(state.messages)
    ]


def rewrite_instruction(decision: OutputReviewDecision) -> str:
    """Private instruction appended before regenerating a rejected reply."""

    return (
        "Private output review rejected the prior draft with codes "
        f"{', '.join(decision.issue_codes)}. Feedback: {decision.feedback} "
        "Write a new answer as concise public prose. Do not repeat or mention "
        "the rejected draft or this review instruction."
    )


async def generate_reviewed_reply(
    *,
    generate: Callable[[OutputReviewDecision | None], Awaitable[str]],
    build_context: Callable[[str, int], OutputReviewContext],
    reviewer: OutputReviewService,
    max_attempts: int = OUTPUT_REVIEW_MAX_ATTEMPTS,
    initial_reply: str | None = None,
    log_context: LogContext | None = None,
) -> ReviewedReply:
    """Generate and privately review up to ``max_attempts`` replies.

    ``generate`` receives the previous rejection (None on the first call) and
    returns the reply text. The loop stops on the first approval, and at once
    when generation or review infrastructure fails: those can't be fixed by
    rewriting, and the caller has a server-owned fallback.
    """

    if not 1 <= max_attempts <= OUTPUT_REVIEW_MAX_ATTEMPTS:
        raise ValueError("max_attempts is outside the output-review budget")

    reply = initial_reply
    previous: OutputReviewDecision | None = None
    codes: list[str] = []

    for attempt in range(1, max_attempts + 1):
        if reply is None:
            try:
                reply = await generate(previous)
            except Exception:
                decision = OutputReviewDecision.reject(
                    "generation.unavailable",
                    feedback="Reply generation is temporarily unavailable.",
                )
                codes.extend(decision.issue_codes)
                return ReviewedReply(None, decision, attempt, tuple(codes))

        text = (reply or "").strip()
        decision = await asyncio.to_thread(
            reviewer.review,
            build_context(text, attempt),
            log_context=log_context,
        )
        codes.extend(decision.issue_codes)

        if decision.approved:
            return ReviewedReply(text, decision, attempt, tuple(codes))

        if decision.unavailable:
            return ReviewedReply(None, decision, attempt, tuple(codes))

        previous = decision
        reply = None

    assert previous is not None
    return ReviewedReply(None, previous, max_attempts, tuple(codes))


__all__ = [
    "CANDIDATE_FINANCIAL_FIELDS",
    "OUTPUT_REVIEW_MAX_ATTEMPTS",
    "current_itinerary",
    "ReviewedReply",
    "effective_plan_revision",
    "generate_reviewed_reply",
    "review_context_from_state",
    "rewrite_instruction",
    "tool_evidence",
    "trusted_requirements",
]
