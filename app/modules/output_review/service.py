"""Fail-closed review of text that may become a public assistant reply.

Order: deterministic rules first (free, and a provable failure needs no model
call), then the semantic reviewer. A reviewer failure is reported as
``review.unavailable`` so callers can use their server-owned fallback instead
of publishing unreviewed text.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from app.modules.output_review.models import (
    SEMANTIC_ISSUE_CODES,
    OutputReviewContext,
    OutputReviewDecision,
)
from app.modules.output_review.policies import deterministic_output_review
from app.modules.output_review.ports import SemanticReviewerPort
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)


class OutputReviewService:
    """Deterministic + semantic review of one proposed reply."""

    def __init__(
        self,
        *,
        reviewer: SemanticReviewerPort,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._reviewer = reviewer
        self._clock = clock

    def review(
        self,
        context: OutputReviewContext,
        *,
        log_context: LogContext | None = None,
    ) -> OutputReviewDecision:
        started = self._clock()
        decision = deterministic_output_review(context)

        if decision.approved:
            decision = self._semantic_review(context)

        log_event(
            logger,
            "review.decision",
            context=log_context,
            stage=context.stage,
            attempt=context.attempt,
            outcome=(
                "approved"
                if decision.approved
                else "unavailable"
                if decision.unavailable
                else "rejected"
            ),
            issue_codes=list(decision.issue_codes),
            elapsed_ms=(self._clock() - started) * 1000,
        )

        return decision

    def check(self, context: OutputReviewContext) -> OutputReviewDecision:
        """Deterministic rules only: for server-owned text and restored history."""

        return deterministic_output_review(context)

    def _semantic_review(self, context: OutputReviewContext) -> OutputReviewDecision:
        try:
            decision = self._reviewer.review(review_payload(context))
        except Exception:
            return _unavailable()

        if not isinstance(decision, OutputReviewDecision) or any(
            code not in SEMANTIC_ISSUE_CODES for code in decision.issue_codes
        ):
            # A reviewer may not claim infrastructure codes or return junk.
            return _unavailable()

        return decision


def review_payload(context: OutputReviewContext) -> dict[str, Any]:
    """The data the semantic reviewer sees; already compact."""

    payload: dict[str, Any] = {
        "latest_user_request": context.latest_user_request,
        "trusted_requirements": dict(context.trusted_requirements),
        "tool_evidence": list(context.tool_evidence),
        "has_qualified_candidate": context.has_qualified_candidate,
        "proposed_reply": context.proposed_reply,
    }

    if context.candidate_summary is not None:
        payload["candidate_plan"] = dict(context.candidate_summary)

    return payload


def _unavailable() -> OutputReviewDecision:
    return OutputReviewDecision.reject(
        "review.unavailable",
        feedback="Output review is temporarily unavailable.",
    )


__all__ = [
    "OutputReviewService",
    "review_payload",
]
