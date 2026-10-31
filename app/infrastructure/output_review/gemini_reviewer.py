"""Gemini-backed semantic output reviewer (no tools, structured output)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.infrastructure.llm.chat import ChatBackend, build_chat_model
from app.modules.output_review.models import OutputReviewDecision, SemanticIssueCode
from app.modules.output_review.prompts import REVIEWER_SYSTEM_PROMPT


_DEFAULT_FEEDBACK = "Rewrite the reply so it is fully supported by the trusted context."


class _ReviewerOutput(BaseModel):
    """What the model may return; normalised before it is trusted."""

    approved: bool
    issue_codes: list[SemanticIssueCode] = Field(default_factory=list)
    feedback: str = ""


class GeminiOutputReviewer:
    """Ask Gemini to judge a reply; it never sees tools or writes replies."""

    def __init__(self, *, backends: Sequence[ChatBackend]) -> None:
        llm = build_chat_model(
            backends,
            temperature=0.0,
            # The caller falls back on failure; one retry is enough.
            max_retries=1,
            thinking_budget=0,
        )
        self._llm = llm.with_structured_output(_ReviewerOutput)

    def review(self, payload: Mapping[str, Any]) -> OutputReviewDecision:
        raw = self._llm.invoke(
            [
                SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
            ]
        )

        if raw is None:
            raise RuntimeError("reviewer returned no structured decision")

        return to_decision(_ReviewerOutput.model_validate(raw))


def to_decision(raw: _ReviewerOutput) -> OutputReviewDecision:
    """Normalise model output, failing closed on contradictions.

    "Approved" with issue codes is treated as a rejection; a rejection with
    no code is incoherent and raises (the caller reports it as unavailable).
    """

    codes = tuple(dict.fromkeys(raw.issue_codes))
    feedback = raw.feedback.strip() or _DEFAULT_FEEDBACK

    if raw.approved and not codes:
        return OutputReviewDecision.approve()

    if not codes:
        raise ValueError("reviewer rejected without an issue code")

    return OutputReviewDecision.reject(*codes, feedback=feedback[:500])


__all__ = [
    "GeminiOutputReviewer",
    "to_decision",
]
