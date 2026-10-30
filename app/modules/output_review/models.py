"""Output-review contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Codes a semantic reviewer may return.
SemanticIssueCode = Literal[
    "reply.unrelated",
    "reply.requirements_missing",
    "reply.destination_mismatch",
    "reply.unsupported_claim",
    "reply.raw_internal_data",
]

# Infrastructure failures: the reply was never judged. Callers use their
# server-owned fallback instead of retrying.
UnavailableIssueCode = Literal[
    "review.unavailable",
    "generation.unavailable",
]

ReviewIssueCode = Literal[SemanticIssueCode, UnavailableIssueCode]

SEMANTIC_ISSUE_CODES: frozenset[str] = frozenset(get_args(SemanticIssueCode))
UNAVAILABLE_ISSUE_CODES: frozenset[str] = frozenset(get_args(UnavailableIssueCode))


class OutputReviewDecision(BaseModel):
    """A coherent approve/reject decision for one proposed reply."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approved: bool = Field(strict=True)
    issue_codes: tuple[ReviewIssueCode, ...] = ()
    feedback: str = ""

    @model_validator(mode="after")
    def _decision_is_coherent(self) -> OutputReviewDecision:
        if self.approved and self.issue_codes:
            raise ValueError("approved output cannot contain issue codes")
        if not self.approved and not self.issue_codes:
            raise ValueError("rejected output must contain an issue code")
        if not self.approved and not self.feedback.strip():
            raise ValueError("rejected output must include rewrite feedback")
        return self

    @classmethod
    def approve(cls) -> OutputReviewDecision:
        return cls(approved=True)

    @classmethod
    def reject(cls, *codes: ReviewIssueCode, feedback: str) -> OutputReviewDecision:
        return cls(approved=False, issue_codes=codes, feedback=feedback)

    @property
    def unavailable(self) -> bool:
        """True when infrastructure failed, not the reply."""

        return any(code in UNAVAILABLE_ISSUE_CODES for code in self.issue_codes)


@dataclass(frozen=True, slots=True)
class OutputReviewContext:
    """Trusted, already-compact inputs for judging one proposed reply.

    The orchestration layer builds this from state; nothing here is raw
    provider or checkpoint data.
    """

    latest_user_request: str
    proposed_reply: str
    trusted_requirements: Mapping[str, Any] = field(default_factory=dict)
    tool_evidence: Sequence[Mapping[str, Any]] = ()
    # Compact qualified plan; None when no plan passed validation this turn.
    candidate_summary: Mapping[str, Any] | None = None
    stage: str = "response_review"
    attempt: int = 0

    @property
    def has_qualified_candidate(self) -> bool:
        return self.candidate_summary is not None


__all__ = [
    "SEMANTIC_ISSUE_CODES",
    "UNAVAILABLE_ISSUE_CODES",
    "OutputReviewContext",
    "OutputReviewDecision",
    "ReviewIssueCode",
    "SemanticIssueCode",
    "UnavailableIssueCode",
]
