"""Contracts required by factual Q&A. Every method raises on provider failure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from app.modules.factual_qa.models import (
    GroundingReview,
    RequestClassification,
    SearchHit,
)


class RequestClassifierPort(Protocol):
    """Classify one user message (an LLM in production)."""

    def classify(self, message: str, trip: Mapping[str, Any]) -> RequestClassification:
        ...


class WebSearchPort(Protocol):
    """General web search returning raw hits."""

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        ...


class AnswerWriterPort(Protocol):
    """Write a concise answer from the supplied payload only."""

    def write(self, payload: Mapping[str, Any]) -> str:
        ...


class GroundingReviewerPort(Protocol):
    """Check every claim in an answer against the supplied evidence."""

    def review(self, payload: Mapping[str, Any]) -> GroundingReview:
        ...


__all__ = [
    "AnswerWriterPort",
    "GroundingReviewerPort",
    "RequestClassifierPort",
    "WebSearchPort",
]
