"""Contract for the semantic (model-based) output reviewer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from app.modules.output_review.models import OutputReviewDecision


class SemanticReviewerPort(Protocol):
    """Judge one proposed reply against trusted context (an LLM in production)."""

    def review(self, payload: Mapping[str, Any]) -> OutputReviewDecision:
        """Return only semantic issue codes. Raise on provider failure."""
        ...


__all__ = [
    "SemanticReviewerPort",
]
