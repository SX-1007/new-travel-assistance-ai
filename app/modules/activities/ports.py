"""Contracts required by activity planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.modules.activities.models import IdeaRequest, PlaceIdea


class ActivityIdeaPort(Protocol):
    """Propose day-by-day place ideas (an LLM in production)."""

    def propose(self, request: IdeaRequest) -> list[PlaceIdea]:
        """Ideas for the whole trip. Raise on provider failure."""
        ...

    def propose_rescue(
        self,
        request: IdeaRequest,
        *,
        missing_days: Sequence[int],
        excluded_places: Sequence[str],
    ) -> list[PlaceIdea]:
        """New ideas only for ``missing_days``. Raise on provider failure."""
        ...


__all__ = [
    "ActivityIdeaPort",
]
