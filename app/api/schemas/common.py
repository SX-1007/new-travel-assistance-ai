"""Generic API response schemas shared by multiple endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


PlanningUnavailableReason = Literal[
    "validation_failed",
    "provider_data_unavailable",
    "review_unavailable",
    "deadline_exhausted",
]


class PlanningUnavailableResponse(BaseModel):
    """Fail-closed planning result with no rejected plan payload."""

    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal[
        "planning_unavailable"
    ]

    reason: PlanningUnavailableReason

    chat_reply: str

    retryable: Literal[True] = True

    # Always null: the frontend expects these keys on every form response.
    itinerary: None = None
    daily_geojson_maps: None = None


__all__ = [
    "PlanningUnavailableReason",
    "PlanningUnavailableResponse",
]