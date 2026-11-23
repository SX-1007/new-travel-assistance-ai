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
    # A paid search provider's quota is used up: retrying soon won't help.
    "provider_quota_exhausted",
    # A chat re-plan over the user's plan limit, or while another of the
    # user's plans runs: the reply says when to try again.
    "request_limit_reached",
    # A chat re-plan while the server plans as many trips as it can, or
    # while the limits can't be checked: try again in a minute.
    "planning_busy",
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