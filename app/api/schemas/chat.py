"""HTTP transport schemas for chat turns and restored chat history.

The frontend checks every response key strictly, so these shapes must match
what it expects exactly (``extra="forbid"`` everywhere).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from app.api.schemas.budget import BudgetConfirmationResponse
from app.api.schemas.common import PlanningUnavailableReason
from app.modules.itinerary.models import DailyItinerary
from app.modules.routing.models import (
    GeoJSONFeatureCollection,
    canonicalize_public_daily_maps,
)
from app.shared.types import CountryCode

# Matches the graph runtime's id limit, so an over-long id is a 422 here
# instead of reaching the runtime.
MAX_ID_LENGTH = 128


class ChatRequest(BaseModel):
    """One authenticated conversational turn."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_ID_LENGTH,
        description=(
            "Unique session identifier "
            "for conversation continuity."
        ),
    )

    user_message: str = Field(
        ...,
        min_length=1,
        max_length=8_192,
        description="The user's chat message.",
    )

    budget_action: Literal[
        "accept_recommended"
    ] | None = Field(
        default=None,
        description=(
            "Explicit acceptance of the "
            "server-owned budget recommendation."
        ),
    )

    budget_assessment_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ID_LENGTH,
        description=(
            "Opaque identifier for the accepted "
            "server-owned assessment."
        ),
    )

    @model_validator(mode="after")
    def validate_budget_confirmation(
        self,
    ) -> "ChatRequest":
        """Require action and assessment ID together."""

        if (
            (self.budget_action is None)
            != (self.budget_assessment_id is None)
        ):
            raise ValueError(
                "budget_action and "
                "budget_assessment_id "
                "must be provided together"
            )

        return self


class ChatSuccessResponse(BaseModel):
    """A revalidated accepted plan and the turn's reviewed reply."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    chat_reply: str
    draft_itinerary: list[DailyItinerary]
    daily_map_info: dict[int, GeoJSONFeatureCollection] = Field(default_factory=dict)
    destination_country_code: CountryCode
    # True only when this turn published a new plan revision.
    itinerary_modified: bool = False
    total_budget: float = 0.0
    currency: str = ""
    budget_allocation: dict[str, float] = Field(default_factory=dict)
    budget_confirmation: None = None

    @field_validator("daily_map_info", mode="before")
    @classmethod
    def _canonical_map_keys(cls, value: Any) -> dict[int, Any]:
        return canonicalize_public_daily_maps(value)


class _PlanFreeChatResponse(BaseModel):
    """Shared shape of the chat responses that carry no plan."""

    model_config = ConfigDict(extra="forbid")

    chat_reply: str
    draft_itinerary: list[DailyItinerary] = Field(default_factory=list, max_length=0)
    daily_map_info: dict[int, Any] = Field(default_factory=dict, max_length=0)
    itinerary_modified: Literal[False] = False
    total_budget: Literal[0.0] = 0.0
    currency: Literal[""] = ""
    budget_allocation: dict[str, float] = Field(default_factory=dict, max_length=0)


class ChatBudgetGateResponse(_PlanFreeChatResponse):
    status: Literal["budget_confirmation_required", "budget_check_unavailable"]
    budget_confirmation: BudgetConfirmationResponse | None = None


class ChatPlanningUnavailableResponse(_PlanFreeChatResponse):
    status: Literal["planning_unavailable"] = "planning_unavailable"
    reason: PlanningUnavailableReason
    retryable: Literal[True] = True
    budget_confirmation: None = None


ChatResponse = Annotated[
    ChatSuccessResponse | ChatBudgetGateResponse | ChatPlanningUnavailableResponse,
    Field(discriminator="status"),
]


class ChatHistoryBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float
    currency: str
    allocation: dict[str, float]


class ChatHistoryMessage(BaseModel):
    """One restored message; plan fields only on the message that produced
    the session's current plan."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "ai"]
    content: str
    itinerary: list[DailyItinerary] | None = None
    maps: dict[int, GeoJSONFeatureCollection] | None = None
    budget: ChatHistoryBudget | None = None
    destination_country_code: CountryCode | None = None
    budget_confirmation: BudgetConfirmationResponse | None = None

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: Any) -> dict[str, Any]:
        # The frontend treats a present plan key as a plan to render, so an
        # absent field is left out rather than sent as null.
        return {key: value for key, value in handler(self).items() if value is not None}


class ChatHistorySession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    destination: str = ""
    updated_at: str
    messages: list[ChatHistoryMessage] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    sessions: list[ChatHistorySession] = Field(default_factory=list)


__all__ = [
    "MAX_ID_LENGTH",
    "ChatBudgetGateResponse",
    "ChatHistoryBudget",
    "ChatHistoryMessage",
    "ChatHistoryResponse",
    "ChatHistorySession",
    "ChatPlanningUnavailableResponse",
    "ChatRequest",
    "ChatResponse",
    "ChatSuccessResponse",
]