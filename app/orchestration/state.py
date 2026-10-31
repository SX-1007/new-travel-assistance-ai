"""LangGraph orchestration state for the travel assistant.

This module owns cross-function workflow state only. Functionality modules should
receive the data they need through explicit arguments/models and must not import
``AgentState`` directly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """Server-owned working state carried across LangGraph nodes."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    # ── Conversation history ──────────────────────────────────────────

    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list,
        description=(
            "Conversation history. The add_messages reducer appends new "
            "messages instead of overwriting the existing history."
        ),
    )

    # ── User origin ───────────────────────────────────────────────────

    origin_country: str | None = Field(
        default=None,
        description="User's home country.",
    )

    origin_state: str | None = Field(
        default=None,
        description="User's home state/province.",
    )

    # ── Trip inputs / resolved destination ────────────────────────────

    country: str | None = Field(
        default=None,
        description="Destination country.",
    )

    city: list[str] = Field(
        default_factory=list,
        description=(
            "Verified server-owned destination cities/areas used by planning "
            "functions."
        ),
    )

    num_people: int | None = Field(
        default=None,
        ge=1,
        description="Number of travellers.",
    )

    total_base_budget: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Total trip budget in the user's base currency.",
    )

    start_date: str | None = Field(
        default=None,
        description="Trip start date in YYYY-MM-DD format.",
    )

    end_date: str | None = Field(
        default=None,
        description="Trip end date in YYYY-MM-DD format.",
    )

    # ── Currency conversion ───────────────────────────────────────────

    base_currency_code: str | None = Field(
        default=None,
        description="ISO 4217 code for the user's base currency.",
    )

    dest_currency_code: str | None = Field(
        default=None,
        description="ISO 4217 code for the destination currency.",
    )

    total_convert_budget: float | None = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Total budget expressed in the destination currency.",
    )

    budget_allocation: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-category budget breakdown in destination currency. Canonical "
            "keys: transportation, accommodation, food, activity, shopping, "
            "emergency_fund."
        ),
    )

    exchange_rate: dict[str, float] = Field(
        default_factory=dict,
        description="Exchange-rate mapping keyed by currency code.",
    )

    currency_fetched_at: str | None = Field(
        default=None,
        description="ISO 8601 UTC timestamp of the last exchange-rate fetch.",
    )

    # ── Provider-grounded budget gate ─────────────────────────────────

    budget_assessment: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Provider-grounded prices and exchange-rate evidence verified "
            "before itinerary planning."
        ),
    )

    pending_budget_proposal: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Current-turn budget intent awaiting deterministic assessment."
        ),
    )

    pending_budget_confirmation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Server-owned recommendation awaiting explicit confirmation."
        ),
    )

    pending_itinerary_mutation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Server-owned itinerary edit intent awaiting a missing target "
            "detail. Provider payloads stored here are never trusted as final "
            "evidence."
        ),
    )

    accepted_budget_decision: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Server-created accepted budget decision awaiting authoritative "
            "validation before planning."
        ),
    )

    budget_gate_outcome: Literal[
        "accepted",
        "budget_confirmation_required",
        "budget_check_unavailable",
    ] | None = None

    budget_gate_reason: str | None = None

    budget_gate_message: str | None = None

    chat_budget_action: Literal[
        "accept_recommended"
    ] | None = None

    chat_budget_assessment_id: str | None = None

    # ── Accepted plan / revision control ──────────────────────────────

    plan_revision: int = Field(
        default=0,
        ge=0,
        description=(
            "Monotonic server-owned revision of the committed plan."
        ),
    )

    accepted_plan_snapshot: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Last fully committed financial, itinerary, and map snapshot. "
            "Candidate replans never write into this object."
        ),
    )

    itinerary_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Server-owned copies of previously accepted itinerary revisions. "
            "Historical entries are never treated as the live plan."
        ),
    )

    # ── Current itinerary / map state ─────────────────────────────────

    draft_itinerary: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Current day-by-day itinerary candidate or accepted plan."
        ),
    )

    daily_map_info: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Mapping of day number to public GeoJSON FeatureCollection."
        ),
    )

    # ── Private planning transaction ──────────────────────────────────

    planning_outcome: Literal[
        "validated",
        "unavailable",
    ] | None = None

    candidate_plan: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Private deterministically validated plan awaiting review."
        ),
    )

    planning_attempts: int = Field(
        default=0,
        ge=0,
    )

    planning_issue_codes: list[str] = Field(
        default_factory=list,
    )

    output_review_attempts: int = Field(
        default=0,
        ge=0,
    )

    output_review_issue_codes: list[str] = Field(
        default_factory=list,
    )

    plan_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Private trip-requirement and financial changes the running "
            "planning transaction plans against. They reach the public "
            "fields only together with a promoted candidate, so a failed "
            "replan never leaves requirements that no longer match the plan."
        ),
    )

    post_tool_outcome: Literal[
        "replan_full",
        "replan_itinerary",
        "review_candidate",
        "assess_budget",
        "budget_gate",
        "reply_sent",
        "agent",
    ] | None = Field(
        default=None,
        description=(
            "What post-tool processing decided for the latest tool batch; "
            "read by the graph router."
        ),
    )

    # ── Current-turn factual Q&A ──────────────────────────────────────

    factual_route: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Server-owned factual routing and retrieval policy for the current "
            "turn only."
        ),
    )

    factual_evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Normalized current-turn factual evidence; never reused as live "
            "evidence on a later turn."
        ),
    )

    factual_validation: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Deterministic factual-evidence validation for this turn."
        ),
    )

    factual_issue_codes: list[str] = Field(
        default_factory=list,
    )


__all__ = [
    "AgentState",
]