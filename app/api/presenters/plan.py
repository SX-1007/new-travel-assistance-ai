"""The public view of a session's accepted plan, and turn-outcome helpers.

Only ``accepted_plan_snapshot`` is ever shown, and only after it passes the
itinerary validator again against the session's own requirements: a
snapshot written by older code, or one whose financials no longer match the
state, is never rendered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.api.schemas.common import PlanningUnavailableReason
from app.modules.itinerary.models import DailyItinerary
from app.modules.itinerary.validators import validate_itinerary_candidate
from app.modules.routing.models import GeoJSONFeatureCollection, canonicalize_daily_map_keys
from app.orchestration.planning_requirements import build_trip_requirements
from app.orchestration.state import AgentState
from app.shared.text import content_text
from app.shared.types import CountryCode


PLANNING_UNAVAILABLE_MESSAGE: Final = "Planning is temporarily unavailable. Please retry."

_SNAPSHOT_FIELDS: Final = frozenset(
    {
        "plan_revision",
        "total_base_budget",
        "base_currency_code",
        "dest_currency_code",
        "total_convert_budget",
        "budget_allocation",
        "draft_itinerary",
        "daily_map_info",
    }
)


class AcceptedPlan(BaseModel):
    """One revalidated accepted snapshot, ready for a response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_revision: int = Field(ge=1)
    total_base_budget: float = Field(ge=0, allow_inf_nan=False)
    base_currency_code: str
    dest_currency_code: str
    destination_country_code: CountryCode
    total_convert_budget: float = Field(ge=0, allow_inf_nan=False)
    budget_allocation: dict[str, float]
    draft_itinerary: list[DailyItinerary]
    daily_map_info: dict[int, GeoJSONFeatureCollection]


def accepted_plan(state: Mapping[str, Any]) -> AcceptedPlan | None:
    """The current accepted plan if it still validates, else None."""

    raw = state.get("accepted_plan_snapshot")

    if not isinstance(raw, Mapping) or set(raw) != _SNAPSHOT_FIELDS:
        return None

    revision = raw["plan_revision"]

    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0 or revision != state.get("plan_revision"):
        return None

    try:
        maps = canonicalize_daily_map_keys(raw["daily_map_info"])
        requirements = build_trip_requirements(AgentState.model_validate(dict(state)))

        if (
            raw["base_currency_code"] != requirements.base_currency
            or raw["dest_currency_code"] != requirements.destination_currency
            or raw["total_convert_budget"] != requirements.total_budget
            or raw["budget_allocation"] != dict(requirements.budget_allocation)
            or raw["total_base_budget"] != state.get("total_base_budget")
        ):
            return None

        # Last: the validator is the expensive check.
        if not validate_itinerary_candidate(requirements, raw["draft_itinerary"], maps).qualified:
            return None

        return AcceptedPlan(
            plan_revision=revision,
            total_base_budget=raw["total_base_budget"],
            base_currency_code=raw["base_currency_code"],
            dest_currency_code=raw["dest_currency_code"],
            destination_country_code=requirements.destination_country_code,
            total_convert_budget=raw["total_convert_budget"],
            budget_allocation=raw["budget_allocation"],
            draft_itinerary=raw["draft_itinerary"],
            daily_map_info=maps,
        )

    except (TypeError, ValueError, ValidationError):
        return None


def planning_unavailable_reason(state: Mapping[str, Any]) -> PlanningUnavailableReason:
    """Collapse private issue codes into one bounded public reason."""

    codes = [
        code.lower()
        for code in (*(state.get("planning_issue_codes") or ()), *(state.get("output_review_issue_codes") or ()))
        if isinstance(code, str)
    ]

    if any("deadline" in code or "timeout" in code for code in codes):
        return "deadline_exhausted"

    if any("provider" in code or "data.unavailable" in code or code.startswith("currency.") for code in codes):
        return "provider_data_unavailable"

    if any(code.startswith("review.") for code in codes):
        return "review_unavailable"

    return "validation_failed"


def turn_reply(messages: Sequence[Any]) -> str | None:
    """This turn's final reply: the last message, if it is a non-empty AI
    message that follows a human one."""

    if not messages or getattr(messages[-1], "type", "") != "ai":
        return None

    if not any(getattr(message, "type", "") == "human" for message in messages[:-1]):
        return None

    return content_text(messages[-1].content).strip() or None


def promoted_reply(messages: Sequence[Any], plan_revision: int) -> str | None:
    """The reply promotion tagged with ``plan_revision``, if it is the last
    message (only promotion writes that tag, after its own review)."""

    if not messages:
        return None

    latest = messages[-1]
    metadata = getattr(latest, "additional_kwargs", None)

    if (
        getattr(latest, "type", "") != "ai"
        or not isinstance(metadata, Mapping)
        or metadata.get("plan_revision") != plan_revision
    ):
        return None

    return content_text(latest.content).strip() or None


__all__ = [
    "PLANNING_UNAVAILABLE_MESSAGE",
    "AcceptedPlan",
    "accepted_plan",
    "planning_unavailable_reason",
    "promoted_reply",
    "turn_reply",
]
