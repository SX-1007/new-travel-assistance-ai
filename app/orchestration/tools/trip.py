"""Chat tool for changing the trip's core requirements.

The tool validates and echoes the change; post-tool processing applies it to
state and re-plans. ``validated_trip_updates`` runs the same validation again
on the tool payload before anything is applied, because a tool message is
model-visible data, not proof.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.orchestration.state import AgentState


TRIP_DETAIL_FIELDS = (
    "num_people",
    "start_date",
    "end_date",
    "country",
    "city",
    "origin_country",
)


class TripDetailsUpdate(BaseModel):
    """Requested requirement changes; only the fields that are set."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    num_people: int | None = Field(default=None, ge=1, strict=True)
    start_date: str | None = None
    end_date: str | None = None
    country: str | None = Field(default=None, min_length=1)
    city: list[str] | None = None
    origin_country: str | None = Field(default=None, min_length=1)

    @field_validator("start_date", "end_date")
    @classmethod
    def _iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None

        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("dates must use YYYY-MM-DD") from exc

        if parsed.isoformat() != value:
            raise ValueError("dates must use YYYY-MM-DD")

        return value

    @field_validator("city")
    @classmethod
    def _cities(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))

        if not cleaned:
            raise ValueError("city must name at least one place")

        return cleaned


def validated_trip_updates(state: AgentState, raw: Any) -> dict[str, Any]:
    """The changes that are valid against the current trip and differ from it.

    Returns {} when the payload is invalid, the merged dates are out of
    order, or nothing would actually change.
    """

    if not isinstance(raw, Mapping):
        return {}

    try:
        update = TripDetailsUpdate.model_validate(raw)
    except ValidationError:
        return {}

    return _effective_changes(state, update)


@tool("update_trip_details")
def update_trip_details(
    state: Annotated[AgentState, InjectedState],
    num_people: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    country: str | None = None,
    city: list[str] | None = None,
    origin_country: str | None = None,
) -> dict[str, Any]:
    """Change the trip's core requirements: travellers, dates, destination or origin.

    Pass only the fields that changed. The itinerary is then re-planned
    automatically, so do not call edit_itinerary for these changes. Dates use
    YYYY-MM-DD; num_people is at least 1.
    """

    raw = {
        "num_people": num_people,
        "start_date": start_date,
        "end_date": end_date,
        "country": country,
        "city": city,
        "origin_country": origin_country,
    }

    try:
        update = TripDetailsUpdate.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        return {"error": f"Invalid trip details: {problems}."}

    if not update.model_dump(exclude_none=True):
        return {"error": "No trip details were provided to update."}

    changes = _effective_changes(state, update)

    if not changes:
        if _dates_out_of_order(state, update):
            return {"error": "end_date must be on or after start_date."}

        # Re-planning the same trip would repeat every paid search.
        return {
            "status": "unchanged",
            "message": "The trip already has these details; nothing needs re-planning.",
        }

    return {
        "status": "accepted",
        "action": "update_trip_details",
        "updates": changes,
        "message": (
            f"Change accepted ({', '.join(sorted(changes))}). The itinerary will be "
            "re-planned with the new requirements; describe the result only after "
            "the new plan is available."
        ),
    }


def _effective_changes(state: AgentState, update: TripDetailsUpdate) -> dict[str, Any]:
    if _dates_out_of_order(state, update):
        return {}

    requested = update.model_dump(exclude_none=True)

    return {
        field: value
        for field, value in requested.items()
        if _comparable(value) != _comparable(getattr(state, field))
    }


def _comparable(value: Any) -> Any:
    """Place names compare case- and space-insensitively."""

    if isinstance(value, str):
        return " ".join(value.split()).casefold()

    if isinstance(value, list):
        return [_comparable(item) for item in value]

    return value


def _dates_out_of_order(state: AgentState, update: TripDetailsUpdate) -> bool:
    start = update.start_date or state.start_date
    end = update.end_date or state.end_date

    if not start or not end:
        return False

    try:
        return date.fromisoformat(end) < date.fromisoformat(start)
    except ValueError:
        return False


__all__ = [
    "TRIP_DETAIL_FIELDS",
    "TripDetailsUpdate",
    "update_trip_details",
    "validated_trip_updates",
]
