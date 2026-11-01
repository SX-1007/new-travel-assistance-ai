"""Gemini-backed activity idea generation (brainstorm + rescue).

The structured-output schema is deliberately lenient and each idea is
validated on its own: with a strict schema, one malformed idea (a negative
cost, day 0) would fail the whole response and discard every good idea.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app.infrastructure.llm.chat import ChatBackend, build_chat_model
from app.modules.activities.models import IdeaRequest, PlaceIdea
from app.modules.activities.policies import MIN_PLACES_PER_DAY
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


_BRAINSTORM_SYSTEM = """\
You are an expert local travel planner. Propose a realistic, day-by-day list of \
ATTRACTIONS and RESTAURANTS for the trip described below.

Hard rules:
1. EVERY place MUST be physically located in {country}, in or around the trip's \
listed cities/areas. NEVER propose a place in any other country or region, even \
if it shares a name or theme with the destination (for a China trip, never a \
'Chinatown' in the US). Set each place's `city` to exactly one listed city/area.
2. Use REAL, well-known, searchable place names exactly as they appear on Google Maps.
3. Assign each place to a day (1..{num_days}) and an `order` within that day.
4. The `order` must be the most logical route for the day: group nearby places \
and progress in one direction. Do not zig-zag.
5. Each day starts from the hotel. Respect each day's place limit in the day \
list below: it reflects real flight times. A day marked "travel day" gets NO \
places at all (the traveller is in transit); a limited day only fits that many.
6. Pacing: 'relaxed' => 3 places/day; 'moderate' => 4-6; 'packed' => 6-8, \
but never more than the day's limit or {max_per_day}. Every day that is not a \
travel day MUST get at least its minimum from the day list (3 on a full day).
7. Include 1-2 restaurants per day (lunch/dinner) that respect dietary needs.
8. Estimate each place's TOTAL cost for the whole party ({num_people} people) in \
the DESTINATION currency. Keep the SUM of attraction costs <= {activity_budget} \
and the SUM of restaurant costs <= {food_budget}. Free attractions cost 0.
9. Personalise to the user's interests. Prefer variety over repetition.
"""

# The trip context MUST be a human message: langchain-google-genai maps the
# system message to Gemini's system_instruction, and a request with no
# contents is rejected ("contents is not specified").
_BRAINSTORM_HUMAN = """\
Plan the places for this trip:
- Destination country: {country}
- Cities/areas: {cities}
- Number of days: {num_days}
- Days (date and place limit): {dates}
- Party size: {num_people}
- Base hotel: {hotel_name}
- User interests: {interests}
- Dietary restrictions: {dietary}
- Travel pacing: {pacing}
- Activity budget (attractions, destination currency): {activity_budget}
- Food budget (restaurants, destination currency): {food_budget}
"""

_RESCUE_SYSTEM = """\
You are repairing itinerary days that have too few places. Generate NEW real, \
searchable ATTRACTIONS or RESTAURANTS only for the requested day numbers; they \
are added to the places those days already have.

Hard rules:
1. Return places ONLY for days in: {missing_days}, and at least the day's \
minimum from the day list for each of them (3 on a full day).
2. Every place must be a real Google-Maps-searchable venue in {country}.
3. Set `city` to exactly one value from: {cities}.
4. Prefer well-known places with unambiguous names so verification succeeds.
5. Never use an excluded place: {excluded_places}.
6. Use a logical order and never exceed a day's place limit (at most {max_per_day}).
7. Keep attraction estimates within {activity_budget} and restaurant estimates \
within {food_budget} (destination currency, whole party).
"""

_RESCUE_HUMAN = """\
Add places to these itinerary days:
- Destination country: {country}
- Cities/areas: {cities}
- Day numbers needing places: {missing_days}
- Days (date and place limit): {dates}
- Party size: {num_people}
- Hotel: {hotel_name}
- Interests: {interests}
- Dietary restrictions: {dietary}
- Travel pacing: {pacing}
"""


class _RawIdea(BaseModel):
    """Lenient LLM schema; ``PlaceIdea`` enforces the real rules."""

    name: str = Field(description="Real, searchable place name as on Google Maps.")
    place_type: str = Field(description="'attraction' or 'restaurant'.")
    city: str = Field(default="", description="One of the trip's cities/areas.")
    day: int = Field(description="Trip day number, 1-based.")
    order: int = Field(description="Visiting order within the day, 1 = first.")
    suggested_time: str = Field(
        default="",
        description="Coarse time hint: Morning, Lunch, Afternoon, Evening.",
    )
    estimated_cost: float = Field(
        default=0.0,
        description="TOTAL cost for the whole party, destination currency.",
    )
    reason: str = Field(default="", description="One short sentence: why this place.")


class _RawPlan(BaseModel):
    places: list[_RawIdea] = Field(default_factory=list)


def _to_ideas(plan: Any) -> list[PlaceIdea]:
    raw_ideas = getattr(plan, "places", None) or []
    ideas: list[PlaceIdea] = []
    rejected = 0

    for raw in raw_ideas:
        try:
            ideas.append(PlaceIdea.model_validate(raw.model_dump()))

        except (ValidationError, AttributeError):
            rejected += 1

    if rejected:
        safe_log(logger, logging.INFO, "provider.activity_ideas.invalid_dropped")

    return ideas


def _listed(values: Sequence[str], default: str) -> str:
    return ", ".join(values) if values else default


def _common_inputs(request: IdeaRequest) -> dict[str, Any]:
    preferences = request.preferences

    return {
        "country": request.country,
        "cities": _listed(request.cities, request.country),
        "num_days": len(request.dates),
        "num_people": request.num_people,
        "hotel_name": request.hotel_name or "the hotel",
        "interests": _listed(
            preferences.interests,
            "general sightseeing, local culture, good food",
        ),
        "dietary": _listed(preferences.dietary, "none"),
        "pacing": preferences.pacing,
        "activity_budget": round(request.activity_budget, 2),
        "food_budget": round(request.food_budget, 2),
        "max_per_day": request.max_per_day,
    }


def _day_list(request: IdeaRequest) -> str:
    """``Day 1 (2026-12-10): travel day, no places; Day 2 (...): 3 to 8 places``."""

    limits = request.day_limits or (request.max_per_day,) * len(request.dates)
    entries = []

    for index, (day_date, limit) in enumerate(zip(request.dates, limits), start=1):
        most = min(limit, request.max_per_day)
        least = min(MIN_PLACES_PER_DAY, most)

        if limit <= 0:
            allowance = "travel day, no places"
        elif least == most:
            allowance = f"exactly {most} place{'s' if most > 1 else ''}"
        else:
            allowance = f"{least} to {most} places"
        entries.append(f"Day {index} ({day_date or 'date unknown'}): {allowance}")

    return "; ".join(entries) or "unknown"


class GeminiActivityIdeaGenerator:
    """Ask Gemini for place ideas; verification happens elsewhere."""

    def __init__(
        self,
        *,
        backends: Sequence[ChatBackend],
        temperature: float = 0.4,
    ) -> None:
        llm = build_chat_model(
            backends,
            temperature=temperature,
            max_retries=2,
        ).with_structured_output(_RawPlan)

        self._brainstorm = (
            ChatPromptTemplate.from_messages(
                [("system", _BRAINSTORM_SYSTEM), ("human", _BRAINSTORM_HUMAN)]
            )
            | llm
        )
        self._rescue = (
            ChatPromptTemplate.from_messages(
                [("system", _RESCUE_SYSTEM), ("human", _RESCUE_HUMAN)]
            )
            | llm
        )

    def propose(self, request: IdeaRequest) -> list[PlaceIdea]:
        inputs = _common_inputs(request)
        inputs["dates"] = _day_list(request)

        return _to_ideas(self._brainstorm.invoke(inputs))

    def propose_rescue(
        self,
        request: IdeaRequest,
        *,
        missing_days: Sequence[int],
        excluded_places: Sequence[str],
    ) -> list[PlaceIdea]:
        if not missing_days:
            return []

        inputs = _common_inputs(request)
        inputs["missing_days"] = ", ".join(str(day) for day in missing_days)
        inputs["excluded_places"] = ", ".join(excluded_places) or "none"
        inputs["dates"] = _day_list(request)

        # The model may still return other days; keep only the ones asked for.
        allowed = set(missing_days)

        return [
            idea
            for idea in _to_ideas(self._rescue.invoke(inputs))
            if idea.day in allowed
        ]


__all__ = [
    "GeminiActivityIdeaGenerator",
]
