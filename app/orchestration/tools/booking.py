"""LangChain-facing booking and itinerary-edit tools.

``modify_existing_booking`` and ``edit_itinerary`` only RETURN an edit
request; the graph applies it with ``app.modules.itinerary.editing`` after
checking it against this turn's search results. Both produce the same request
shape, so there is exactly one apply path.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.modules.booking.service import BookingPlanningService
from app.modules.itinerary.editing import ItineraryEdit
from app.orchestration.state import AgentState
from app.orchestration.workflows.booking import booking_trip_from_state


def _edit_request(edits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pending_validation",
        "action": "edit_itinerary",
        "edits": edits,
        "message": f"Submitted {len(edits)} itinerary edit(s) for validation.",
    }


def build_search_alternative_opt_tool(
    booking_service: BookingPlanningService,
) -> BaseTool:
    """Create the flight/hotel alternatives tool."""

    @tool("search_alternative_opt")
    def search_alternative_opt(
        category: Literal["flight", "hotel", "accommodation"],
        day_num: int,
        state: Annotated[AgentState, InjectedState],
        leg: Optional[Literal["outbound", "return"]] = None,
    ) -> dict[str, Any]:
        """Search up to three real alternatives to the current flight or hotel.

        Flights: ``day_num`` must be the first or last day. ``leg`` is only
        needed on a same-day trip. Hotels: alternatives cover the whole stay.
        """

        try:
            trip = booking_trip_from_state(state)

        except ValueError as exc:
            return {"error": f"Trip details are incomplete: {exc}."}

        return booking_service.search_alternatives(
            trip,
            category=category,
            day=day_num,
            itinerary=state.draft_itinerary,
            leg=leg,
        )

    return search_alternative_opt


@tool("modify_existing_booking")
def modify_existing_booking(
    day_num: int,
    category: Literal["flight", "hotel", "accommodation"],
    new_details: dict[str, Any],
) -> dict[str, Any]:
    """Replace a flight or hotel with an option from search_alternative_opt.

    Only call this after the user explicitly confirmed the new selection.
    """

    return _edit_request(
        [
            {
                "day": day_num,
                "action": "replace",
                "category": "hotel" if category == "accommodation" else category,
                "new_details": new_details,
            }
        ]
    )


@tool("edit_itinerary")
def edit_itinerary(edits: list[ItineraryEdit]) -> dict[str, Any]:
    """Apply one or more confirmed itinerary changes in a single call.

    • flight/hotel → action='replace', new_details from search_alternative_opt.
    • activity/restaurant add → action='add', new_details from search_places.
    • activity/restaurant remove → action='remove', index=<the 1-based [n]>.
    • activity/restaurant replace → action='replace', index=<n>, new_details
      from search_places.

    For a place remove/replace, the target day must be stated by the user.
    Only call this after the user confirmed the change and the target day; a
    request that leaves the new place to you ("something else") counts as
    confirmed, so pick the best search result not already in the itinerary.
    """

    return _edit_request([edit.model_dump() for edit in edits])


__all__ = [
    "build_search_alternative_opt_tool",
    "edit_itinerary",
    "modify_existing_booking",
]
