"""LangChain-facing place tools with their services injected."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.modules.activities.service import ActivitySearchService
from app.modules.amenities.service import NearbyAmenityService
from app.orchestration.state import AgentState


def build_search_places_tool(
    search_service: ActivitySearchService,
) -> BaseTool:
    """Create the chat attraction/restaurant search tool."""

    @tool("search_places")
    def search_places(
        category: str,
        query: str,
        state: Annotated[AgentState, InjectedState],
        city: str = "",
    ) -> dict[str, Any]:
        """Search for a verified attraction or restaurant to add or swap.

        ``category`` is 'attraction' or 'restaurant'. ``city`` is an optional
        search hint (for example a nearby day-trip town); results always stay
        within the trip's configured city area.
        """

        return search_service.search(
            place_type=category,
            query=query,
            country=state.country or "",
            cities=state.city,
            city_hint=city,
        )

    return search_places


def build_search_nearby_amenities_tool(
    amenity_service: NearbyAmenityService,
) -> BaseTool:
    """Create the nearest-amenity tool (police, hospital, pharmacy, ...)."""

    @tool("search_nearby_amenities")
    def search_nearby_amenities(
        category: str,
        state: Annotated[AgentState, InjectedState],
        specific_location: Optional[str] = None,
    ) -> dict[str, Any]:
        """Find the nearest amenity of ``category`` around a named location.

        ``specific_location`` may be the itinerary hotel's name or another
        place inside the trip city. Returns ranked results and route
        estimates to the nearest one.
        """

        return amenity_service.find_nearest(
            category=category,
            location_name=specific_location,
            itinerary=state.draft_itinerary,
            country=state.country or "",
            cities=state.city,
        )

    return search_nearby_amenities


__all__ = [
    "build_search_nearby_amenities_tool",
    "build_search_places_tool",
]
