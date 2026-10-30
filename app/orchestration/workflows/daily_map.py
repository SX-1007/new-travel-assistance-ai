"""AgentState adapter for daily route maps."""

from __future__ import annotations

from typing import Any

from app.modules.routing.service import (
    DailyRouteService,
)
from app.orchestration.state import (
    AgentState,
)


def run_daily_map(
    state: AgentState,
    *,
    route_service: DailyRouteService,
) -> dict[str, Any]:
    """
    Route the draft itinerary and return only
    the state fields to update.
    """

    result = route_service.build_daily_maps(
        state.draft_itinerary
    )

    return {
        "draft_itinerary":
            result.itinerary,
        "daily_map_info":
            result.daily_maps,
    }


__all__ = [
    "run_daily_map",
]
