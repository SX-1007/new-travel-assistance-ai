"""GET /api/geocode/?q=…: exact coordinates for a place named in a reply.

Never fails the page: a miss or a provider failure is ``found: false``
(the client keeps the pin it already has).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, RequestId
from app.api.providers import PinLocator
from app.api.schemas.geocode import GeocodeResponse
from app.modules.places.exceptions import PlaceSearchUnavailable
from app.shared.observability import LogContext, log_event


logger = logging.getLogger(__name__)
router = APIRouter(tags=["geocode"])


@router.get(
    "/",
    response_model=GeocodeResponse,
    response_model_exclude_none=True,
    summary="Resolve a place name/address to Google Maps coordinates",
)
async def geocode(
    user_id: CurrentUser,
    request_id: RequestId,
    locator: PinLocator,
    q: str = Query(..., min_length=3, max_length=300),
    lat: float | None = Query(None, ge=-90, le=90),
    lng: float | None = Query(None, ge=-180, le=180),
) -> GeocodeResponse:
    """``lat``/``lng`` (both or neither) bias the search area."""

    del user_id  # authentication only
    near = (lat, lng) if lat is not None and lng is not None else None

    try:
        place = await asyncio.to_thread(locator.locate, q, near=near)
    except PlaceSearchUnavailable:
        _log(request_id, "unavailable")
        return GeocodeResponse(status="error", found=False)

    if place is None:
        _log(request_id, "not_found")
        return GeocodeResponse(status="success", found=False)

    return GeocodeResponse(
        status="success",
        found=True,
        name=place.title,
        address=place.address,
        lat=place.latitude,
        lng=place.longitude,
    )


def _log(request_id: str, outcome: str) -> None:
    log_event(logger, "geocode.lookup", context=LogContext(request_id=request_id), stage="geocode", outcome=outcome)
