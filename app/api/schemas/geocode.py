"""HTTP transport schema for place pin lookups."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class GeocodeResponse(BaseModel):
    """Always 200: ``found: false`` tells the client to keep its own pin.

    Absent fields are omitted from the JSON, never sent as null (the client
    rejects a null name/address/lat/lng).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"]
    found: bool
    name: str | None = None
    address: str | None = None
    lat: float | None = None
    lng: float | None = None


__all__ = [
    "GeocodeResponse",
]
