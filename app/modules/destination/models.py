"""Destination-resolution domain models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.shared.types import CountryCode, NonEmptyText


DestinationSource = Literal["user", "city_state", "ai"]


class ResolvedDestination(BaseModel):
    """Server-owned, country-verified planning cities."""

    model_config = ConfigDict(frozen=True)

    country: NonEmptyText
    country_code: CountryCode
    cities: tuple[NonEmptyText, ...] = Field(min_length=1)
    source: DestinationSource

    @property
    def primary_city(self) -> str:
        return self.cities[0]


__all__ = [
    "DestinationSource",
    "ResolvedDestination",
]
