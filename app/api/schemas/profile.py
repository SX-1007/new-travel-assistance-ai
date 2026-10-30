"""HTTP transport schemas for onboarding/profile endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.shared.countries import country_alpha2


class OnboardingRequest(BaseModel):
    """One-time onboarding profile captured before trip planning."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User's display name.",
    )

    origin_country: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User's home country.",
    )

    origin_state: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User's home state or province.",
    )

    @field_validator("origin_country")
    @classmethod
    def _known_country(cls, value: str) -> str:
        # Every trip starts here: an unknown country would only surface
        # later, as a failed budget check on every trip.
        if country_alpha2(value) is None:
            raise ValueError("origin_country must be a recognised country name or code")

        return value


class ProfileFields(BaseModel):
    """The onboarding columns of the stored profile (nothing else is sent)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    home_country: str
    home_state: str


class ProfileResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    onboarded: bool
    # Empty until onboarding is complete.
    profile: ProfileFields | dict[str, str] = Field(default_factory=dict)


class ProfileSavedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    user_id: str


__all__ = [
    "OnboardingRequest",
    "ProfileFields",
    "ProfileResponse",
    "ProfileSavedResponse",
]
