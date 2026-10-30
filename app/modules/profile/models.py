"""The onboarding profile and the single rule for "onboarded"."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OnboardingProfile:
    """Who the traveller is and where trips start from."""

    name: str
    home_country: str
    home_state: str

    def as_record(self) -> dict[str, str]:
        return {"name": self.name, "home_country": self.home_country, "home_state": self.home_state}


def onboarding_profile(record: Mapping[str, Any] | None) -> OnboardingProfile | None:
    """The onboarding fields of a stored profile row, or None while any of
    the three is missing (the user must finish onboarding)."""

    if not record:
        return None

    name, country, state = (str(record.get(key) or "").strip() for key in ("name", "home_country", "home_state"))

    if not (name and country and state):
        return None

    return OnboardingProfile(name=name, home_country=country, home_state=state)


__all__ = [
    "OnboardingProfile",
    "onboarding_profile",
]
