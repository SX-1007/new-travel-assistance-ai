"""Contracts required by the profile use cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class ProfileStorePort(Protocol):
    def fetch_profile(self, user_id: str) -> Mapping[str, Any]:
        """The stored profile row ({} when none); raises when unreadable."""
        ...

    def create_profile(self, user_id: str, profile: Mapping[str, Any]) -> bool:
        """Insert or update the given columns; False when the write failed."""
        ...


__all__ = [
    "ProfileStorePort",
]
