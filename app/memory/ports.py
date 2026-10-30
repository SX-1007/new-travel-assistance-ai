"""Contracts required by memory use cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class UserProfileStorePort(Protocol):
    """Persistence contract for relational user-profile memory."""

    def fetch_profile(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """Return profile data or raise when storage cannot be read."""
        ...

    def update_profile(
        self,
        user_id: str,
        updates: Mapping[str, Any],
    ) -> bool:
        """Persist already-prepared profile fields."""
        ...


class SemanticPreferenceStorePort(Protocol):
    """Persistence contract for semantic long-term preferences."""

    def fetch_preferences(
        self,
        user_id: str,
    ) -> dict[str, list[str]]:
        """Return semantic preferences grouped by category."""
        ...

    def insert_preferences(
        self,
        user_id: str,
        categorized_preferences: Mapping[
            str,
            list[str],
        ],
    ) -> bool:
        """Persist semantic preferences."""
        ...


__all__ = [
    "UserProfileStorePort",
    "SemanticPreferenceStorePort",
]