"""Memory use-cases over injected persistence contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.memory.ports import (
    SemanticPreferenceStorePort,
    UserProfileStorePort,
)
from app.memory.preferences import (
    MEMORY_LIST_FIELDS,
    MEM0_TO_PROFILE_FIELD,
    merge_unique_memory_values,
)


class MemoryService:
    """
    Coordinate long-term-memory behaviour without
    depending on concrete storage technology.
    """

    def __init__(
        self,
        *,
        profile_store: UserProfileStorePort,
        semantic_store: SemanticPreferenceStorePort,
    ) -> None:
        self._profile_store = profile_store
        self._semantic_store = semantic_store

    def update_profile(
        self,
        user_id: str,
        updates: Mapping[str, Any],
    ) -> bool:
        """
        Merge learned list-based memory fields before
        persisting the resulting profile update.
        """

        if not user_id or not updates:
            return False

        # IMPORTANT:
        # Updating memory must fail closed if the old
        # profile cannot be read. Otherwise an outage
        # could cause existing list memories to be
        # overwritten by only the newly extracted values.
        current = self._profile_store.fetch_profile(
            user_id
        )

        payload = dict(updates)

        for field in MEMORY_LIST_FIELDS:
            if field in payload:
                payload[field] = (
                    merge_unique_memory_values(
                        current.get(field),
                        payload[field],
                    )
                )

        return self._profile_store.update_profile(
            user_id,
            payload,
        )

    def fetch_context(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Return the merged relational and semantic
        long-term-memory context.
        """

        if not user_id:
            return {}

        try:
            merged: dict[str, Any] = dict(
                self._profile_store.fetch_profile(
                    user_id
                )
            )

        except Exception:
            # Context retrieval may fail soft because
            # failing to retrieve optional context must
            # not make an ordinary chat request fail.
            merged = {}

        vector_memory = (
            self._semantic_store
            .fetch_preferences(
                user_id
            )
        )

        for category, values in (
            vector_memory.items()
        ):
            if not values:
                continue

            if category == "travel_pacing":
                if not merged.get(
                    "travel_pacing"
                ):
                    merged[
                        "travel_pacing"
                    ] = values[0]

                continue

            profile_field = (
                MEM0_TO_PROFILE_FIELD.get(
                    category
                )
            )

            if profile_field:
                merged[profile_field] = (
                    merge_unique_memory_values(
                        merged.get(
                            profile_field
                        ),
                        values,
                    )
                )

        return merged

    def insert_preferences(
        self,
        user_id: str,
        categorized_preferences: Mapping[
            str,
            list[str],
        ],
    ) -> bool:
        """Persist extracted semantic preferences."""

        if (
            not user_id
            or not categorized_preferences
        ):
            return False

        return (
            self._semantic_store
            .insert_preferences(
                user_id,
                categorized_preferences,
            )
        )


__all__ = [
    "MemoryService",
]