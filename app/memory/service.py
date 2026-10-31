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
    """Long-term memory over the profile row and the preference store."""

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
        """Save learned traits; list fields are merged with the stored ones, not replaced."""

        if not user_id or not updates:
            return False

        # Fails closed: if the current profile can't be read, raise rather
        # than save, or an outage would overwrite the stored lists with only
        # the newly learned values.
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
        """The user's profile with the preference store's sentences merged in, for
        prompts. ``{}`` when there is nothing (or the profile can't be read).
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
            # Fails soft: personalisation is optional, a chat turn is not.
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

            # Pacing is one value: the profile column wins over the copy.
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