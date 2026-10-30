"""Supabase REST persistence for relational application data.

No client is created at import time. The composition root builds one client
with ``create_supabase_client`` and injects it into each store.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict

from supabase import Client, create_client

from app.shared.logging import safe_log

logger = logging.getLogger(__name__)


def create_supabase_client(url: str, api_key: str) -> Client:
    """Build one Supabase REST client from explicit credentials."""
    if not url or not url.strip():
        raise ValueError("url must not be empty")

    if not api_key or not api_key.strip():
        raise ValueError("api_key must not be empty")

    return create_client(url.strip(), api_key.strip())


# ── User profile persistence ────────────────────────────────────────
class SupabaseUserProfileStore:
    """
    Supabase implementation of the memory
    user-profile persistence contract.
    """

    def __init__(self, *, client: Client) -> None:
        self._client = client

    def fetch_profile(
        self,
        user_id: str,
        *,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve a user's persisted relational profile.

        Raises by default so memory updates fail closed; pass
        ``raise_on_error=False`` where an empty profile is acceptable.
        """
        try:
            response = (
                self._client.table("user_profiles")
                .select("*")
                .eq("id", user_id)
                .execute()
            )
            return response.data[0] if response.data else {}
        except Exception:
            safe_log(
                logger,
                logging.ERROR,
                "storage.supabase.profile_fetch_failed",
            )

            if raise_on_error:
                raise

            return {}

    def create_profile(
        self,
        user_id: str,
        profile: Mapping[str, Any],
    ) -> bool:
        """Create or overwrite a user's onboarding profile."""
        payload = {
            "id": user_id,
            **profile,
        }

        try:
            (
                self._client.table("user_profiles")
                .upsert(
                    payload,
                    on_conflict="id",
                )
                .execute()
            )

            safe_log(
                logger,
                logging.INFO,
                "storage.supabase.profile_created",
            )
            return True

        except Exception:
            safe_log(
                logger,
                logging.ERROR,
                "storage.supabase.profile_create_failed",
            )
            return False

    def update_profile(
        self,
        user_id: str,
        updates: Mapping[str, Any],
    ) -> bool:
        """Persist already-prepared profile fields.

        This infrastructure method intentionally contains no memory merge
        rules. Memory-specific preparation belongs to app.memory.service.
        """
        if not user_id or not updates:
            return False

        try:
            result = (
                self._client.table("user_profiles")
                .update(dict(updates))
                .eq("id", user_id)
                .execute()
            )

            if result.data:
                safe_log(
                    logger,
                    logging.INFO,
                    "storage.supabase.profile_updated",
                )
                return True

            safe_log(
                logger,
                logging.WARNING,
                "storage.supabase.profile_update_empty",
            )
            return False

        except Exception:
            safe_log(
                logger,
                logging.ERROR,
                "storage.supabase.profile_update_failed",
            )
            return False


# ── Saved-trip persistence ──────────────────────────────────────────
class SupabaseSavedTripStore:
    """Persist completed trips in the ``saved_trips`` table."""

    def __init__(self, *, client: Client) -> None:
        self._client = client

    def save_final_itinerary(
        self,
        user_id: str,
        trip_id: str,
        itinerary_data: dict,
        geojson_data: dict,
    ) -> None:
        """Persist the completed trip."""
        payload = {
            "user_id": user_id,
            "trip_id": trip_id,
            "itinerary_text": itinerary_data,
            "map_geojson": geojson_data,
        }

        try:
            result = (
                self._client.table("saved_trips")
                .upsert(
                    payload,
                    on_conflict="trip_id",
                )
                .execute()
            )

            if result.data:
                safe_log(
                    logger,
                    logging.INFO,
                    "storage.supabase.itinerary_saved",
                )
            else:
                safe_log(
                    logger,
                    logging.WARNING,
                    "storage.supabase.itinerary_save_empty",
                )

        except Exception:
            safe_log(
                logger,
                logging.ERROR,
                "storage.supabase.itinerary_save_failed",
            )


__all__ = [
    "create_supabase_client",
    "SupabaseUserProfileStore",
    "SupabaseSavedTripStore",
]
