"""Booking-search cache adapter backed by the generic Firestore cache store."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from app.infrastructure.database.firebase import FirestoreCacheStore
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_COLLECTION: Final = "api_cache"


class FirebaseBookingSearchCache:
    """Small cache adapter used by booking provider infrastructure."""

    def __init__(self, *, store: FirestoreCacheStore) -> None:
        self._store = store

    def get(
        self,
        *,
        namespace: str,
        key: Mapping[
            str,
            Any,
        ],
    ) -> Any | None:
        try:
            return self._store.get(
                _COLLECTION,
                namespace,
                key,
            )

        except Exception:
            safe_log(
                logger,
                logging.WARNING,
                "booking.cache.read_failed",
            )

            return None

    def set(
        self,
        *,
        namespace: str,
        key: Mapping[
            str,
            Any,
        ],
        payload: Any,
        ttl_hours: int = 1,
    ) -> bool:
        try:
            return bool(
                self._store.set(
                    _COLLECTION,
                    namespace,
                    key,
                    payload,
                    ttl_hours=ttl_hours,
                )
            )

        except Exception:
            safe_log(
                logger,
                logging.WARNING,
                "booking.cache.write_failed",
            )

            return False


__all__ = [
    "FirebaseBookingSearchCache",
]
