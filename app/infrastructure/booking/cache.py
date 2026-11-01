"""Booking-search cache adapter backed by the generic Redis cache store."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.infrastructure.cache.redis_store import RedisCacheStore
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)


class RedisBookingSearchCache:
    """Booking-search cache. A cache failure counts as a miss (or an unsaved
    entry), never an error: search works without the cache."""

    def __init__(self, *, store: RedisCacheStore) -> None:
        self._store = store

    def get(self, *, namespace: str, key: Mapping[str, Any]) -> Any | None:
        try:
            return self._store.get(namespace, key)

        except Exception:
            safe_log(logger, logging.WARNING, "booking.cache.read_failed")
            return None

    def set(
        self,
        *,
        namespace: str,
        key: Mapping[str, Any],
        payload: Any,
        ttl_hours: int = 1,
    ) -> bool:
        try:
            return bool(self._store.set(namespace, key, payload, ttl_hours=ttl_hours))

        except Exception:
            safe_log(logger, logging.WARNING, "booking.cache.write_failed")
            return False


__all__ = [
    "RedisBookingSearchCache",
]
