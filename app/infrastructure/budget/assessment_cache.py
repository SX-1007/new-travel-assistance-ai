"""Budget-assessment cache adapter over the generic Redis cache store.

The prefix must not change while assessments are being confirmed: a saved
assessment is read back under it in a later request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from app.infrastructure.cache.redis_store import ClaimStatus, RedisCacheStore


_PREFIX: Final = "budget_assessment"


class RedisBudgetAssessmentCache:
    """Implements ``BudgetAssessmentCachePort`` with Redis leases."""

    def __init__(self, *, store: RedisCacheStore) -> None:
        self._store = store

    def get(self, key: Mapping[str, Any]) -> Any | None:
        return self._store.get(_PREFIX, key)

    def claim(
        self,
        key: Mapping[str, Any],
        *,
        lease_seconds: int,
    ) -> tuple[ClaimStatus, Any, str | None]:
        return self._store.claim(_PREFIX, key, lease_seconds=lease_seconds)

    def publish(
        self,
        key: Mapping[str, Any],
        payload: Any,
        *,
        claim_id: str,
        ttl_hours: int,
    ) -> bool:
        return self._store.complete_claim(_PREFIX, key, payload, claim_id=claim_id, ttl_hours=ttl_hours)

    def release(self, key: Mapping[str, Any], *, claim_id: str) -> bool:
        return self._store.release_claim(_PREFIX, key, claim_id=claim_id)

    def invalidate(self, key: Mapping[str, Any]) -> bool:
        return self._store.invalidate(_PREFIX, key)

    def wait_for_change(self, *, timeout_seconds: float) -> None:
        self._store.wait_for_claim_change(timeout_seconds=timeout_seconds)


__all__ = [
    "RedisBudgetAssessmentCache",
]
