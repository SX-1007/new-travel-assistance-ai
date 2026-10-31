"""Budget-assessment cache adapter over the generic Firestore cache store.

The collection and prefix must not change: assessments already in Firestore
are stored under them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from app.infrastructure.database.firebase import ClaimStatus, FirestoreCacheStore


_COLLECTION: Final = "api_cache"
_PREFIX: Final = "budget_assessment"


class FirestoreBudgetAssessmentCache:
    """Implements ``BudgetAssessmentCachePort`` with Firestore leases."""

    def __init__(self, *, store: FirestoreCacheStore) -> None:
        self._store = store

    def get(self, key: Mapping[str, Any]) -> Any | None:
        return self._store.get(_COLLECTION, _PREFIX, key)

    def claim(
        self,
        key: Mapping[str, Any],
        *,
        lease_seconds: int,
    ) -> tuple[ClaimStatus, Any, str | None]:
        return self._store.claim(
            _COLLECTION, _PREFIX, key, lease_seconds=lease_seconds
        )

    def publish(
        self,
        key: Mapping[str, Any],
        payload: Any,
        *,
        claim_id: str,
        ttl_hours: int,
    ) -> bool:
        return self._store.complete_claim(
            _COLLECTION,
            _PREFIX,
            key,
            payload,
            claim_id=claim_id,
            ttl_hours=ttl_hours,
        )

    def release(self, key: Mapping[str, Any], *, claim_id: str) -> bool:
        return self._store.release_claim(_COLLECTION, _PREFIX, key, claim_id=claim_id)

    def invalidate(self, key: Mapping[str, Any]) -> bool:
        return self._store.invalidate(_COLLECTION, _PREFIX, key)

    def wait_for_change(self, *, timeout_seconds: float) -> None:
        self._store.wait_for_claim_change(timeout_seconds=timeout_seconds)


__all__ = [
    "FirestoreBudgetAssessmentCache",
]
