"""Budget-facing contracts for replaceable external capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol


class CurrencyProviderPort(Protocol):
    """
    Minimal currency capability required by
    budget/orchestration code.
    """

    def get_currency_code(
        self,
        country: str,
    ) -> str:
        """Resolve a country to an ISO-4217 code."""
        ...

    def get_currency_rate(
        self,
        origin_curr_code: str,
        des_curr_code: str,
    ) -> float:
        """
        Return destination currency units for
        one origin currency unit.
        """
        ...


ClaimStatus = Literal["cached", "claimed", "busy", "error"]


class BudgetAssessmentCachePort(Protocol):
    """
    Shared, expiring storage for assessments with a
    single-creator lease, so concurrent requests for the
    same trip pay for provider searches only once.
    """

    def get(
        self,
        key: Mapping[str, Any],
    ) -> Any | None:
        """Return the unexpired payload, or None on miss/error."""
        ...

    def claim(
        self,
        key: Mapping[str, Any],
        *,
        lease_seconds: int,
    ) -> tuple[ClaimStatus, Any, str | None]:
        """Return a cached payload or lease the key for construction."""
        ...

    def publish(
        self,
        key: Mapping[str, Any],
        payload: Any,
        *,
        claim_id: str,
        ttl_hours: int,
    ) -> bool:
        """Store payload only while the caller still owns the lease."""
        ...

    def release(
        self,
        key: Mapping[str, Any],
        *,
        claim_id: str,
    ) -> bool:
        """Give up a lease so another caller can build the entry."""
        ...

    def invalidate(
        self,
        key: Mapping[str, Any],
    ) -> bool:
        """Delete one entry."""
        ...

    def wait_for_change(
        self,
        *,
        timeout_seconds: float,
    ) -> None:
        """Block briefly while another caller holds the lease."""
        ...


__all__ = [
    "ClaimStatus",
    "BudgetAssessmentCachePort",
    "CurrencyProviderPort",
]