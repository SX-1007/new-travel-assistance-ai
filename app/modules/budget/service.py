"""Budget-assessment use cases over injected provider and cache contracts.

This module coordinates *where data comes from*; the rules themselves live in
``assessment.py``. It never imports a concrete provider, cache, settings,
FastAPI or LangGraph — the composition root injects them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from app.modules.booking.hotels import calculate_total_nights
from app.modules.booking.ports import FlightSearchPort, HotelSearchPort
from app.modules.budget.assessment import (
    ASSESSMENT_TTL_HOURS,
    assessment_cache_key,
    build_assessment,
    matching_assessment,
)
from app.modules.budget.exceptions import BudgetAssessmentUnavailable
from app.modules.budget.models import (
    BudgetAssessment,
    BudgetAssessmentResult,
    BudgetTrip,
)
from app.modules.budget.ports import (
    BudgetAssessmentCachePort,
    CurrencyProviderPort,
)
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

ASSESSMENT_CLAIM_LEASE_SECONDS = 120
ASSESSMENT_CLAIM_WAIT_SECONDS = 10.0
ASSESSMENT_CLAIM_POLL_SECONDS = 0.25


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cheapest(future: Future, label: str) -> dict[str, Any]:
    """Read one cheapest result without accepting missing provider evidence."""
    try:
        results: Sequence[Any] = future.result()
    except Exception as exc:
        raise BudgetAssessmentUnavailable(f"{label} search failed.") from exc

    if not results:
        raise BudgetAssessmentUnavailable(f"No {label} result is available.")

    first = results[0]
    if isinstance(first, BaseModel):
        return first.model_dump(mode="json")
    if isinstance(first, dict):
        return dict(first)
    raise BudgetAssessmentUnavailable(f"No {label} result is available.")


class BudgetAssessmentService:
    """Create, reuse and reload provider-grounded minimum budgets."""

    def __init__(
        self,
        *,
        currency_provider: CurrencyProviderPort,
        flight_search: FlightSearchPort,
        hotel_search: HotelSearchPort,
        cache: BudgetAssessmentCachePort,
        clock: Callable[[], datetime] = _utc_now,
        claim_wait_seconds: float = ASSESSMENT_CLAIM_WAIT_SECONDS,
    ) -> None:
        self._currency = currency_provider
        self._flights = flight_search
        self._hotels = hotel_search
        self._cache = cache
        self._clock = clock
        self._claim_wait_seconds = claim_wait_seconds

    # ── Public use cases ────────────────────────────────────────────

    def get_or_create(self, trip: BudgetTrip) -> BudgetAssessmentResult:
        """Return a cached assessment or build one from cheapest real prices.

        ``persisted`` is False when the new assessment could not be saved
        (including when the cache is down); such an assessment must not be
        offered for later confirmation. Raises BudgetAssessmentUnavailable
        when real data is missing.
        """
        base_currency, destination_currency = self._currencies(trip)
        key = assessment_cache_key(
            trip,
            base_currency=base_currency,
            destination_currency=destination_currency,
        )

        cached = self._matching(self._safe_get(key, "budget.cache_read_failed"), trip)
        if cached is not None:
            return BudgetAssessmentResult(assessment=cached, persisted=True)

        claim = self._claim_or_reuse(key, trip)
        if isinstance(claim, BudgetAssessment):
            return BudgetAssessmentResult(assessment=claim, persisted=True)
        claim_id = claim

        try:
            assessment = self._build(
                trip,
                base_currency=base_currency,
                destination_currency=destination_currency,
            )
        except Exception:
            if claim_id is not None:
                self._cache.release(key, claim_id=claim_id)
            raise

        if claim_id is None:
            # The cache is down: nothing can be saved, but a sufficient
            # budget can still be accepted.
            return BudgetAssessmentResult(assessment=assessment, persisted=False)

        try:
            persisted = self._cache.publish(
                key,
                assessment.model_dump(),
                claim_id=claim_id,
                ttl_hours=ASSESSMENT_TTL_HOURS,
            )
        except Exception:
            safe_log(logger, logging.WARNING, "budget.cache_publish_failed")
            persisted = False

        if not persisted:
            self._cache.release(key, claim_id=claim_id)

        return BudgetAssessmentResult(assessment=assessment, persisted=persisted)

    def load_confirmed(
        self,
        trip: BudgetTrip,
        *,
        assessment_id: str,
    ) -> BudgetAssessment | None:
        """Reload a saved assessment by id without calling any paid provider."""
        try:
            base_currency, destination_currency = self._currencies(trip)
        except BudgetAssessmentUnavailable:
            return None

        key = assessment_cache_key(
            trip,
            base_currency=base_currency,
            destination_currency=destination_currency,
        )
        assessment = self._matching(
            self._safe_get(key, "budget.confirmed_cache_read_failed"),
            trip,
        )
        if assessment is None or assessment.assessment_id != assessment_id:
            return None
        return assessment

    def matches(self, assessment: Any, trip: BudgetTrip) -> bool:
        """Whether evidence is unexpired and for this trip, by this service's clock."""
        return self._matching(assessment, trip) is not None

    # ── Internals ───────────────────────────────────────────────────

    def _currencies(self, trip: BudgetTrip) -> tuple[str, str]:
        try:
            return (
                self._currency.get_currency_code(trip.origin),
                self._currency.get_currency_code(trip.destination),
            )
        except Exception as exc:
            raise BudgetAssessmentUnavailable("Currency resolution failed.") from exc

    def _matching(self, payload: Any, trip: BudgetTrip) -> BudgetAssessment | None:
        return matching_assessment(payload, trip, now=self._clock())

    def _safe_get(self, key: dict[str, Any], failure_event: str) -> Any:
        try:
            return self._cache.get(key)
        except Exception:
            safe_log(logger, logging.WARNING, failure_event)
            return None

    def _claim_or_reuse(
        self,
        key: dict[str, Any],
        trip: BudgetTrip,
    ) -> BudgetAssessment | str | None:
        """Return another caller's finished assessment, our lease id, or None
        when the cache can't lease (the caller builds without one)."""
        deadline = time.monotonic() + self._claim_wait_seconds

        while True:
            try:
                status, payload, claim_id = self._cache.claim(
                    key,
                    lease_seconds=ASSESSMENT_CLAIM_LEASE_SECONDS,
                )
            except Exception:
                status, payload, claim_id = "error", None, None

            if status == "error":
                safe_log(logger, logging.WARNING, "budget.cache_claim_failed")
                return None

            if status == "cached":
                reused = self._matching(payload, trip)
                if reused is not None:
                    return reused
                # A stored entry that fails validation must not block the
                # trip forever: drop it and claim again. If it can't be
                # dropped, claiming again would find it again, so build
                # without a lease instead of looping.
                try:
                    dropped = self._cache.invalidate(key)
                except Exception:
                    dropped = False
                if not dropped:
                    safe_log(logger, logging.WARNING, "budget.cache_invalidate_failed")
                    return None
                continue

            if status == "claimed" and claim_id:
                return claim_id

            if status != "busy" or time.monotonic() >= deadline:
                raise BudgetAssessmentUnavailable(
                    "Budget assessment creation is unavailable."
                )

            self._cache.wait_for_change(
                timeout_seconds=min(
                    ASSESSMENT_CLAIM_POLL_SECONDS,
                    max(0.0, deadline - time.monotonic()),
                ),
            )

    def _build(
        self,
        trip: BudgetTrip,
        *,
        base_currency: str,
        destination_currency: str,
    ) -> BudgetAssessment:
        try:
            exchange_rate = self._currency.get_currency_rate(
                base_currency,
                destination_currency,
            )
        except Exception as exc:
            raise BudgetAssessmentUnavailable("Exchange-rate lookup failed.") from exc

        nights = calculate_total_nights(trip.start_date, trip.end_date)

        # Outbound, return and hotel searches are independent: run in parallel.
        with ThreadPoolExecutor(max_workers=3 if nights else 2) as executor:
            outbound_future = executor.submit(
                self._flights.search_flights,
                origin=trip.origin,
                destination=trip.destination,
                departure_date=trip.start_date,
                currency=destination_currency,
                adults=trip.num_people,
            )
            return_future = executor.submit(
                self._flights.search_flights,
                origin=trip.destination,
                destination=trip.origin,
                departure_date=trip.end_date,
                currency=destination_currency,
                adults=trip.num_people,
            )
            hotel_future = (
                executor.submit(
                    self._hotels.search_hotels,
                    destination=f"{trip.destination_city.strip()}, {trip.destination.strip()}",
                    check_in=trip.start_date,
                    check_out=trip.end_date,
                    currency=destination_currency,
                    adults=trip.num_people,
                )
                if nights
                else None
            )

            outbound = _cheapest(outbound_future, "outbound flight")
            return_flight = _cheapest(return_future, "return flight")
            hotel = _cheapest(hotel_future, "hotel") if hotel_future else {}

        try:
            return build_assessment(
                trip,
                base_currency=base_currency,
                destination_currency=destination_currency,
                exchange_rate=exchange_rate,
                outbound_flight=outbound,
                return_flight=return_flight,
                hotel=hotel,
                now=self._clock(),
            )
        except ValueError as exc:
            raise BudgetAssessmentUnavailable(str(exc)) from exc


__all__ = [
    "BudgetAssessmentService",
]
