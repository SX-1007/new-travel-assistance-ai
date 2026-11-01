"""Pure rules for the provider-grounded minimum trip budget.

No network, cache, settings or orchestration imports. Everything here can be
tested with plain values; ``service.py`` supplies the real provider data.

The minimum is the larger of:
  • cheapest outbound + return flight, divided by the transportation ratio
  • cheapest hotel price x nights, divided by the accommodation ratio
so that the default allocation always covers the real cheapest options.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError

from app.modules.booking.hotels import calculate_total_nights
from app.modules.budget.models import (
    BudgetAssessment,
    BudgetEvidence,
    BudgetTrip,
)
from app.modules.budget.policies import DEFAULT_BUDGET_RATIOS


# Stored inside every assessment and its cache key. Bump it whenever the
# formula or DEFAULT_BUDGET_RATIOS change so stale assessments are ignored.
CALCULATION_VERSION = "allocation-v1"
ASSESSMENT_TTL_HOURS = 1

_PRICE_REL_TOL = 1e-12


def primary_destination_city(cities: Any) -> str:
    """Return the exact first requested city used by the hotel budget floor."""
    if not isinstance(cities, (list, tuple)) or not cities:
        return ""
    city = cities[0]
    return city.strip() if isinstance(city, str) else ""


def assessment_cache_key(
    trip: BudgetTrip,
    *,
    base_currency: str,
    destination_currency: str,
) -> dict[str, Any]:
    """The budget-independent identity of a trip's assessment.

    Its field names and normalisation must not change while assessments are
    being confirmed: they locate the saved assessment in a later request.
    """
    return {
        "origin": trip.origin.strip().lower(),
        "destination": trip.destination.strip().lower(),
        "destination_city": trip.destination_city.strip().lower(),
        "start_date": trip.start_date.strip(),
        "end_date": trip.end_date.strip(),
        "num_people": int(trip.num_people),
        "base_currency": base_currency.strip().upper(),
        "destination_currency": destination_currency.strip().upper(),
        "calculation_version": CALCULATION_VERSION,
    }


def finite_positive(value: Any) -> float | None:
    """Return a positive finite number, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def minimum_destination_budget(
    *,
    outbound_flight_price: float,
    return_flight_price: float,
    hotel_price_per_night: float,
    hotel_nights: int,
) -> float:
    """Smallest destination-currency total whose default split covers the evidence."""
    return max(
        (outbound_flight_price + return_flight_price)
        / DEFAULT_BUDGET_RATIOS["transportation"],
        (hotel_price_per_night * hotel_nights)
        / DEFAULT_BUDGET_RATIOS["accommodation"],
    )


def recommended_base_budget(
    minimum_destination: float,
    exchange_rate: float,
) -> float:
    """Convert the minimum to the user's currency, rounded UP to the cent."""
    return math.ceil((minimum_destination / exchange_rate) * 100.0) / 100.0


def build_assessment(
    trip: BudgetTrip,
    *,
    base_currency: str,
    destination_currency: str,
    exchange_rate: float,
    outbound_flight: dict[str, Any],
    return_flight: dict[str, Any],
    hotel: dict[str, Any],
    now: datetime,
) -> BudgetAssessment:
    """Create an assessment from already-fetched cheapest provider results.

    Raises ValueError when any required price is missing or invalid.
    """
    rate = finite_positive(exchange_rate)
    if rate is None:
        raise ValueError("Invalid exchange rate.")

    nights = calculate_total_nights(trip.start_date, trip.end_date)

    outbound_price = finite_positive(outbound_flight.get("price"))
    return_price = finite_positive(return_flight.get("price"))
    if outbound_price is None or return_price is None:
        raise ValueError("Invalid flight price.")

    if nights > 0:
        hotel_price = finite_positive(hotel.get("price_per_night"))
        if hotel_price is None:
            raise ValueError("Invalid hotel price.")
    else:
        hotel, hotel_price = {}, 0.0

    minimum = minimum_destination_budget(
        outbound_flight_price=outbound_price,
        return_flight_price=return_price,
        hotel_price_per_night=hotel_price,
        hotel_nights=nights,
    )

    return BudgetAssessment(
        assessment_id=str(uuid.uuid4()),
        calculation_version=CALCULATION_VERSION,
        origin=trip.origin.strip(),
        destination=trip.destination.strip(),
        destination_city=trip.destination_city.strip(),
        start_date=trip.start_date,
        end_date=trip.end_date,
        num_people=trip.num_people,
        base_currency=base_currency,
        destination_currency=destination_currency,
        exchange_rate=rate,
        minimum_destination_budget=minimum,
        recommended_minimum_budget=recommended_base_budget(minimum, rate),
        evidence=BudgetEvidence(
            outbound_flight=outbound_flight,
            return_flight=return_flight,
            hotel=hotel,
            outbound_flight_price=outbound_price,
            return_flight_price=return_price,
            hotel_price_per_night=hotel_price,
            hotel_nights=nights,
        ),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=ASSESSMENT_TTL_HOURS)).isoformat(),
    )


def assessment_arithmetic_is_valid(assessment: BudgetAssessment) -> bool:
    """Reject stored evidence whose objects and derived amounts disagree."""
    evidence = assessment.evidence

    object_outbound = finite_positive(evidence.outbound_flight.get("price"))
    object_return = finite_positive(evidence.return_flight.get("price"))
    if object_outbound is None or object_return is None:
        return False

    price_pairs = [
        (object_outbound, evidence.outbound_flight_price),
        (object_return, evidence.return_flight_price),
    ]
    if evidence.hotel_nights:
        object_hotel = finite_positive(evidence.hotel.get("price_per_night"))
        if object_hotel is None:
            return False
        price_pairs.append((object_hotel, evidence.hotel_price_per_night))
    elif evidence.hotel or evidence.hotel_price_per_night != 0:
        return False

    if not all(
        math.isclose(left, right, rel_tol=_PRICE_REL_TOL)
        for left, right in price_pairs
    ):
        return False

    try:
        nights = calculate_total_nights(assessment.start_date, assessment.end_date)
    except ValueError:
        return False
    if evidence.hotel_nights != nights:
        return False

    expected_minimum = minimum_destination_budget(
        outbound_flight_price=evidence.outbound_flight_price,
        return_flight_price=evidence.return_flight_price,
        hotel_price_per_night=evidence.hotel_price_per_night,
        hotel_nights=evidence.hotel_nights,
    )
    return math.isclose(
        assessment.minimum_destination_budget,
        expected_minimum,
        rel_tol=_PRICE_REL_TOL,
    ) and math.isclose(
        assessment.recommended_minimum_budget,
        recommended_base_budget(expected_minimum, assessment.exchange_rate),
        rel_tol=_PRICE_REL_TOL,
    )


def assessment_is_unexpired(
    assessment: BudgetAssessment,
    *,
    now: datetime | None = None,
) -> bool:
    try:
        expires_at = datetime.fromisoformat(assessment.expires_at)
    except ValueError:
        return False
    current = now or datetime.now(timezone.utc)
    return expires_at.tzinfo is not None and expires_at > current


def assessment_matches_trip(
    assessment: BudgetAssessment | dict[str, Any],
    trip: BudgetTrip,
    *,
    now: datetime | None = None,
) -> bool:
    """Return whether evidence is unexpired, self-consistent and for this exact trip."""
    parsed = parse_assessment(assessment)
    if parsed is None or not assessment_is_unexpired(parsed, now=now):
        return False
    return (
        parsed.calculation_version == CALCULATION_VERSION
        and assessment_arithmetic_is_valid(parsed)
        and parsed.origin.strip().lower() == trip.origin.strip().lower()
        and parsed.destination.strip().lower() == trip.destination.strip().lower()
        and parsed.destination_city.strip().lower()
        == trip.destination_city.strip().lower()
        and parsed.start_date == trip.start_date
        and parsed.end_date == trip.end_date
        and parsed.num_people == trip.num_people
    )


def parse_assessment(payload: Any) -> BudgetAssessment | None:
    """Validate an untrusted stored payload; None when it is not an assessment."""
    if isinstance(payload, BudgetAssessment):
        return payload
    if not payload:
        return None
    try:
        return BudgetAssessment.model_validate(payload)
    except (TypeError, ValueError, ValidationError):
        return None


def matching_assessment(
    payload: Any,
    trip: BudgetTrip,
    *,
    now: datetime | None = None,
) -> BudgetAssessment | None:
    """Return the payload as an assessment only if it matches this trip."""
    parsed = parse_assessment(payload)
    if parsed is None or not assessment_matches_trip(parsed, trip, now=now):
        return None
    return parsed


def is_budget_sufficient(
    total_base_budget: Any,
    assessment: BudgetAssessment,
) -> bool:
    """Compare the entered amount at full destination-currency precision."""
    entered_budget = finite_positive(total_base_budget)
    return (
        entered_budget is not None
        and entered_budget * assessment.exchange_rate
        >= assessment.minimum_destination_budget
    )


__all__ = [
    "CALCULATION_VERSION",
    "ASSESSMENT_TTL_HOURS",
    "primary_destination_city",
    "assessment_cache_key",
    "finite_positive",
    "minimum_destination_budget",
    "recommended_base_budget",
    "build_assessment",
    "assessment_arithmetic_is_valid",
    "assessment_is_unexpired",
    "assessment_matches_trip",
    "parse_assessment",
    "matching_assessment",
    "is_budget_sufficient",
]
