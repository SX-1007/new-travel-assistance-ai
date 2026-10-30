"""Booking planning: the trip's flights and hotel, and chat alternatives.

``plan_skeleton`` produces the day skeleton that activity planning fills in.
It reuses the flights and hotel the budget check already priced when the
caller supplies them, otherwise it searches.

``search_alternatives`` serves the chat "find me another flight/hotel"
request. Applying a chosen alternative is an itinerary edit
(``app.modules.itinerary.editing``), not a booking concern.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypeVar

from pydantic import ValidationError

from app.modules.booking.flights import normalise_iata_code
from app.modules.booking.hotels import normalise_hotel_coordinates
from app.modules.booking.models import (
    BookingTrip,
    FlightResult,
    HotelResult,
    SkeletonResult,
)
from app.modules.booking.ports import FlightSearchPort, HotelSearchPort
from app.modules.booking.selection import (
    alternatives,
    budget_cap,
    build_skeleton,
    flight_identity,
    hotel_identity,
    nightly_cap,
    pick_flight_pair,
    pick_hotel,
)
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

ALTERNATIVES_LIMIT = 3

Leg = Literal["outbound", "return"]

_T = TypeVar("_T")


def _safe_results(call: Callable[[], list[_T]], event: str) -> list[_T]:
    """One provider failure must not erase the other searches' results."""

    try:
        return list(call() or [])

    except Exception:
        safe_log(logger, logging.WARNING, event)
        return []


def _flight_legs(day: Mapping[str, Any]) -> list[FlightResult]:
    raw = day.get("flight")
    raw_legs = raw if isinstance(raw, list) else [raw] if isinstance(raw, Mapping) else []
    legs: list[FlightResult] = []

    for leg in raw_legs:
        try:
            legs.append(FlightResult.model_validate(leg))
        except (ValidationError, TypeError):
            continue

    return legs


def _day(itinerary: Sequence[Mapping[str, Any]], number: int) -> Mapping[str, Any] | None:
    return next(
        (day for day in itinerary if isinstance(day, Mapping) and day.get("day") == number),
        None,
    )


def _day_numbers(itinerary: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        day["day"]
        for day in itinerary
        if isinstance(day, Mapping)
        and isinstance(day.get("day"), int)
        and not isinstance(day.get("day"), bool)
    ]


class BookingPlanningService:
    """Flights + hotel for a trip, without LangGraph or AgentState."""

    def __init__(
        self,
        *,
        flights: FlightSearchPort,
        hotels: HotelSearchPort,
    ) -> None:
        self._flights = flights
        self._hotels = hotels

    # ── Day skeleton ────────────────────────────────────────────────

    def plan_skeleton(
        self,
        trip: BookingTrip,
        *,
        reuse: Mapping[str, Any] | None = None,
    ) -> SkeletonResult:
        """Return the day skeleton.

        ``reuse`` is budget evidence ({outbound_flight, return_flight, hotel})
        that the caller already matched to this exact trip. It is still
        checked structurally here, with no network calls.
        """

        reused = self._reusable(trip, reuse) if reuse is not None else None

        if reused is not None:
            outbound, returning, hotel = reused
            return SkeletonResult(
                itinerary=build_skeleton(
                    start_date=trip.start_date,
                    nights=trip.nights,
                    outbound=outbound,
                    returning=returning,
                    hotel=hotel,
                ),
                source="assessment",
            )

        outbound_options, return_options, hotel_options = self._search_all(trip)

        outbound, returning = pick_flight_pair(
            outbound_options,
            return_options,
            budget_cap(trip.transportation_budget),
        )
        hotel = pick_hotel(
            hotel_options,
            nightly_cap(trip.accommodation_budget, trip.nights),
        )

        missing = tuple(
            label
            for label, found in (
                ("outbound_flight", outbound is not None),
                ("return_flight", returning is not None),
                ("hotel", hotel is not None or trip.nights == 0),
            )
            if not found
        )

        if missing:
            safe_log(logger, logging.WARNING, "booking.skeleton_incomplete")

        return SkeletonResult(
            itinerary=build_skeleton(
                start_date=trip.start_date,
                nights=trip.nights,
                outbound=outbound,
                returning=returning,
                hotel=hotel,
            ),
            source="search",
            missing=missing,  # type: ignore[arg-type]
        )

    # ── Chat alternatives ───────────────────────────────────────────

    def search_alternatives(
        self,
        trip: BookingTrip,
        *,
        category: str,
        day: int,
        itinerary: Sequence[Mapping[str, Any]],
        leg: Leg | None = None,
    ) -> dict[str, Any]:
        """Up to three real alternatives to the current flight or hotel.

        ``leg`` only matters on a same-day trip, where day 1 holds both
        flights; it defaults to the outbound flight.
        """

        kind = (category or "").strip().lower()

        if kind == "flight":
            return self._flight_alternatives(trip, day=day, itinerary=itinerary, leg=leg)

        if kind in {"hotel", "accommodation"}:
            return self._hotel_alternatives(trip, itinerary=itinerary)

        return {"error": f"Unknown category: '{category}'. Use 'flight' or 'hotel'."}

    # ── Internals ───────────────────────────────────────────────────

    def _search_all(
        self,
        trip: BookingTrip,
    ) -> tuple[list[FlightResult], list[FlightResult], list[HotelResult]]:
        """Outbound, return and hotel searches in parallel, unfiltered.

        Budget preference is applied afterwards, so a tight budget degrades
        to "cheapest real option, flagged" instead of an empty slot.
        """

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="booking") as executor:
            outbound = executor.submit(
                _safe_results,
                lambda: self._search_leg(trip, outbound=True),
                "booking.outbound_search_failed",
            )
            returning = executor.submit(
                _safe_results,
                lambda: self._search_leg(trip, outbound=False),
                "booking.return_search_failed",
            )
            hotels = executor.submit(
                _safe_results,
                lambda: self._search_hotels(trip) if trip.nights > 0 else [],
                "booking.hotel_search_failed",
            )

            return outbound.result(), returning.result(), hotels.result()

    def _search_leg(self, trip: BookingTrip, *, outbound: bool) -> list[FlightResult]:
        return self._flights.search_flights(
            origin=trip.origin_country if outbound else trip.destination_country,
            destination=trip.destination_country if outbound else trip.origin_country,
            departure_date=trip.start_date if outbound else trip.end_date,
            currency=trip.currency,
            max_budget=None,
            adults=trip.adults,
        )

    def _search_hotels(self, trip: BookingTrip) -> list[HotelResult]:
        return self._hotels.search_hotels(
            destination=trip.hotel_destination,
            check_in=trip.start_date,
            check_out=trip.end_date,
            currency=trip.currency,
            max_price_per_night=None,
            adults=trip.adults,
            destination_country_code=trip.destination_country_code,
        )

    @staticmethod
    def _reusable(
        trip: BookingTrip,
        evidence: Mapping[str, Any],
    ) -> tuple[FlightResult, FlightResult, HotelResult | None] | None:
        """Validate reused evidence without any provider call."""

        try:
            outbound = FlightResult.model_validate(evidence.get("outbound_flight"))
            returning = FlightResult.model_validate(evidence.get("return_flight"))
            raw_hotel = evidence.get("hotel") or None
            hotel = HotelResult.model_validate(raw_hotel) if raw_hotel else None

        except (ValidationError, TypeError, AttributeError):
            return None

        def iata(airport: Any) -> str | None:
            return normalise_iata_code(getattr(airport, "id", None))

        # The two legs must be one round trip between the same airports.
        origin = iata(outbound.departure_airport)
        destination = iata(outbound.arrival_airport)

        if (
            outbound.price <= 0
            or returning.price <= 0
            or origin is None
            or destination is None
            or iata(returning.departure_airport) != destination
            or iata(returning.arrival_airport) != origin
        ):
            return None

        if trip.nights == 0:
            return (outbound, returning, None) if hotel is None else None

        if (
            hotel is None
            or hotel.price_per_night <= 0
            or hotel.location is None
            or hotel.location.country_code != trip.destination_country_code
            or normalise_hotel_coordinates(hotel.location.model_dump()) is None
        ):
            return None

        return outbound, returning, hotel

    def _flight_alternatives(
        self,
        trip: BookingTrip,
        *,
        day: int,
        itinerary: Sequence[Mapping[str, Any]],
        leg: Leg | None,
    ) -> dict[str, Any]:
        days = _day_numbers(itinerary)
        first, last = (min(days), max(days)) if days else (1, trip.nights + 1)

        if day not in {first, last}:
            return {"error": "Flight alternatives are only valid on the first or last day."}

        outbound = leg != "return" if first == last else day == first
        current_outbound = _flight_legs(_day(itinerary, first) or {})[:1]
        current_return = _flight_legs(_day(itinerary, last) or {})[-1:]
        current = current_outbound if outbound else current_return
        kept = current_return if outbound else current_outbound

        # The transport budget covers BOTH legs, so this leg may only use what
        # the kept leg leaves over (never negative: then nothing fits).
        cap = budget_cap(trip.transportation_budget)

        if cap is not None and kept:
            cap = max(cap - kept[0].price, 0.0)

        options = _safe_results(
            lambda: self._search_leg(trip, outbound=outbound),
            "booking.alternative_flight_failed",
        )

        found, within_budget = alternatives(
            options,
            price=lambda flight: flight.price,
            cap=cap,
            exclude=flight_identity(current[0]) if current else None,
            identity=flight_identity,
            limit=ALTERNATIVES_LIMIT,
        )

        return self._payload(
            found,
            within_budget=within_budget,
            empty_error="No alternative flights found for this date.",
            warning=(
                f"No flight fits the remaining transport budget "
                f"({cap or 0.0:.2f} {trip.currency}). These are the cheapest "
                "real options; warn the user and suggest reallocating the "
                "budget or changing dates."
            ),
        )

    def _hotel_alternatives(
        self,
        trip: BookingTrip,
        *,
        itinerary: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if trip.nights == 0:
            return {"error": "A same-day trip has no hotel nights."}

        current = next(
            (
                day.get("hotel")
                for day in itinerary
                if isinstance(day, Mapping) and isinstance(day.get("hotel"), Mapping)
            ),
            None,
        )
        cap = nightly_cap(trip.accommodation_budget, trip.nights)

        options = _safe_results(
            lambda: self._search_hotels(trip),
            "booking.alternative_hotel_failed",
        )

        found, within_budget = alternatives(
            options,
            price=lambda hotel: hotel.price_per_night,
            cap=cap,
            exclude=hotel_identity(current) if current is not None else None,
            identity=hotel_identity,
            limit=ALTERNATIVES_LIMIT,
        )

        return self._payload(
            found,
            within_budget=within_budget,
            empty_error="No alternative hotels found for these dates.",
            warning=(
                f"No hotel fits {cap or 0.0:.2f} {trip.currency}/night. These are the "
                "cheapest real options; warn the user and suggest reallocating "
                "the budget or changing dates."
            ),
        )

    @staticmethod
    def _payload(
        options: Sequence[FlightResult | HotelResult],
        *,
        within_budget: bool,
        empty_error: str,
        warning: str,
    ) -> dict[str, Any]:
        if not options:
            return {
                "error": empty_error,
                "suggestion": "Consider adjusting the dates or the destination.",
            }

        payload: dict[str, Any] = {
            "options": [option.model_dump(mode="json") for option in options],
            "count": len(options),
        }

        if not within_budget:
            payload["warning"] = warning

        return payload


__all__ = [
    "ALTERNATIVES_LIMIT",
    "BookingPlanningService",
    "Leg",
]
