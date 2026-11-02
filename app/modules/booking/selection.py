"""Pure flight/hotel selection and itinerary-skeleton rules.

Provider results arrive sorted cheapest-first; nothing here calls a provider.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final, TypeVar

from app.modules.booking.flight_timing import flight_time, trip_free_hours
from app.modules.booking.flights import normalise_iata_code
from app.modules.booking.hotels import hotel_check_in
from app.modules.booking.models import FlightResult, HotelResult


_Option = TypeVar("_Option", FlightResult, HotelResult)

# Flight choice: a pair may cost up to 15 % more than the cheapest pair if
# it leaves more time at the destination ...
PRICE_TOLERANCE: Final = 0.15
# ... but only if it adds more than an hour; within an hour of the best
# time, the cheapest pair wins.
TIME_SLACK_HOURS: Final = 1.0


def money(value: Any) -> float:
    """A finite non-negative amount; anything else counts as 0."""

    if isinstance(value, bool):
        return 0.0

    try:
        number = float(value)

    except (TypeError, ValueError, OverflowError):
        return 0.0

    return number if math.isfinite(number) and number > 0 else 0.0


# ── Identities (used to exclude the current booking and match options) ──


def flight_identity(flight: FlightResult | Mapping[str, Any]) -> tuple[str, ...]:
    """Airline + flight number + departure airport/time."""

    data = flight.model_dump() if isinstance(flight, FlightResult) else dict(flight)
    departure = data.get("departure_airport") or {}

    return (
        " ".join(str(data.get("airline") or "").split()).casefold(),
        "".join(str(data.get("flight_number") or "").split()).upper(),
        normalise_iata_code(departure.get("id")) or "",
        str(departure.get("time") or data.get("departure_time") or "").strip(),
    )


def hotel_identity(hotel: HotelResult | Mapping[str, Any]) -> tuple[Any, ...]:
    """Hotel name + coordinates rounded to ~10 m."""

    data = hotel.model_dump() if isinstance(hotel, HotelResult) else dict(hotel)
    location = data.get("location") or {}

    def coordinate(*keys: str) -> float | None:
        for key in keys:
            value = location.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return round(float(value), 4)
        return None

    return (
        " ".join(str(data.get("hotel_name") or "").split()).casefold(),
        coordinate("lat", "latitude"),
        coordinate("lng", "longitude"),
    )


# ── Budgets ─────────────────────────────────────────────────────────


def transportation_budget(allocation: Mapping[str, Any] | None) -> float:
    """The transportation cap; sessions saved by earlier versions call it "flight"."""

    if not isinstance(allocation, Mapping):
        return 0.0

    key = "transportation" if "transportation" in allocation else "flight"
    return money(allocation.get(key))


def accommodation_budget(allocation: Mapping[str, Any] | None) -> float:
    if not isinstance(allocation, Mapping):
        return 0.0

    return money(allocation.get("accommodation"))


def budget_cap(amount: float) -> float | None:
    """An allocated amount as a cap; an unallocated (0) category means no cap."""

    return amount if amount > 0 else None


def nightly_cap(accommodation: float, nights: int) -> float | None:
    """Per-night hotel cap, or None for "no cap"."""

    return accommodation / nights if nights > 0 and accommodation > 0 else None


# ── Selection ───────────────────────────────────────────────────────


def pick_hotel(
    options: Sequence[HotelResult],
    cap_per_night: float | None,
) -> HotelResult | None:
    """Cheapest hotel within the nightly cap.

    If none fits, the cheapest overall is returned flagged ``over_budget``
    so the trip shows the shortfall instead of an empty hotel slot.
    """

    if not options:
        return None

    if cap_per_night is None:
        return options[0]

    for hotel in options:
        if hotel.price_per_night <= cap_per_night:
            return hotel

    return options[0].model_copy(update={"over_budget": True})


@dataclass(frozen=True, slots=True)
class _Pair:
    outbound: FlightResult | None
    returning: FlightResult | None
    # Usable hours at the destination; None when a leg's time is unreadable.
    hours: float | None

    @property
    def legs(self) -> tuple[FlightResult, ...]:
        return tuple(leg for leg in (self.outbound, self.returning) if leg is not None)

    @property
    def price(self) -> float:
        return sum(leg.price for leg in self.legs)

    @property
    def stops(self) -> int:
        return sum(leg.stops for leg in self.legs)

    @property
    def duration(self) -> int:
        return sum(leg.duration for leg in self.legs)


def pick_flight_pair(
    outbound_options: Sequence[FlightResult],
    return_options: Sequence[FlightResult],
    *,
    start_date: str,
    end_date: str,
    cap: float | None = None,
) -> tuple[FlightResult | None, FlightResult | None]:
    """The outbound + return pair that leaves the most time at the destination,
    among pairs costing at most ``PRICE_TOLERANCE`` more than the cheapest.

    Time is the trip's usable hours (``flight_timing``): an outbound flight
    landing by about noon and a return leaving in the evening score best.
    Pairs within ``TIME_SLACK_HOURS`` of the best time count as equally
    good, and the cheapest of those wins, so a few minutes never cost extra.
    A flight whose time can't be read only wins when no pair's time can be.

    ``cap`` (the transport budget, None for none) also bounds the choice.
    When even the cheapest pair is over it, that pair is returned with both
    legs flagged ``over_budget``. The budget check and the plan both choose
    with this rule, so the plan can reuse the budget check's flights.
    """

    if not outbound_options and not return_options:
        return None, None

    # A missing leg (None) is priced 0 and imposes no time limit.
    outbound_list: list[FlightResult | None] = list(outbound_options) or [None]
    return_list: list[FlightResult | None] = list(return_options) or [None]

    arrivals = [flight_time(leg, "arrival") if leg is not None else None for leg in outbound_list]
    departures = [flight_time(leg, "departure") if leg is not None else None for leg in return_list]
    pairs: list[_Pair] = []

    for out_index, outbound in enumerate(outbound_list):
        for return_index, returning in enumerate(return_list):
            unreadable = (outbound is not None and arrivals[out_index] is None) or (
                returning is not None and departures[return_index] is None
            )
            hours = None if unreadable else trip_free_hours(
                start_date,
                end_date,
                arrival=arrivals[out_index],
                departure=departures[return_index],
            )
            pairs.append(_Pair(outbound, returning, hours))

    cheapest = min(pair.price for pair in pairs)
    ceiling = cheapest * (1 + PRICE_TOLERANCE)

    if cap is not None:
        # Over budget either way: the smallest shortfall is the cheapest pair.
        ceiling = min(ceiling, cap) if cheapest <= cap else cheapest

    candidates = [pair for pair in pairs if pair.price <= ceiling]
    known = [pair.hours for pair in candidates if pair.hours is not None]

    if known:
        enough = max(known) - TIME_SLACK_HOURS
        candidates = [pair for pair in candidates if pair.hours is not None and pair.hours >= enough]

    # Ties: more time, then fewer stops, then a shorter journey. min() keeps
    # the provider's (cheapest-first) order among exact ties.
    chosen = min(
        candidates,
        key=lambda pair: (pair.price, -(pair.hours or 0.0), pair.stops, pair.duration),
    )
    outbound, returning = chosen.outbound, chosen.returning

    if cap is not None and chosen.price > cap:
        outbound = outbound.model_copy(update={"over_budget": True}) if outbound else None
        returning = returning.model_copy(update={"over_budget": True}) if returning else None

    return outbound, returning


def alternatives(
    options: Sequence[_Option],
    *,
    price: Callable[[_Option], float],
    cap: float | None,
    exclude: tuple[Any, ...] | None,
    identity: Callable[[_Option], tuple[Any, ...]],
    limit: int,
) -> tuple[list[_Option], bool]:
    """Up to ``limit`` options other than the current booking.

    ``cap`` None means no cap. Returns (options, within_budget). When nothing
    fits the cap, the cheapest real options are returned flagged
    ``over_budget`` and within_budget=False.
    """

    candidates = [
        option for option in options if exclude is None or identity(option) != exclude
    ]

    if cap is None:
        return candidates[:limit], True

    within = [option for option in candidates if price(option) <= cap]

    if within:
        return within[:limit], True

    return [
        option.model_copy(update={"over_budget": True})
        for option in candidates[:limit]
    ], False


# ── Skeleton ────────────────────────────────────────────────────────


def trip_dates(start_date: str, nights: int) -> list[str]:
    start = date.fromisoformat(start_date)
    return [(start + timedelta(days=offset)).isoformat() for offset in range(nights + 1)]


def build_skeleton(
    *,
    start_date: str,
    nights: int,
    outbound: FlightResult | None,
    returning: FlightResult | None,
    hotel: HotelResult | None,
) -> list[dict[str, Any]]:
    """One day per date: boundary flights, the hotel on overnight days,
    empty activity slots, and each day's cost.

    The last day is checkout, so it has no hotel and no hotel charge. Nights
    before the outbound flight lands (``hotel_check_in``) have no hotel
    either. A same-day trip carries both legs on day 1.
    """

    days: list[dict[str, Any]] = []
    last_day = nights + 1
    dates = trip_dates(start_date, nights)
    first_night = hotel_check_in(start_date, dates[-1], outbound)

    for day_number, day_date in enumerate(dates, start=1):
        legs: list[FlightResult] = []

        if day_number == 1 and outbound is not None:
            legs.append(outbound)

        if day_number == last_day and returning is not None:
            legs.append(returning)

        overnight = day_number <= nights and hotel is not None and day_date >= first_night

        days.append(
            {
                "day": day_number,
                "date": day_date,
                "flight": [leg.model_dump(mode="json") for leg in legs] or None,
                "hotel": hotel.model_dump(mode="json") if overnight else None,
                "activities": [],
                "route": None,
                "day_total_cost": round(
                    sum(leg.price for leg in legs)
                    + (hotel.price_per_night if overnight else 0.0),
                    2,
                ),
            }
        )

    return days


__all__ = [
    "PRICE_TOLERANCE",
    "TIME_SLACK_HOURS",
    "accommodation_budget",
    "alternatives",
    "budget_cap",
    "build_skeleton",
    "flight_identity",
    "hotel_identity",
    "money",
    "nightly_cap",
    "pick_flight_pair",
    "pick_hotel",
    "transportation_budget",
    "trip_dates",
]
