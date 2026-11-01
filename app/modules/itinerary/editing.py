"""Pure, atomic itinerary edits requested from chat.

``apply_edits`` takes the current itinerary, the requested edits and the
provider results the assistant actually saw this turn, and returns either a
new itinerary with every edit applied or the unchanged one with a reason.
Nothing is mutated in place and nothing is half-applied.

Trust rules:

* a new flight, hotel or place must come from a fresh provider result
  (``trusted_*``); the stored payload is always the trusted provider object,
  never the assistant's re-typed copy;
* removing or replacing a place needs the day to be authorised by the caller
  (the user must have stated it); the caller decides how.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.modules.booking.flights import normalise_iata_code
from app.modules.booking.hotels import hotel_check_in
from app.modules.booking.models import FlightResult, HotelResult
from app.modules.booking.selection import flight_identity, hotel_identity, money
from app.modules.itinerary.models import ActivityResult


# 1e-4 degrees ≈ 11 m: tolerates the assistant rounding coordinates while
# still separating two same-named places.
_COORDINATE_TOLERANCE = 1e-4

EditCategory = Literal["flight", "hotel", "activity", "restaurant"]


class ItineraryEdit(BaseModel):
    """One change to apply to the itinerary (the ``edit_itinerary`` schema)."""

    day: int = Field(
        ge=1,
        description=(
            "Day number to edit (1-based). For activity/restaurant remove or "
            "replace, this day MUST be stated in the user's current message."
        ),
    )
    action: Literal["add", "remove", "replace"] = Field(
        description="'add' a new item, 'remove' an existing one, or 'replace' one."
    )
    category: EditCategory = Field(
        description="'activity' and 'restaurant' live in the day's activities list.",
    )
    index: int | None = Field(
        default=None,
        ge=1,
        description=(
            "For activity/restaurant remove/replace: the 1-based number shown "
            "in that day's activities list. Ignored for flight/hotel."
        ),
    )
    new_details: dict[str, Any] | None = Field(
        default=None,
        description=(
            "For add/replace: the full object from search_alternative_opt "
            "(flight/hotel) or search_places (activity/restaurant)."
        ),
    )

    @model_validator(mode="after")
    def _shape_matches_action(self) -> "ItineraryEdit":
        booking = self.category in {"flight", "hotel"}

        if booking and self.action != "replace":
            raise ValueError("Flights and hotels can only be replaced.")

        if self.action in {"add", "replace"} and not self.new_details:
            raise ValueError("add/replace needs new_details from a search result.")

        if not booking and self.action in {"remove", "replace"} and self.index is None:
            raise ValueError("remove/replace of a place needs its 1-based index.")

        return self


@dataclass(frozen=True, slots=True)
class TrustedBookingOption:
    """One flight/hotel option returned to the assistant this turn."""

    category: Literal["flight", "hotel"]
    day: int
    option: Mapping[str, Any]


# Why a batch was refused, for callers that react differently per reason.
RejectionCode = Literal[
    "no_edits",
    "invalid_edit",
    "unknown_day",
    "untrusted_booking",
    "untrusted_place",
    "incomplete_result",
    "flight_day",
    "no_hotel_nights",
    "day_not_authorized",
    "unknown_item",
]


@dataclass(frozen=True, slots=True)
class EditOutcome:
    itinerary: list[dict[str, Any]]
    applied: bool
    error: str | None = None
    affected_days: tuple[int, ...] = ()
    code: RejectionCode | None = None


class _Rejected(Exception):
    """Internal: one edit cannot be applied; the whole batch is refused."""

    def __init__(self, message: str, *, code: RejectionCode) -> None:
        super().__init__(message)
        self.code = code


# ── Public API ──────────────────────────────────────────────────────


def apply_edits(
    itinerary: Sequence[Mapping[str, Any]],
    edits: Sequence[ItineraryEdit | Mapping[str, Any]],
    *,
    trusted_places: Sequence[Mapping[str, Any]] = (),
    trusted_bookings: Sequence[TrustedBookingOption] = (),
    is_day_authorized: Callable[[int], bool] = lambda day: True,
    transportation_budget: float = 0.0,
) -> EditOutcome:
    """Apply every edit or none of them."""

    original = [dict(day) for day in itinerary if isinstance(day, Mapping)]

    if not edits:
        return EditOutcome(itinerary=original, applied=False, error="No edits were requested.", code="no_edits")

    working = copy.deepcopy(original)
    affected: set[int] = set()

    try:
        for raw in edits:
            edit = _parse(raw)
            _apply_one(
                working,
                edit,
                trusted_places=trusted_places,
                trusted_bookings=trusted_bookings,
                is_day_authorized=is_day_authorized,
                affected=affected,
            )

    except _Rejected as exc:
        return EditOutcome(
            itinerary=original,
            applied=False,
            error=f"{exc} Your itinerary has not been changed.",
            code=exc.code,
        )

    reassess_flight_budget(working, transportation_budget)

    for day in working:
        if day.get("day") in affected:
            day["day_total_cost"] = day_cost(day, is_checkout=_is_checkout(working, day))

    return EditOutcome(
        itinerary=working,
        applied=True,
        affected_days=tuple(sorted(affected)),
    )


def day_cost(day: Mapping[str, Any], *, is_checkout: bool) -> float:
    """Flights + overnight hotel + estimated place costs for one day."""

    flights = day.get("flight")
    legs = flights if isinstance(flights, list) else [flights] if isinstance(flights, Mapping) else []
    hotel = day.get("hotel")
    activities = day.get("activities")

    total = sum(money(leg.get("price")) for leg in legs if isinstance(leg, Mapping))

    if isinstance(hotel, Mapping) and not is_checkout:
        total += money(hotel.get("price_per_night"))

    if isinstance(activities, list):
        total += sum(
            money(activity.get("estimated_cost"))
            for activity in activities
            if isinstance(activity, Mapping)
        )

    return round(total, 2)


def reassess_flight_budget(itinerary: list[dict[str, Any]], cap: float) -> None:
    """Re-flag the boundary legs ``over_budget`` against the shared cap."""

    legs = [
        leg
        for day in _boundary_days(itinerary)
        for leg in (day.get("flight") or [])
        if isinstance(leg, dict)
    ]

    for leg in legs:
        leg.pop("over_budget", None)

    if cap > 0 and sum(money(leg.get("price")) for leg in legs) > cap:
        for leg in legs:
            leg["over_budget"] = True


# ── Internals ───────────────────────────────────────────────────────


def _parse(raw: ItineraryEdit | Mapping[str, Any]) -> ItineraryEdit:
    if isinstance(raw, ItineraryEdit):
        return raw

    try:
        return ItineraryEdit.model_validate(raw)

    except ValidationError as exc:
        reason = exc.errors()[0].get("msg", "invalid edit") if exc.errors() else "invalid edit"
        raise _Rejected(f"That change is not valid ({reason}).", code="invalid_edit") from exc


def _boundary_days(itinerary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not itinerary:
        return []

    first = min(itinerary, key=lambda day: day.get("day", math.inf))
    last = max(itinerary, key=lambda day: day.get("day", -math.inf))

    return [first] if first is last else [first, last]


def _is_checkout(itinerary: list[dict[str, Any]], day: Mapping[str, Any]) -> bool:
    numbers = [d.get("day") for d in itinerary if isinstance(d.get("day"), int)]
    return len(numbers) > 1 and day.get("day") == max(numbers)


def _find_day(itinerary: list[dict[str, Any]], number: int) -> dict[str, Any]:
    for day in itinerary:
        if day.get("day") == number:
            return day

    raise _Rejected(f"Day {number} does not exist in your itinerary.", code="unknown_day")


def _apply_one(
    itinerary: list[dict[str, Any]],
    edit: ItineraryEdit,
    *,
    trusted_places: Sequence[Mapping[str, Any]],
    trusted_bookings: Sequence[TrustedBookingOption],
    is_day_authorized: Callable[[int], bool],
    affected: set[int],
) -> None:
    day = _find_day(itinerary, edit.day)

    if edit.category == "flight":
        affected.update(_replace_flight(itinerary, day, edit, trusted_bookings))
    elif edit.category == "hotel":
        affected.update(_replace_hotel(itinerary, edit, trusted_bookings))
    else:
        _edit_place(day, edit, trusted_places, is_day_authorized)
        affected.add(edit.day)


def _trusted_booking(
    edit: ItineraryEdit,
    trusted: Sequence[TrustedBookingOption],
    identity: Callable[[Mapping[str, Any]], tuple[Any, ...]],
) -> dict[str, Any]:
    wanted = identity(edit.new_details or {})

    for option in trusted:
        if option.category == edit.category and option.day == edit.day:
            if identity(option.option) == wanted:
                return copy.deepcopy(dict(option.option))

    raise _Rejected(
        f"That {edit.category} was not in this turn's search results; "
        "search for alternatives first.",
        code="untrusted_booking",
    )


def _replace_flight(
    itinerary: list[dict[str, Any]],
    day: dict[str, Any],
    edit: ItineraryEdit,
    trusted: Sequence[TrustedBookingOption],
) -> set[int]:
    boundary = _boundary_days(itinerary)

    if day not in boundary:
        raise _Rejected("Flights can only change on the first or last day.", code="flight_day")

    payload = _trusted_booking(edit, trusted, flight_identity)

    try:
        flight = FlightResult.model_validate(payload)
    except ValidationError as exc:
        raise _Rejected("That flight result is incomplete.", code="incomplete_result") from exc

    if flight.price <= 0 or not flight.airline.strip() or not flight.flight_number.strip():
        raise _Rejected("That flight result is incomplete.", code="incomplete_result")

    legs = [leg for leg in (day.get("flight") or []) if isinstance(leg, dict)]
    replacement = flight.model_dump(mode="json")

    if len(legs) <= 1:
        day["flight"] = [replacement]
    else:
        # Same-day trip: replace the leg that departs from the same airport.
        departure = normalise_iata_code((flight.departure_airport.id if flight.departure_airport else None))
        position = next(
            (
                index
                for index, leg in enumerate(legs)
                if normalise_iata_code((leg.get("departure_airport") or {}).get("id")) == departure
            ),
            0,
        )
        legs[position] = replacement
        day["flight"] = legs

    # A new outbound flight can land on another night: move the hotel nights.
    return {day["day"]} | _sync_hotel_nights(itinerary)


def _replace_hotel(
    itinerary: list[dict[str, Any]],
    edit: ItineraryEdit,
    trusted: Sequence[TrustedBookingOption],
) -> set[int]:
    payload = _trusted_booking(edit, trusted, hotel_identity)

    try:
        hotel = HotelResult.model_validate(payload)
    except ValidationError as exc:
        raise _Rejected("That hotel result is incomplete.", code="incomplete_result") from exc

    if hotel.price_per_night <= 0 or not hotel.hotel_name.strip() or hotel.location is None:
        raise _Rejected("That hotel result is incomplete.", code="incomplete_result")

    # Alternatives are searched and priced for the whole stay, so the new
    # hotel replaces every overnight day, not just the day mentioned.
    overnight = _hotel_night_days(itinerary)

    if not overnight or len(itinerary) == 1:
        raise _Rejected("This trip has no hotel nights.", code="no_hotel_nights")

    replacement = hotel.model_dump(mode="json")

    for day in overnight:
        day["hotel"] = copy.deepcopy(replacement)

    return {day["day"] for day in overnight}


def _hotel_night_days(itinerary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Days that need a hotel: not checkout, and not before the outbound
    flight lands (``hotel_check_in``)."""

    dated = sorted(
        (day for day in itinerary if isinstance(day.get("date"), str)),
        key=lambda day: day["date"],
    )

    if not dated:
        return []

    first_legs = _boundary_days(itinerary)[0].get("flight") or []
    outbound = first_legs[0] if isinstance(first_legs, list) and first_legs else None
    first_night = hotel_check_in(dated[0]["date"], dated[-1]["date"], outbound)

    return [
        day
        for day in itinerary
        if not _is_checkout(itinerary, day) and str(day.get("date", "")) >= first_night
    ]


def _sync_hotel_nights(itinerary: list[dict[str, Any]]) -> set[int]:
    """Add or remove the stay's hotel so exactly the hotel-night days carry
    it; returns the days that changed."""

    stay = next((day["hotel"] for day in itinerary if isinstance(day.get("hotel"), Mapping)), None)

    if stay is None:
        return set()

    nights = {id(day) for day in _hotel_night_days(itinerary)}
    changed: set[int] = set()

    for day in itinerary:
        has_hotel = isinstance(day.get("hotel"), Mapping)

        if id(day) in nights and not has_hotel:
            day["hotel"] = copy.deepcopy(dict(stay))
            changed.add(day["day"])
        elif id(day) not in nights and has_hotel:
            day["hotel"] = None
            changed.add(day["day"])

    return changed


def _same_place(details: Mapping[str, Any], option: Mapping[str, Any]) -> bool:
    location = details.get("location") or {}
    option_location = option.get("location") or {}

    try:
        return math.isclose(
            float(location.get("latitude")),
            float(option_location.get("latitude")),
            abs_tol=_COORDINATE_TOLERANCE,
        ) and math.isclose(
            float(location.get("longitude")),
            float(option_location.get("longitude")),
            abs_tol=_COORDINATE_TOLERANCE,
        )

    except (TypeError, ValueError):
        return False


def _trusted_place(
    details: Mapping[str, Any],
    trusted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The fresh search result the assistant picked (by name, then position)."""

    name_key = " ".join(str(details.get("name") or "").split()).casefold()
    same_name = [
        option
        for option in trusted
        if isinstance(option, Mapping)
        and " ".join(str(option.get("name") or "").split()).casefold() == name_key
    ]

    if len(same_name) == 1:
        return copy.deepcopy(dict(same_name[0]))

    for option in same_name:
        if _same_place(details, option):
            return copy.deepcopy(dict(option))

    raise _Rejected(
        "That place was not in this turn's search results; search for it first.",
        code="untrusted_place",
    )


def _normalised_place(option: dict[str, Any], category: EditCategory) -> dict[str, Any]:
    """Coerce a search result into the strict public activity contract."""

    place = dict(option)
    place["type"] = "restaurant" if category == "restaurant" else "attraction"

    description = place.get("description")
    if isinstance(description, (list, tuple, set)):
        description = ", ".join(str(item).strip() for item in description if str(item).strip())
    place["description"] = (description.strip() or None) if isinstance(description, str) else None

    place["estimated_cost"] = round(money(place.get("estimated_cost")), 2)
    place["is_estimated"] = bool(place.get("is_estimated", True))
    place["order"] = 1

    try:
        ActivityResult.model_validate(place)
    except ValidationError as exc:
        raise _Rejected("That place result is incomplete.", code="incomplete_result") from exc

    return place


def _edit_place(
    day: dict[str, Any],
    edit: ItineraryEdit,
    trusted: Sequence[Mapping[str, Any]],
    is_day_authorized: Callable[[int], bool],
) -> None:
    activities = [a for a in (day.get("activities") or []) if isinstance(a, dict)]

    if edit.action in {"remove", "replace"}:
        if not is_day_authorized(edit.day):
            raise _Rejected(
                "Please tell me which day you mean before I change that activity.",
                code="day_not_authorized",
            )

        if edit.index is None or edit.index > len(activities):
            raise _Rejected(
                f"Day {edit.day} has {len(activities)} activities, so activity "
                f"{edit.index} does not exist.",
                code="unknown_item",
            )

    if edit.action in {"add", "replace"}:
        place = _normalised_place(_trusted_place(edit.new_details or {}, trusted), edit.category)

        if edit.action == "add":
            activities.append(place)
        else:
            activities[edit.index - 1] = place  # type: ignore[operator]
    else:
        activities.pop(edit.index - 1)  # type: ignore[operator]

    for order, activity in enumerate(activities, start=1):
        activity["order"] = order

    day["activities"] = activities


__all__ = [
    "EditOutcome",
    "RejectionCode",
    "ItineraryEdit",
    "TrustedBookingOption",
    "apply_edits",
    "day_cost",
    "reassess_flight_budget",
]
