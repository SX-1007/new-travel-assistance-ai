"""Atomic itinerary edits against trusted search results."""

import copy

import pytest

from app.modules.booking.models import FlightAirport, FlightResult, HotelLocation, HotelResult
from app.modules.booking.selection import build_skeleton
from app.modules.itinerary.editing import (
    ItineraryEdit,
    TrustedBookingOption,
    apply_edits,
)


def flight(number, price, origin="KUL", destination="HND", time="2026-11-01 08:00"):
    return FlightResult(
        airline="AirTest",
        flight_number=number,
        price=price,
        departure_airport=FlightAirport(id=origin, time=time),
        arrival_airport=FlightAirport(id=destination),
    )


def hotel(name, price, lat=35.69):
    return HotelResult(
        hotel_name=name,
        price_per_night=price,
        location=HotelLocation(lat=lat, lng=139.69, country_code="JP"),
    )


def place(name, lat, cost=10.0):
    return {
        "name": name,
        "type": "attraction",
        "description": ["Temple", "Landmark"],
        "address": "Tokyo",
        "estimated_cost": cost,
        "is_estimated": True,
        "location": {
            "place_name": name,
            "latitude": lat,
            "longitude": 139.79,
            "country_code": "JP",
            "requested_city": "Tokyo",
            "verified_locality": "Tokyo",
        },
    }


def itinerary(nights=2):
    days = build_skeleton(
        start_date="2026-11-01",
        nights=nights,
        outbound=flight("T1", 300),
        returning=flight("R1", 250, "HND", "KUL", "2026-11-03 20:00"),
        hotel=hotel("Budget Inn", 80),
    )
    days[0]["activities"] = [dict(place("Senso-ji", 35.71), order=1)]
    days[0]["day_total_cost"] = 390.0
    return days


NEW_OUT = flight("T9", 500).model_dump(mode="json")
NEW_HOTEL = hotel("Grand", 200, 35.70).model_dump(mode="json")


def test_flight_replace_stores_trusted_payload_and_recosts():
    retyped = copy.deepcopy(NEW_OUT)
    retyped["airline_logo"] = "assistant-invented"

    outcome = apply_edits(
        itinerary(),
        [{"day": 1, "action": "replace", "category": "flight", "new_details": retyped}],
        trusted_bookings=[TrustedBookingOption("flight", 1, NEW_OUT)],
        transportation_budget=600,
    )

    day1 = outcome.itinerary[0]
    assert outcome.applied and outcome.affected_days == (1,)
    assert day1["flight"][0]["flight_number"] == "T9"
    assert day1["flight"][0]["airline_logo"] == ""
    assert day1["day_total_cost"] == 590.0
    # 500 + 250 > 600 → both boundary legs flagged.
    assert day1["flight"][0]["over_budget"] and outcome.itinerary[-1]["flight"][0]["over_budget"]


def test_untrusted_or_middle_day_flight_is_refused_and_nothing_changes():
    original = itinerary()

    refused = apply_edits(
        original,
        [{"day": 1, "action": "replace", "category": "flight", "new_details": NEW_OUT}],
    )
    middle = apply_edits(
        original,
        [{"day": 2, "action": "replace", "category": "flight", "new_details": NEW_OUT}],
        trusted_bookings=[TrustedBookingOption("flight", 2, NEW_OUT)],
    )

    assert not refused.applied and "search" in refused.error
    assert not middle.applied and "first or last day" in middle.error
    assert refused.itinerary == original == itinerary()


def test_hotel_replace_covers_every_night():
    outcome = apply_edits(
        itinerary(),
        [{"day": 2, "action": "replace", "category": "hotel", "new_details": NEW_HOTEL}],
        trusted_bookings=[TrustedBookingOption("hotel", 2, NEW_HOTEL)],
    )

    names = [d["hotel"]["hotel_name"] if d["hotel"] else None for d in outcome.itinerary]
    assert names == ["Grand", "Grand", None]
    assert [d["day_total_cost"] for d in outcome.itinerary] == [510.0, 200.0, 250.0]


def test_add_place_uses_trusted_result_and_normalises_it():
    trusted = [place("Tokyo Tower", 35.6586, cost=-4)]
    retyped = place("tokyo  tower", 35.66)

    outcome = apply_edits(
        itinerary(),
        [{"day": 1, "action": "add", "category": "restaurant", "new_details": retyped}],
        trusted_places=trusted,
    )

    added = outcome.itinerary[0]["activities"][1]
    assert added["name"] == "Tokyo Tower"
    assert added["location"]["latitude"] == 35.6586
    assert added["type"] == "restaurant"
    assert added["description"] == "Temple, Landmark"
    assert added["estimated_cost"] == 0.0
    assert added["order"] == 2


def test_same_name_places_are_told_apart_by_position():
    trusted = [place("Starbucks", 35.60), place("Starbucks", 35.70)]

    outcome = apply_edits(
        itinerary(),
        [{"day": 2, "action": "add", "category": "restaurant", "new_details": place("Starbucks", 35.70004)}],
        trusted_places=trusted,
    )

    assert outcome.itinerary[1]["activities"][0]["location"]["latitude"] == 35.70


def test_place_replace_and_remove_need_authorised_day_and_valid_index():
    trusted = [place("Tokyo Tower", 35.6586)]
    replace = {"day": 1, "action": "replace", "category": "activity", "index": 1,
               "new_details": place("Tokyo Tower", 35.6586)}

    unauthorised = apply_edits(itinerary(), [replace], trusted_places=trusted,
                               is_day_authorized=lambda day: False)
    out_of_range = apply_edits(itinerary(), [{**replace, "index": 5}], trusted_places=trusted)
    replaced = apply_edits(itinerary(), [replace], trusted_places=trusted)
    removed = apply_edits(itinerary(), [{"day": 1, "action": "remove", "category": "activity", "index": 1}])

    assert not unauthorised.applied and "which day" in unauthorised.error
    assert not out_of_range.applied and "does not exist" in out_of_range.error
    assert replaced.itinerary[0]["activities"][0]["name"] == "Tokyo Tower"
    assert removed.itinerary[0]["activities"] == []
    assert removed.itinerary[0]["day_total_cost"] == 380.0


def test_batch_is_atomic():
    trusted = [place("Tokyo Tower", 35.6586)]

    outcome = apply_edits(
        itinerary(),
        [
            {"day": 2, "action": "add", "category": "activity", "new_details": place("Tokyo Tower", 35.6586)},
            {"day": 9, "action": "remove", "category": "activity", "index": 1},
        ],
        trusted_places=trusted,
    )

    assert not outcome.applied and "Day 9" in outcome.error
    assert outcome.itinerary[1]["activities"] == []


def test_same_day_trip_replaces_the_matching_leg():
    days = itinerary(nights=0)
    new_return = flight("R9", 260, "HND", "KUL", "2026-11-01 21:00").model_dump(mode="json")

    outcome = apply_edits(
        days,
        [{"day": 1, "action": "replace", "category": "flight", "new_details": new_return}],
        trusted_bookings=[TrustedBookingOption("flight", 1, new_return)],
    )

    assert [f["flight_number"] for f in outcome.itinerary[0]["flight"]] == ["T1", "R9"]


@pytest.mark.parametrize(
    "bad",
    [
        {"day": 1, "action": "remove", "category": "flight"},
        {"day": 1, "action": "add", "category": "activity"},
        {"day": 1, "action": "remove", "category": "activity"},
        {"day": 0, "action": "remove", "category": "activity", "index": 1},
    ],
)
def test_malformed_edits_are_rejected(bad):
    outcome = apply_edits(itinerary(), [bad])
    assert not outcome.applied


def test_schema_rejects_booking_removal():
    with pytest.raises(ValueError):
        ItineraryEdit(day=1, action="remove", category="hotel")


def test_no_edits():
    assert not apply_edits(itinerary(), []).applied
