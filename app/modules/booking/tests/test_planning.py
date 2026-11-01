"""Booking planning and alternatives with fake flight/hotel providers."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.modules.booking.models import (
    BookingTrip,
    FlightAirport,
    FlightResult,
    HotelLocation,
    HotelResult,
)
from app.modules.booking.hotels import hotel_check_in
from app.modules.booking.selection import nightly_cap, pick_hotel
from app.modules.booking.service import BookingPlanningService
from app.modules.budget.assessment import build_assessment
from app.modules.budget.models import BudgetTrip
from app.modules.itinerary.models import DailyItinerary
from app.orchestration.workflows.booking import booking_trip_from_state, run_booking_planning


def flight(number, price, origin="KUL", destination="HND", time="2026-11-01 08:00"):
    return FlightResult(
        airline="AirTest",
        flight_number=number,
        price=price,
        departure_airport=FlightAirport(id=origin, name=f"{origin} airport", time=time),
        arrival_airport=FlightAirport(id=destination, name=f"{destination} airport"),
    )


def hotel(name, price, lat=35.69):
    return HotelResult(
        hotel_name=name,
        price_per_night=price,
        location=HotelLocation(lat=lat, lng=139.69, country_code="JP"),
    )


OUT = [flight("T1", 300), flight("T2", 450), flight("T3", 600)]
RET = [
    flight("R1", 250, "HND", "KUL", "2026-11-04 20:00"),
    flight("R2", 400, "HND", "KUL", "2026-11-04 22:00"),
]
HOTELS = [hotel("Budget Inn", 80, 35.60), hotel("Mid Hotel", 120, 35.61), hotel("Grand", 300, 35.62)]


class FakeFlights:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def search_flights(self, *, origin, destination, departure_date, currency, max_budget=None, adults=1):
        self.calls.append((origin, destination, departure_date, currency, max_budget))
        if self.error:
            raise self.error
        return list(OUT if origin == "Malaysia" else RET)


class FakeHotels:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def search_hotels(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return list(HOTELS)


def trip(start="2026-11-01", end="2026-11-04", transport=0.0, accommodation=0.0):
    return BookingTrip(
        origin_country="Malaysia",
        destination_country="Japan",
        destination_country_code="JP",
        city="Tokyo",
        start_date=start,
        end_date=end,
        currency="JPY",
        adults=2,
        transportation_budget=transport,
        accommodation_budget=accommodation,
    )


def service(flights=None, hotels=None):
    flights = flights or FakeFlights()
    hotels = hotels or FakeHotels()
    return BookingPlanningService(flights=flights, hotels=hotels), flights, hotels


# ── Skeleton ────────────────────────────────────────────────────────


def test_skeleton_from_search_matches_public_contract():
    planner, flights, hotels = service()

    result = planner.plan_skeleton(trip(transport=1000, accommodation=600))
    days = result.itinerary

    assert result.source == "search" and result.missing == ()
    assert [d["date"] for d in days] == ["2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04"]
    assert days[0]["flight"][0]["flight_number"] == "T1"
    assert days[-1]["flight"][0]["flight_number"] == "R1"
    assert [d["hotel"]["hotel_name"] if d["hotel"] else None for d in days] == [
        "Budget Inn", "Budget Inn", "Budget Inn", None,
    ]
    assert [d["day_total_cost"] for d in days] == [380.0, 80.0, 80.0, 250.0]
    for day in days:
        DailyItinerary.model_validate(day)
    # Searches are unfiltered; budget preference is applied afterwards.
    assert all(call[4] is None for call in flights.calls)
    assert hotels.calls[0]["destination"] == "Tokyo, Japan"


def long_flight(number, price, arrives):
    """KUL 19:15 on 1 Nov → HND at ``arrives`` (found live: a ~30 h journey)."""
    return FlightResult(
        airline="AirTest",
        flight_number=number,
        price=price,
        departure_airport=FlightAirport(id="KUL", name="KUL airport", time="2026-11-01 19:15"),
        arrival_airport=FlightAirport(id="HND", name="HND airport", time=arrives),
        departure_time="2026-11-01 19:15",
        arrival_time=arrives,
    )


@pytest.mark.parametrize(
    ("arrives", "check_in"),
    [
        ("2026-11-03 01:00", "2026-11-02"),  # small hours: the night before
        ("2026-11-03 06:00", "2026-11-03"),
        ("2026-11-01 23:30", "2026-11-01"),
        ("2026-10-31 23:00", "2026-11-01"),  # never before the trip
        ("2026-11-09 12:00", "2026-11-04"),  # never after it
        ("not a time", "2026-11-01"),
    ],
)
def test_hotel_check_in_is_the_landing_night(arrives, check_in):
    assert hotel_check_in("2026-11-01", "2026-11-04", long_flight("L", 1, arrives)) == check_in


def test_no_hotel_is_booked_before_the_outbound_flight_lands():
    class LongFlights(FakeFlights):
        def search_flights(self, *, origin, **kwargs):
            if origin == "Malaysia":
                return [long_flight("L1", 300, "2026-11-03 01:00")]
            return super().search_flights(origin=origin, **kwargs)

    planner, _, hotels = service(flights=LongFlights())

    days = planner.plan_skeleton(trip()).itinerary

    # Lands 01:00 on day 3: day 1 is in the air, day 2's night has the bed.
    assert [d["hotel"]["hotel_name"] if d["hotel"] else None for d in days] == [
        None, "Budget Inn", "Budget Inn", None,
    ]
    assert [d["day_total_cost"] for d in days] == [300.0, 80.0, 80.0, 250.0]
    # The hotel is searched for exactly those nights.
    assert (hotels.calls[0]["check_in"], hotels.calls[0]["check_out"]) == ("2026-11-02", "2026-11-04")
    for day in days:
        DailyItinerary.model_validate(day)


def test_hotel_alternatives_use_the_landing_night():
    planner, _, hotels = service()
    itinerary = planner.plan_skeleton(trip()).itinerary
    itinerary[0]["flight"] = [long_flight("L1", 300, "2026-11-03 01:00").model_dump(mode="json")]

    planner.search_alternatives(trip(), category="hotel", day=2, itinerary=itinerary)

    assert hotels.calls[-1]["check_in"] == "2026-11-02"


def test_over_budget_choices_are_flagged_not_dropped():
    planner, _, _ = service()

    days = planner.plan_skeleton(trip(transport=100, accommodation=30)).itinerary

    assert days[0]["flight"][0]["over_budget"] and days[-1]["flight"][0]["over_budget"]
    assert days[0]["hotel"]["over_budget"]


def test_cheapest_hotel_within_nightly_cap():
    within = pick_hotel(HOTELS, nightly_cap(390, 3))
    nothing_fits = pick_hotel(HOTELS, nightly_cap(150, 3))

    assert within.hotel_name == "Budget Inn" and not within.over_budget
    assert nothing_fits.hotel_name == "Budget Inn" and nothing_fits.over_budget
    assert pick_hotel(HOTELS, None).hotel_name == "Budget Inn"
    assert nightly_cap(0, 3) is None


def test_same_day_trip_has_both_legs_and_no_hotel_search():
    planner, _, hotels = service()

    result = planner.plan_skeleton(trip(end="2026-11-01"))

    assert len(result.itinerary) == 1
    assert [f["flight_number"] for f in result.itinerary[0]["flight"]] == ["T1", "R1"]
    assert result.itinerary[0]["hotel"] is None
    assert hotels.calls == []


def test_one_failed_search_keeps_the_others():
    planner, _, _ = service(hotels=FakeHotels(error=RuntimeError("down")))

    result = planner.plan_skeleton(trip())

    assert result.missing == ("hotel",)
    assert result.itinerary[0]["flight"][0]["flight_number"] == "T1"


def evidence(**overrides):
    values = {
        "outbound_flight": OUT[1].model_dump(mode="json"),
        "return_flight": RET[1].model_dump(mode="json"),
        "hotel": HOTELS[2].model_dump(mode="json"),
    }
    values.update(overrides)
    return values


def test_valid_budget_evidence_is_reused_without_searching():
    planner, flights, hotels = service()

    result = planner.plan_skeleton(trip(), reuse=evidence())

    assert result.source == "assessment"
    assert result.itinerary[0]["flight"][0]["flight_number"] == "T2"
    assert result.itinerary[0]["hotel"]["hotel_name"] == "Grand"
    assert flights.calls == [] and hotels.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        {"return_flight": flight("X", 100, "ICN", "KUL").model_dump()},
        {"hotel": hotel("Elsewhere", 100).model_copy(
            update={"location": HotelLocation(lat=1, lng=1, country_code="KR")}
        ).model_dump()},
        {"hotel": {}},
        {"outbound_flight": {"unexpected": "shape"}},
    ],
)
def test_inconsistent_evidence_falls_back_to_search(bad):
    planner, flights, _ = service()

    result = planner.plan_skeleton(trip(), reuse=evidence(**bad))

    assert result.source == "search"
    assert flights.calls


# ── Alternatives ────────────────────────────────────────────────────


def planned_itinerary():
    planner, _, _ = service()
    return planner.plan_skeleton(trip()).itinerary


def test_flight_alternatives_exclude_current_and_use_remaining_budget():
    planner, flights, _ = service()

    # Transport 900, kept return leg costs 250 → this leg may use 650.
    result = planner.search_alternatives(
        trip(transport=900), category="flight", day=1, itinerary=planned_itinerary()
    )

    assert [o["flight_number"] for o in result["options"]] == ["T2", "T3"]
    assert "warning" not in result
    assert len(flights.calls) == 1


def test_flight_alternatives_over_budget_are_flagged_with_warning():
    planner, _, _ = service()

    result = planner.search_alternatives(
        trip(transport=500), category="flight", day=1, itinerary=planned_itinerary()
    )

    assert result["warning"]
    assert all(option["over_budget"] for option in result["options"])


def test_return_leg_alternatives_and_boundary_rule():
    planner, flights, _ = service()
    itinerary = planned_itinerary()

    result = planner.search_alternatives(trip(), category="flight", day=4, itinerary=itinerary)

    assert [o["flight_number"] for o in result["options"]] == ["R2"]
    assert flights.calls[0][0] == "Japan"
    assert "error" in planner.search_alternatives(trip(), category="flight", day=2, itinerary=itinerary)


def test_hotel_alternatives_respect_nightly_cap_and_exclude_current():
    planner, _, _ = service()

    result = planner.search_alternatives(
        trip(accommodation=450), category="accommodation", day=1, itinerary=planned_itinerary()
    )

    assert [o["hotel_name"] for o in result["options"]] == ["Mid Hotel"]


def test_alternative_errors():
    planner, _, _ = service()

    assert "error" in planner.search_alternatives(trip(), category="car", day=1, itinerary=[])
    assert "error" in planner.search_alternatives(
        trip(end="2026-11-01"), category="hotel", day=1, itinerary=[]
    )


# ── Workflow adapter ────────────────────────────────────────────────


def state(**overrides):
    values = dict(
        origin_country="Malaysia",
        country="Japan",
        city=["Tokyo"],
        start_date="2026-11-01",
        end_date="2026-11-04",
        dest_currency_code="jpy",
        num_people=2,
        budget_allocation={"transportation": 1000.0, "accommodation": 600.0},
        budget_assessment=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_state_without_currency_is_refused_not_guessed():
    with pytest.raises(ValueError, match="dest_currency_code"):
        booking_trip_from_state(state(dest_currency_code=None))

    assert booking_trip_from_state(state()).currency == "JPY"


def test_workflow_reuses_matching_assessment():
    assessment = build_assessment(
        BudgetTrip(
            origin="Malaysia",
            destination="Japan",
            destination_city="Tokyo",
            start_date="2026-11-01",
            end_date="2026-11-04",
            num_people=2,
        ),
        base_currency="MYR",
        destination_currency="JPY",
        exchange_rate=32.0,
        outbound_flight=OUT[1].model_dump(mode="json"),
        return_flight=RET[1].model_dump(mode="json"),
        hotel=HOTELS[2].model_dump(mode="json"),
        now=datetime.now(timezone.utc),
    )
    planner, flights, _ = service()

    update = run_booking_planning(state(budget_assessment=assessment.model_dump()), planner=planner)

    assert update["draft_itinerary"][0]["flight"][0]["flight_number"] == "T2"
    assert flights.calls == []

    # A different trip (other dates) must not reuse it.
    run_booking_planning(
        state(budget_assessment=assessment.model_dump(), end_date="2026-11-05"),
        planner=planner,
    )
    assert flights.calls
