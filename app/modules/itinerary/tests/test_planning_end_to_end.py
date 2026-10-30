"""Whole planning flow: real services and validator, fake network edges only.

Booking, activity planning, place verification, routing and the itinerary
validator are the production classes; only flight/hotel search, the idea
generator, place search, geocoding and directions are faked.
"""

import asyncio
import logging

from app.modules.activities.models import PlaceIdea
from app.modules.activities.service import ActivityPlanningService
from app.modules.booking.models import FlightAirport, FlightResult, HotelLocation, HotelResult
from app.modules.booking.service import BookingPlanningService
from app.shared.observability import LogContext
from app.modules.itinerary.service import PLANNING_MAX_ATTEMPTS, ItineraryPlanningService
from app.modules.places.exceptions import PlaceSearchQuotaExhausted
from app.modules.places.models import PlaceCandidate
from app.modules.places.service import VerifiedPlaceSearch
from app.modules.routing.ports import DirectionsRoute
from app.modules.routing.service import DailyRouteService
from app.orchestration.state import AgentState
from app.orchestration.workflows.planning import (
    log_context_from_config,
    run_planning_transaction,
)
from app.shared.geo import GeocodedPlace


TOKYO = (35.6812, 139.7671)


# ── Fake network edges ──────────────────────────────────────────────


def _flight(number, origin, destination, time, price, arrives=None):
    return FlightResult(
        airline="AirTest",
        flight_number=number,
        price=price,
        departure_airport=FlightAirport(id=origin, name=f"{origin} Airport", time=time),
        arrival_airport=FlightAirport(
            id=destination,
            name=f"{destination} Airport",
            **({"time": arrives} if arrives else {}),
        ),
    )


class FakeFlights:
    def __init__(self, *, empty=False):
        self.empty = empty
        self.calls = 0

    def search_flights(self, *, origin, destination, departure_date, currency, max_budget=None, adults=1):
        self.calls += 1
        if self.empty:
            return []
        if origin == "Malaysia":
            return [_flight("T1", "KUL", "HND", f"{departure_date} 08:00", 300.0)]
        return [_flight("R1", "HND", "KUL", f"{departure_date} 20:00", 250.0)]


class FakeHotels:
    def __init__(self, *, empty=False):
        self.empty = empty
        self.calls = 0

    def search_hotels(self, **kwargs):
        self.calls += 1
        if self.empty:
            return []
        return [
            HotelResult(
                hotel_name="Park Hotel",
                price_per_night=100.0,
                location=HotelLocation(lat=35.69, lng=139.70, country_code="JP"),
            )
        ]


PLACES = {
    "senso-ji": ("Senso-ji", 0.01),
    "tokyo tower": ("Tokyo Tower", 0.02),
    "meiji shrine": ("Meiji Shrine", 0.03),
    "ueno park": ("Ueno Park", 0.04),
    "ichiran": ("Ichiran Shibuya", 0.05),
}


class FakePlaceSearch:
    def __init__(self, *, missing=()):
        self.missing = set(missing)
        self.calls = 0

    def search(self, query, *, near=None, zoom=11, quota=None):
        self.calls += 1
        if quota is not None and not quota.try_acquire():
            raise PlaceSearchQuotaExhausted("spent")
        head = query.split(",")[0].strip().casefold()
        if head in self.missing or head not in PLACES:
            return []
        title, offset = PLACES[head]
        return [
            PlaceCandidate(
                title=title,
                latitude=TOKYO[0] + offset,
                longitude=TOKYO[1] + offset,
                address=f"{title}, Tokyo",
                category="Tourist attraction",
                rating=4.5,
            )
        ]


class FakeGeocoder:
    def geocode(self, query, *, proximity=None):
        return GeocodedPlace(latitude=TOKYO[0], longitude=TOKYO[1], place_name=query)


class FakeLocator:
    def resolve_country_code(self, latitude, longitude):
        return "JP"

    def resolve_locality_names(self, latitude, longitude, destination_country_code):
        return ("Tokyo",)


def _idea(name, day, order, place_type="attraction"):
    return PlaceIdea(name=name, place_type=place_type, city="Tokyo", day=day, order=order, estimated_cost=20.0)


class FakeIdeas:
    def __init__(self, initial, rescues=()):
        self.initial = list(initial)
        self.rescues = [list(batch) for batch in rescues]
        self.propose_calls = 0
        self.rescue_calls = []

    def propose(self, request):
        self.propose_calls += 1
        self.request = request
        return list(self.initial)

    def propose_rescue(self, request, *, missing_days, excluded_places):
        self.rescue_calls.append(tuple(missing_days))
        return self.rescues.pop(0) if self.rescues else []


class FakeDirections:
    def __init__(self):
        self.calls = 0

    def fetch_route(self, coordinates, profile):
        self.calls += 1
        return DirectionsRoute(distance_m=4000.0, duration_s=900.0, coordinates=tuple(coordinates))


class FakeAirportGeocoder:
    def geocode(self, query, *, proximity=None):
        return GeocodedPlace(latitude=35.55, longitude=139.78, place_name=query)


# ── Wiring ──────────────────────────────────────────────────────────


def state(**overrides):
    values = dict(
        origin_country="Malaysia",
        country="Japan",
        city=["Tokyo"],
        num_people=1,
        start_date="2026-11-01",
        end_date="2026-11-03",
        base_currency_code="MYR",
        dest_currency_code="JPY",
        total_convert_budget=6000.0,
        budget_allocation={
            "transportation": 1500.0,
            "accommodation": 1500.0,
            "food": 1000.0,
            "activity": 1000.0,
            "shopping": 500.0,
            "emergency_fund": 500.0,
        },
    )
    values.update(overrides)
    return AgentState(**values)


class World:
    def __init__(self, *, ideas, flights=None, hotels=None, search=None):
        self.flights = flights or FakeFlights()
        self.hotels = hotels or FakeHotels()
        self.search = search or FakePlaceSearch()
        self.ideas = ideas
        self.directions = FakeDirections()

    def run(self, agent_state, **kwargs):
        places = VerifiedPlaceSearch(search=self.search, geocoder=FakeGeocoder(), locator=FakeLocator())
        return asyncio.run(
            run_planning_transaction(
                agent_state,
                planner=ItineraryPlanningService(),
                booking=BookingPlanningService(flights=self.flights, hotels=self.hotels),
                activities=ActivityPlanningService(ideas=self.ideas, places=places),
                routes=DailyRouteService(directions=self.directions, geocoder=FakeAirportGeocoder()),
                **kwargs,
            )
        )


GOOD_IDEAS = [
    _idea("Senso-ji", 1, 1),
    _idea("Ichiran", 1, 2, "restaurant"),
    _idea("Tokyo Tower", 2, 1),
    _idea("Meiji Shrine", 3, 1),
]


# ── Tests ───────────────────────────────────────────────────────────


def test_full_flow_produces_a_validated_candidate(caplog):
    world = World(ideas=FakeIdeas(GOOD_IDEAS))

    with caplog.at_level(logging.INFO, logger="app.modules.itinerary.service"):
        update = world.run(state(), log_context=LogContext(request_id="req-1", session_id="s-1"))

    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]
    assert update["planning_attempts"] == 1
    assert update["planning_issue_codes"] == []

    candidate = update["candidate_plan"]
    assert [day["day"] for day in candidate["itinerary"]] == [1, 2, 3]
    assert sorted(candidate["maps"]) == [1, 2, 3]
    assert candidate["validation"] == {"issues": []}
    assert candidate["attempt"] == 1

    # Log fields are visible in the message itself.
    complete = [r.getMessage() for r in caplog.records if r.getMessage().startswith("planning.complete")]
    assert complete and "request_id=req-1" in complete[0] and "country=JP" in complete[0]
    assert "outcome=validated" in complete[0]


def test_an_empty_day_is_replanned_alone_and_booking_is_not_repeated():
    # Day 3's idea can't be verified on the first pass; the planner's own
    # rescue rounds also fail, so the transaction must retry only day 3.
    ideas = FakeIdeas(
        [_idea("Senso-ji", 1, 1), _idea("Tokyo Tower", 2, 1), _idea("Nowhere", 3, 1)],
        rescues=[[], [], [_idea("Meiji Shrine", 3, 1)]],
    )
    world = World(ideas=ideas)

    update = world.run(state())

    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]
    assert update["planning_attempts"] == 2
    assert update["planning_issue_codes"] == ["day.activities.empty"]
    # One booking search round, one full brainstorm; the retry asked for day 3 only.
    assert (world.flights.calls, world.hotels.calls) == (2, 1)
    assert ideas.propose_calls == 1
    assert ideas.rescue_calls[-1] == (3,)
    days = update["candidate_plan"]["itinerary"]
    assert [a["name"] for a in days[0]["activities"]] == ["Senso-ji"]
    assert [a["name"] for a in days[2]["activities"]] == ["Meiji Shrine"]


def test_no_hotel_stops_after_one_fresh_search_without_activity_work():
    world = World(ideas=FakeIdeas(GOOD_IDEAS), hotels=FakeHotels(empty=True))

    update = world.run(state())

    assert update["planning_outcome"] == "unavailable"
    assert update["candidate_plan"] is None
    # Final: later graph turns must not restart it.
    assert update["planning_attempts"] == PLANNING_MAX_ATTEMPTS
    assert update["planning_issue_codes"] == ["hotel.night.missing", "hotel.night.missing"]
    assert world.hotels.calls == 2
    assert world.ideas.propose_calls == 0
    assert world.search.calls == 0
    assert world.directions.calls == 0


def test_unplannable_trip_makes_no_provider_calls():
    world = World(ideas=FakeIdeas(GOOD_IDEAS))

    update = world.run(state(dest_currency_code=None))

    assert update["planning_outcome"] == "unavailable"
    assert update["planning_issue_codes"] == ["planning.input.invalid"]
    assert update["planning_attempts"] == PLANNING_MAX_ATTEMPTS
    assert world.flights.calls == world.hotels.calls == 0


def test_reentry_only_gets_the_attempts_that_are_left():
    world = World(ideas=FakeIdeas(GOOD_IDEAS))

    exhausted = world.run(state(planning_attempts=PLANNING_MAX_ATTEMPTS))
    assert exhausted == {"planning_outcome": "unavailable", "candidate_plan": None}
    assert world.flights.calls == 0

    update = world.run(state(planning_attempts=2, planning_issue_codes=["earlier"]))
    assert update["planning_outcome"] == "validated"
    assert update["planning_attempts"] == 3
    assert update["candidate_plan"]["attempt"] == 3
    assert update["planning_issue_codes"] == ["earlier"]


def test_log_context_from_config_reads_ids():
    context = log_context_from_config(
        {"configurable": {"thread_id": "t-9"}, "metadata": {"request_id": "r-9"}}
    )
    assert (context.request_id, context.session_id) == ("r-9", "t-9")
    assert log_context_from_config(None) == LogContext()


class OvernightFlights(FakeFlights):
    """The schedule found by the live smoke test: lands 01:00 on day 2,
    leaves 02:15 on day 3."""

    def search_flights(self, *, origin, destination, departure_date, **kwargs):
        self.calls += 1
        if origin == "Malaysia":
            return [_flight("VJ826", "KUL", "HND", f"{departure_date} 13:35", 300.0, arrives="2026-11-02 01:00")]
        return [_flight("TR805", "HND", "KUL", f"{departure_date} 02:15", 250.0, arrives=f"{departure_date} 18:15")]


def test_travel_days_get_no_places_and_the_plan_still_validates():
    ideas = FakeIdeas(GOOD_IDEAS + [_idea("Ueno Park", 2, 2)])
    world = World(ideas=ideas, flights=OvernightFlights())

    update = world.run(state())

    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]
    days = update["candidate_plan"]["itinerary"]
    assert [day["activities"] for day in (days[0], days[2])] == [[], []]
    assert [a["name"] for a in days[1]["activities"]] == ["Tokyo Tower", "Ueno Park"]
    # Travel days were neither offered to the model nor "rescued".
    assert ideas.request.day_limits == (0, 8, 0)
    assert ideas.rescue_calls == []
