"""Budget service and chat gate with fake providers and an in-memory cache."""

import json
from datetime import datetime, timezone

import pytest

from app.modules.booking.models import BookingTrip, FlightAirport, FlightResult, HotelResult
from app.modules.booking.service import BookingPlanningService
from app.modules.budget.assessment import assessment_cache_key
from app.modules.budget.chat_gate import evaluate_chat_budget
from app.modules.budget.exceptions import BudgetAssessmentUnavailable
from app.modules.budget.models import BudgetTrip
from app.modules.budget.service import BudgetAssessmentService


NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
TRIP_KW = dict(
    origin="Malaysia",
    destination="Japan",
    destination_city="Tokyo",
    start_date="2026-11-01",
    end_date="2026-11-05",
    num_people=2,
)
TRIP = BudgetTrip(**TRIP_KW)


class FakeCurrency:
    def get_currency_code(self, country):
        return {"Malaysia": "MYR", "Japan": "JPY"}[country]

    def get_currency_rate(self, origin, destination):
        return 32.0


class FakeFlights:
    def __init__(self, prices=(40000.0, 35000.0)):
        self.prices = list(prices)
        self.calls = 0

    def search_flights(self, **kwargs):
        self.calls += 1
        price = self.prices[0] if kwargs["origin"] == "Malaysia" else self.prices[1]
        return [FlightResult(price=price), FlightResult(price=price * 2)]


class FakeHotels:
    def __init__(self, results=None):
        self.results = [HotelResult(price_per_night=100.0)] if results is None else results
        self.calls = 0
        self.check_ins = []

    def search_hotels(self, **kwargs):
        self.calls += 1
        self.check_ins.append(kwargs["check_in"])
        return self.results


class MemoryCache:
    """In-memory stand-in for the Redis lease cache."""

    def __init__(self, *, publish_ok=True):
        self.entries = {}
        self.claims = {}
        self.publish_ok = publish_ok

    @staticmethod
    def _k(key):
        return json.dumps(key, sort_keys=True)

    def get(self, key):
        return self.entries.get(self._k(key))

    def claim(self, key, *, lease_seconds):
        k = self._k(key)
        if k in self.entries:
            return "cached", self.entries[k], None
        if k in self.claims:
            return "busy", None, None
        self.claims[k] = "claim-1"
        return "claimed", None, "claim-1"

    def publish(self, key, payload, *, claim_id, ttl_hours):
        k = self._k(key)
        if not self.publish_ok or self.claims.get(k) != claim_id:
            return False
        self.entries[k] = payload
        del self.claims[k]
        return True

    def release(self, key, *, claim_id):
        return self.claims.pop(self._k(key), None) is not None

    def invalidate(self, key):
        return self.entries.pop(self._k(key), None) is not None

    def wait_for_change(self, *, timeout_seconds):
        pass


def make_service(*, flights=None, hotels=None, cache=None):
    return BudgetAssessmentService(
        currency_provider=FakeCurrency(),
        flight_search=flights or FakeFlights(),
        hotel_search=hotels or FakeHotels(),
        cache=cache or MemoryCache(),
        clock=lambda: NOW,
        claim_wait_seconds=0.0,
    )


# ── Service ─────────────────────────────────────────────────────────


def test_creates_once_then_reuses_cache():
    flights = FakeFlights()
    service = make_service(flights=flights)

    first = service.get_or_create(TRIP)
    second = service.get_or_create(TRIP)

    assert first.persisted and second.persisted
    assert first.assessment == second.assessment
    assert first.assessment.recommended_minimum_budget == 9375.0
    assert flights.calls == 2  # outbound + return, only on the first call


def test_hotel_is_searched_from_the_night_the_outbound_flight_lands():
    class LongFlights(FakeFlights):
        def search_flights(self, **kwargs):
            if kwargs["origin"] != "Malaysia":
                return super().search_flights(**kwargs)
            self.calls += 1
            return [FlightResult(price=40000.0, arrival_time="2026-11-03 01:00")]

    hotels = FakeHotels()

    result = make_service(flights=LongFlights(), hotels=hotels).get_or_create(TRIP)

    assert hotels.check_ins == ["2026-11-02"]
    assert result.assessment.evidence.hotel_nights == 3


class TimedFlights:
    """KUL ⇄ HND with real times: the cheapest legs leave little time there."""

    def __init__(self):
        self.calls = 0

    def search_flights(self, **kwargs):
        self.calls += 1

        def leg(number, price, departs, arrives, origin, destination):
            return FlightResult(
                flight_number=number,
                price=price,
                departure_airport=FlightAirport(id=origin, time=departs),
                arrival_airport=FlightAirport(id=destination, time=arrives),
            )

        if kwargs["origin"] == "Malaysia":
            return [
                leg("LATE", 40000.0, "2026-11-01 15:10", "2026-11-01 23:00", "KUL", "HND"),
                leg("EARLY", 44000.0, "2026-11-01 00:30", "2026-11-01 08:30", "KUL", "HND"),
            ]
        return [
            leg("DAWN", 35000.0, "2026-11-05 06:00", "2026-11-05 13:00", "HND", "KUL"),
            leg("NIGHT", 38000.0, "2026-11-05 21:00", "2026-11-06 03:00", "HND", "KUL"),
        ]


def test_budget_evidence_is_the_flights_the_plan_books():
    result = make_service(flights=TimedFlights()).get_or_create(TRIP)
    evidence = result.assessment.evidence

    # 82,000 is within 15 % of the cheapest 75,000 and gives two more half days.
    assert (evidence.outbound_flight["flight_number"], evidence.return_flight["flight_number"]) == ("EARLY", "NIGHT")
    assert (evidence.outbound_flight_price, evidence.return_flight_price) == (44000.0, 38000.0)
    # The minimum is priced from those flights: 82,000 / 0.25 / 32 MYR.
    assert result.assessment.recommended_minimum_budget == 10250.0

    # The plan's own search, from the same results, books the same pair.
    planner = BookingPlanningService(flights=TimedFlights(), hotels=FakeHotels())
    days = planner.plan_skeleton(
        BookingTrip(
            origin_country="Malaysia",
            destination_country="Japan",
            destination_country_code="JP",
            city="Tokyo",
            start_date=TRIP.start_date,
            end_date=TRIP.end_date,
            currency="JPY",
            adults=2,
        )
    ).itinerary
    assert (days[0]["flight"][0]["flight_number"], days[-1]["flight"][-1]["flight_number"]) == ("EARLY", "NIGHT")


def test_flights_without_a_real_price_are_not_evidence():
    class FreeFlights(FakeFlights):
        def search_flights(self, **kwargs):
            self.calls += 1
            return [FlightResult(price=0.0)]

    with pytest.raises(BudgetAssessmentUnavailable):
        make_service(flights=FreeFlights()).get_or_create(TRIP)


def test_missing_hotel_makes_assessment_unavailable_and_releases_lease():
    cache = MemoryCache()
    service = make_service(hotels=FakeHotels(results=[]), cache=cache)

    with pytest.raises(BudgetAssessmentUnavailable):
        service.get_or_create(TRIP)
    assert cache.claims == {}  # lease released so a retry can build it


def test_unsaved_assessment_is_flagged_not_persisted():
    result = make_service(cache=MemoryCache(publish_ok=False)).get_or_create(TRIP)
    assert result.persisted is False


def test_busy_lease_times_out_as_unavailable():
    cache = MemoryCache()
    key = assessment_cache_key(TRIP, base_currency="MYR", destination_currency="JPY")
    # Another request holds the lease for the same trip.
    cache.claims[cache._k(key)] = "someone-else"

    with pytest.raises(BudgetAssessmentUnavailable):
        make_service(cache=cache).get_or_create(TRIP)


class DownCache(MemoryCache):
    """A cache that is unreachable: every lease attempt fails."""

    def __init__(self, *, raises=False):
        super().__init__()
        self.raises = raises
        self.released = []

    def get(self, key):
        return None

    def claim(self, key, *, lease_seconds):
        if self.raises:
            raise RuntimeError("cache client bug")
        return "error", None, None

    def publish(self, key, payload, *, claim_id, ttl_hours):
        raise AssertionError("nothing can be published without a lease")

    def release(self, key, *, claim_id):
        self.released.append(claim_id)
        return False


@pytest.mark.parametrize("raises", [False, True])
def test_cache_down_still_builds_an_unsaved_assessment(raises):
    cache = DownCache(raises=raises)
    flights = FakeFlights()

    result = make_service(flights=flights, cache=cache).get_or_create(TRIP)

    assert result.persisted is False
    assert result.assessment.recommended_minimum_budget == 9375.0
    assert flights.calls == 2
    assert cache.released == []


def test_cache_down_and_build_fails_releases_nothing():
    cache = DownCache()

    with pytest.raises(BudgetAssessmentUnavailable):
        make_service(hotels=FakeHotels(results=[]), cache=cache).get_or_create(TRIP)

    assert cache.released == []


def test_cache_down_accepts_a_sufficient_budget_but_never_offers_a_confirmation():
    service = make_service(cache=DownCache())

    enough = gate(service, {"mode": "amount", "total_base_budget": 10000})
    too_low = gate(service, {"mode": "amount", "total_base_budget": 100})

    assert enough.status == "accepted"
    assert too_low.status == "budget_check_unavailable"
    assert too_low.reason == "assessment_cache_unavailable"


class StuckEntryCache(MemoryCache):
    """Holds an entry that fails validation and can't be deleted."""

    def claim(self, key, *, lease_seconds):
        self.claim_calls = getattr(self, "claim_calls", 0) + 1
        return "cached", {"not": "an assessment"}, None

    def invalidate(self, key):
        return False


def test_undeletable_invalid_entry_builds_unsaved_instead_of_looping():
    cache = StuckEntryCache()

    result = make_service(cache=cache).get_or_create(TRIP)

    assert result.persisted is False
    assert result.assessment.recommended_minimum_budget == 9375.0
    assert cache.claim_calls == 1


def test_load_confirmed_requires_matching_id():
    service = make_service()
    created = service.get_or_create(TRIP).assessment

    assert service.load_confirmed(TRIP, assessment_id=created.assessment_id) == created
    assert service.load_confirmed(TRIP, assessment_id="other") is None


# ── Chat gate ───────────────────────────────────────────────────────


def gate(service, proposal, pending=None, current=None, **overrides):
    return evaluate_chat_budget(
        proposal=proposal,
        pending_confirmation=pending,
        current_assessment=current,
        assessment_service=service,
        **{**TRIP_KW, **overrides},
    )


def test_sufficient_amount_is_accepted():
    decision = gate(make_service(), {"mode": "amount", "total_base_budget": 10000})
    assert decision.status == "accepted"
    assert decision.accepted_total_base_budget == 10000


def test_low_amount_requires_confirmation_then_confirm_accepts_minimum():
    service = make_service()
    low = gate(service, {"mode": "amount", "total_base_budget": 100})
    assert low.status == "budget_confirmation_required"
    assert low.reason == "insufficient_budget"

    confirmed = gate(service, {"mode": "confirm"}, pending=low.pending_confirmation)
    assert confirmed.status == "accepted"
    assert confirmed.accepted_total_base_budget == 9375.0


def test_confirm_without_pending_or_with_tampered_record_is_rejected():
    service = make_service()
    assert gate(service, {"mode": "confirm"}).reason == "no_pending_budget_confirmation"

    pending = gate(service, {"mode": "recommendation"}).pending_confirmation
    pending["assessment"]["recommended_minimum_budget"] = 1.0
    assert gate(service, {"mode": "confirm"}, pending=pending).reason == (
        "assessment_expired_or_invalid"
    )


def test_recommendation_needs_a_saved_assessment():
    service = make_service(cache=MemoryCache(publish_ok=False))
    decision = gate(service, {"mode": "recommendation"})
    assert decision.reason == "assessment_cache_unavailable"


@pytest.mark.parametrize(
    ("proposal", "overrides", "reason"),
    [
        ("not a dict", {}, "invalid_budget_proposal"),
        ({"mode": "amount", "total_base_budget": -5}, {}, "invalid_budget_proposal"),
        ({"mode": "unknown"}, {}, "invalid_budget_proposal"),
        ({"mode": "amount", "total_base_budget": 10}, {"end_date": "2026-10-01"}, "invalid_trip_details"),
        ({"mode": "amount", "total_base_budget": 10}, {"num_people": 0}, "invalid_trip_details"),
    ],
)
def test_invalid_inputs_are_unavailable(proposal, overrides, reason):
    decision = gate(make_service(), proposal, **overrides)
    assert decision.status == "budget_check_unavailable"
    assert decision.reason == reason
