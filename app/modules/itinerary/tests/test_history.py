"""Itinerary revision history and the model-facing projections."""

from app.modules.itinerary.history import (
    MAX_HISTORY_REVISIONS,
    append_previous_itinerary,
    recover_itinerary_history,
)
from app.modules.itinerary.projection import compact_itinerary, history_text, itinerary_text


def day(n, *, place="Senso-ji", flights=None):
    plan = {
        "day": n,
        "date": f"2026-11-0{n}",
        "day_total_cost": 120.0,
        "hotel": {"hotel_name": "Park Hotel", "price_per_night": 100.0, "image": "big.jpg",
                  "location": {"lat": 35.6, "lng": 139.7}},
        "activities": [
            {"name": place, "type": "attraction", "order": 1, "estimated_cost": 20.0,
             "thumbnail": "t.png", "description": "long text", "location": {"latitude": 1, "longitude": 2}},
        ],
    }
    if flights:
        plan["flight"] = flights
    return plan


def flight(number, origin, destination):
    return {"airline": "AirTest", "flight_number": number, "price": 300.0, "airline_logo": "x.png",
            "departure_airport": {"id": origin, "name": origin, "time": "2026-11-01 08:00"},
            "arrival_airport": {"id": destination, "name": destination, "time": "2026-11-01 16:00"}}


def snapshot(revision, place="Senso-ji"):
    return {"plan_revision": revision, "draft_itinerary": [day(1, place=place)]}


def test_entries_are_stored_compact():
    [entry] = append_previous_itinerary([], snapshot(1))

    stored = entry["draft_itinerary"][0]
    assert entry["revision"] == 1
    assert stored["activities"] == [
        {"name": "Senso-ji", "type": "attraction", "order": 1, "estimated_cost": 20.0}
    ]
    assert "location" not in stored["hotel"] and "image" not in stored["hotel"]


def test_compacting_twice_keeps_flights():
    plan = [day(1, flights=[flight("T1", "KUL", "HND"), flight("R1", "HND", "KUL")])]

    once = compact_itinerary(plan)

    assert compact_itinerary(once) == once
    assert [f["flight_number"] for f in once[0]["flights"]] == ["T1", "R1"]
    # Re-appending stored history does not lose anything either.
    history = append_previous_itinerary([], {"plan_revision": 1, "draft_itinerary": plan})
    assert append_previous_itinerary(history, None) == history


def test_duplicates_are_ignored_and_first_revision_is_kept_forever():
    history: list = []

    for revision in range(1, 30):
        history = append_previous_itinerary(history, snapshot(revision, place=f"Place {revision}"))

    history = append_previous_itinerary(history, snapshot(29, place="Replay"))
    revisions = [entry["revision"] for entry in history]

    assert len(history) == MAX_HISTORY_REVISIONS
    assert revisions[0] == 1 and revisions[1:] == list(range(11, 30))
    assert history[-1]["draft_itinerary"][0]["activities"][0]["name"] == "Place 29"


def test_invalid_entries_are_skipped():
    junk = [None, {"revision": True, "draft_itinerary": [day(1)]}, {"revision": 2, "draft_itinerary": []}]

    assert append_previous_itinerary(junk, None) == []


def test_recovery_from_legacy_checkpoints_excludes_the_current_revision():
    states = [
        {"plan_revision": 1, "draft_itinerary": [day(1, place="First")]},
        {"accepted_plan_snapshot": snapshot(2, place="Second")},
        {"plan_revision": 3, "draft_itinerary": [day(1, place="Current")]},
        "junk",
    ]

    history = recover_itinerary_history(states, current_revision=3)

    assert [entry["revision"] for entry in history] == [1, 2]
    assert recover_itinerary_history(states, current_revision=1) == []


def test_itinerary_text_shows_every_leg_and_item_indexes():
    plan = [day(1, flights=[flight("T1", "KUL", "HND"), flight("R1", "HND", "KUL")])]
    plan[0]["hotel"]["over_budget"] = True

    text = itinerary_text(plan)

    assert "AirTest T1 (KUL → HND" in text and "AirTest R1 (HND → KUL" in text
    assert "[OVER BUDGET]" in text
    assert "[1] attraction: Senso-ji ~20.00" in text
    assert itinerary_text([]).startswith("No itinerary drafted yet")


def test_history_text_lists_revisions():
    text = history_text(append_previous_itinerary([], snapshot(4, place="Old Museum")))

    assert text.startswith("Revision 4") and "[1] attraction: Old Museum" in text
