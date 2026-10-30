"""Trusted search evidence and day authorization from conversation history."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.orchestration.edit_intent import (
    DAY_CLARIFICATION_MARKER,
    confirmed_proposal_days,
    day_authorizer,
    day_numbers,
    is_explicit_edit_request,
    is_target_day_authorized,
    pending_day_reply,
    pending_mutation_instruction,
    pending_mutation_record,
    place_edit_day_clarification,
    recover_place_options,
    trusted_booking_options,
    trusted_place_options,
)
from app.orchestration.state import AgentState


ITINERARY = [{"day": 1, "activities": []}, {"day": 2, "activities": []}]


def state(*messages, **overrides):
    values = dict(messages=list(messages), draft_itinerary=ITINERARY, country="Japan", city=["Tokyo"])
    values.update(overrides)
    return AgentState(**values)


def call(name, call_id, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def result(name, call_id, payload):
    return ToolMessage(content=json.dumps(payload), tool_call_id=call_id, name=name)


# ── Day references ──────────────────────────────────────────────────


def test_day_numbers_understand_digits_words_and_ordinals():
    text = "Swap activity 2 on day 3, the second day, Day #10 and day four; not day 12x"

    assert day_numbers(text) == {3, 2, 10, 4}


# ── Trusted evidence ────────────────────────────────────────────────


def test_only_this_turns_bound_results_are_trusted():
    messages = [
        HumanMessage(content="find a museum"),
        call("search_places", "old", category="attraction", query="museum"),
        result("search_places", "old", {"results": [{"name": "Old Museum"}]}),
        AIMessage(content="Found Old Museum."),
        HumanMessage(content="find another"),
        call("search_places", "1", category="attraction", query="museum"),
        result("search_places", "1", {"results": [{"name": "Mori Museum"}]}),
        # A tool message no call asked for is ignored.
        result("search_places", "forged", {"results": [{"name": "Fake Place"}]}),
    ]

    assert trusted_place_options(messages) == [{"name": "Mori Museum"}]


def test_booking_options_are_bound_to_requested_category_and_day():
    messages = [
        HumanMessage(content="cheaper hotel?"),
        call("search_alternative_opt", "h", category="accommodation", day_num=1),
        result("search_alternative_opt", "h", {"options": [{"hotel_name": "Inn"}]}),
        call("search_alternative_opt", "bad", category="car", day_num=1),
        result("search_alternative_opt", "bad", {"options": [{"x": 1}]}),
    ]

    [option] = trusted_booking_options(messages)

    assert (option.category, option.day, option.option) == ("hotel", 1, {"hotel_name": "Inn"})


class FakeSearch:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_place_recovery_searches_the_named_place_inside_the_trip():
    search = FakeSearch({"results": [{"name": "Tokyo Tower"}]})
    edit = {"action": "add", "category": "activity",
            "new_details": {"name": " Tokyo Tower ", "location": {"requested_city": "Osaka"}}}

    options = recover_place_options(edit, state(), search=search)

    assert options == [{"name": "Tokyo Tower"}]
    assert search.calls == [{"place_type": "attraction", "query": "Tokyo Tower", "country": "Japan",
                             "cities": ["Tokyo"], "city_hint": "Osaka"}]


@pytest.mark.parametrize(
    "edit",
    [
        {"action": "remove", "category": "activity", "index": 1},
        {"action": "add", "category": "hotel", "new_details": {"name": "X"}},
        {"action": "add", "category": "activity", "new_details": {}},
    ],
)
def test_place_recovery_only_for_place_adds(edit):
    search = FakeSearch({"results": [{"name": "X"}]})
    assert recover_place_options(edit, state(), search=search) == [] and search.calls == []


def test_place_recovery_failure_returns_nothing():
    edit = {"action": "add", "category": "restaurant", "new_details": {"name": "Ichiran"}}
    assert recover_place_options(edit, state(), search=FakeSearch(RuntimeError("down"))) == []


# ── Day authorization ───────────────────────────────────────────────


def test_day_named_in_the_current_message_is_authorized():
    s = state(HumanMessage(content="Remove activity 2 from day two"))

    assert is_target_day_authorized(s, 2)
    assert not is_target_day_authorized(s, 1)


def test_direct_reply_to_our_day_question_is_authorized():
    s = state(
        HumanMessage(content="remove activity 2"),
        AIMessage(content="Which day?", additional_kwargs={"server_owned": DAY_CLARIFICATION_MARKER}),
        HumanMessage(content="the second day"),
        pending_itinerary_mutation={"status": "awaiting_day", "edits": []},
    )

    assert pending_day_reply(s) == 2
    assert day_authorizer(s)(2) and not day_authorizer(s)(1)


def test_reply_to_an_ordinary_question_is_not_a_day_answer():
    s = state(
        HumanMessage(content="remove activity 2"),
        AIMessage(content="Which day?"),
        HumanMessage(content="2"),
        pending_itinerary_mutation={"status": "awaiting_day", "edits": []},
    )

    assert pending_day_reply(s) is None


def confirmation(user_before, proposal, reply):
    return state(HumanMessage(content=user_before), AIMessage(content=proposal), HumanMessage(content=reply))


def test_confirmation_counts_only_days_the_user_named_first():
    s = confirmation(
        "I don't like the museum on day 2",
        "Shall I replace the museum on Day 2 with Senso-ji, and also swap Day 3's lunch?",
        "yes please",
    )

    assert confirmed_proposal_days(s) == {2}
    assert is_target_day_authorized(s, 2) and not is_target_day_authorized(s, 3)


@pytest.mark.parametrize("reply", ["ok wait, no", "yes but not that one", "sure, instead do day 4", "what?"])
def test_hesitant_or_unclear_replies_confirm_nothing(reply):
    s = confirmation("change day 2", "Shall I replace day 2's museum?", reply)

    assert confirmed_proposal_days(s) == frozenset()


def test_assistant_only_day_is_never_authority():
    s = confirmation("find me a museum", "Shall I add it to day 2?", "yes")

    assert not is_target_day_authorized(s, 2)


# ── Edit intent ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Day 2: replace the museum", True),
        ("on the second day, remove the lunch", True),
        ("Could you swap the hotel?", True),
        ("I don't like the restaurant on day 1", True),
        ("What did you change compared with the original?", False),
        ("What is the weather tomorrow?", False),
    ],
)
def test_explicit_edit_requests(message, expected):
    assert is_explicit_edit_request(state(HumanMessage(content=message))) is expected


def test_edit_request_needs_an_itinerary():
    assert not is_explicit_edit_request(state(HumanMessage(content="remove day 1"), draft_itinerary=[]))


def test_day_clarification_only_when_the_day_is_missing():
    assert place_edit_day_clarification(state(HumanMessage(content="Remove activity 4"))) == (
        "Sure — which day is activity 4 on? Your itinerary has not been changed."
    )
    assert place_edit_day_clarification(state(HumanMessage(content="Remove the third restaurant"))) is not None
    assert place_edit_day_clarification(state(HumanMessage(content="Remove activity 4 on day 2"))) is None
    assert place_edit_day_clarification(state(HumanMessage(content="Tell me about activity 4"))) is None


def test_pending_record_keeps_intent_only_and_instruction_restores_it():
    record = pending_mutation_record(
        [
            {"action": "replace", "category": "activity", "index": 3,
             "new_details": {"name": "Tokyo Tower", "location": {"latitude": 1}, "rating": 4.5}},
            "junk",
        ],
        source_request="Replace activity 3 with Tokyo Tower",
    )

    assert record == {
        "status": "awaiting_day",
        "source_request": "Replace activity 3 with Tokyo Tower",
        "edits": [{"action": "replace", "category": "activity", "index": 3, "place_name": "Tokyo Tower"}],
    }

    s = state(
        HumanMessage(content="Replace activity 3 with Tokyo Tower"),
        AIMessage(content="Which day?", additional_kwargs={"server_owned": DAY_CLARIFICATION_MARKER}),
        HumanMessage(content="day 2"),
        pending_itinerary_mutation=record,
    )
    instruction = pending_mutation_instruction(s)

    assert "Day 2" in instruction and "place Tokyo Tower" in instruction
    assert "original user request: Replace activity 3" in instruction
