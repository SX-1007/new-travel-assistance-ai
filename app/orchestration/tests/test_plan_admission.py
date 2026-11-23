"""Chat re-plans pass the same limits as a new trip from the form: the
daily plan limit, one plan per user at a time, and the server's planning
slots. A refused re-plan spends nothing and says why."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from app.api.presenters.plan import planning_unavailable_reason
from app.api.request_limits import PLANS, ChatPlanAdmission, PlanSlots, RequestLimits
from app.api.tests.test_request_limits import MemoryCounter
from app.modules.output_review.service import OutputReviewService
from app.orchestration.nodes.planning import (
    execute_validated_transaction_node,
    generate_reviewed_response_node,
)
from app.orchestration.nodes.replies import server_reply
from app.orchestration.plan_admission import PLAN_ADMITTED_KEY, refusal_for
from app.orchestration.tests.node_fakes import (
    CONFIG,
    Network,
    ScriptedReviewer,
    committed_state,
    profiles,
    reply_services,
    run,
)
from app.shared.exceptions import (
    PlanAlreadyRunning,
    PlanningCapacityFull,
    RateLimitExceeded,
    RateLimitUnavailable,
)

LIMIT = RateLimitExceeded(
    "You've reached the limit of 3 new trip plans a day. Please try again in 5 h 12 min.",
    retry_after_seconds=18720,
)


class RecordingAdmission:
    def __init__(self, error=None):
        self.error = error
        self.events = []

    @asynccontextmanager
    async def admit(self, user_id):
        self.events.append(("enter", user_id))

        if self.error is not None:
            raise self.error

        try:
            yield
        finally:
            self.events.append(("exit", user_id))


def replan_state():
    # A committed trip whose new end date needs a full re-plan.
    return committed_state(plan_overrides={"end_date": "2026-11-03"})


def execute(network, admission, config=CONFIG):
    loader, _ = profiles()
    planning = network.planning()
    planning = type(planning)(**{**{f: getattr(planning, f) for f in planning.__slots__}, "admission": admission})
    return run(execute_validated_transaction_node(replan_state(), config, planning=planning, profiles=loader))


# ── The planning node ───────────────────────────────────────────────


def test_a_refused_replan_spends_nothing_and_keeps_the_reason():
    network = Network()

    update = execute(network, RecordingAdmission(LIMIT))

    assert network.flights.calls == 0 and network.places.calls == 0
    assert update["planning_outcome"] == "unavailable" and update["candidate_plan"] is None
    assert "planning.limit.reached" in update["planning_issue_codes"]
    assert update["planning_refusal"].startswith(LIMIT.message)


def test_the_refusal_is_the_reply_and_is_then_cleared():
    state = replan_state().model_copy(update=execute(Network(), RecordingAdmission(LIMIT)))
    replies, _ = reply_services()
    loader, _ = profiles()

    update = run(generate_reviewed_response_node(state, CONFIG, replies=replies, profiles=loader))

    assert update["messages"][0].content == state.planning_refusal
    assert update["planning_refusal"] is None
    assert update["plan_overrides"] is None  # the change is dropped, the plan kept


def test_an_admitted_replan_holds_its_admission_until_planning_ends():
    admission = RecordingAdmission()

    update = execute(Network(), admission)

    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]
    assert admission.events == [("enter", "user-1"), ("exit", "user-1")]


def test_a_run_the_api_already_admitted_is_not_counted_again():
    admission = RecordingAdmission(LIMIT)
    config = {**CONFIG, "configurable": {**CONFIG["configurable"], PLAN_ADMITTED_KEY: True}}

    update = execute(Network(), admission, config)

    assert admission.events == []
    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]


def test_a_run_without_a_user_id_is_refused():
    network = Network()
    config = {"configurable": {"thread_id": "session-1"}}

    update = execute(network, RecordingAdmission(), config)

    assert update["planning_outcome"] == "unavailable"
    assert "planning.limits.unavailable" in update["planning_issue_codes"]
    assert network.places.calls == 0


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (LIMIT, "request_limit_reached"),
        (PlanAlreadyRunning(), "request_limit_reached"),
        (PlanningCapacityFull(), "planning_busy"),
        (RateLimitUnavailable(), "planning_busy"),
    ],
)
def test_each_refusal_has_a_public_reason(error, reason):
    update = execute(Network(), RecordingAdmission(error))

    assert planning_unavailable_reason(update) == reason


@pytest.mark.parametrize("error", [LIMIT, PlanAlreadyRunning(), PlanningCapacityFull(), RateLimitUnavailable()])
def test_the_refusal_texts_pass_the_reply_rules(error):
    text = refusal_for(error).reply
    reviewer = OutputReviewService(reviewer=ScriptedReviewer())

    message = server_reply(replan_state(), text, marker="test", reviewer=reviewer, fallback="FALLBACK")

    assert message.content == text
    assert "not been changed" in text


# ── ChatPlanAdmission ───────────────────────────────────────────────


def admission(counter=None, *, max_running=2):
    counter = counter or MemoryCounter()
    slots = PlanSlots(max_running=max_running)
    return ChatPlanAdmission(limits=RequestLimits(counter, fail_closed=True), slots=slots), slots, counter


def test_a_chat_replan_counts_as_a_plan():
    chat, _, counter = admission()

    async def scenario():
        async with chat.admit("user-1"):
            pass

    asyncio.run(scenario())

    assert counter.counts[(PLANS.name, "user-1")] == 1


def test_the_fourth_plan_of_the_day_is_refused():
    chat, _, counter = admission()
    counter.counts[(PLANS.name, "user-1")] = PLANS.limit

    async def scenario():
        async with chat.admit("user-1"):
            pass

    with pytest.raises(RateLimitExceeded):
        asyncio.run(scenario())


def test_a_full_server_refuses_without_using_up_a_plan():
    chat, slots, counter = admission(max_running=1)

    async def scenario():
        async with slots.planning():
            async with chat.admit("user-1"):
                pass

    with pytest.raises(PlanningCapacityFull):
        asyncio.run(scenario())

    assert counter.counts == {}


def test_a_user_with_a_plan_running_cannot_start_a_second():
    chat, slots, counter = admission()

    async def scenario():
        async with slots.for_user("user-1"):
            async with chat.admit("user-1"):
                pass

    with pytest.raises(PlanAlreadyRunning):
        asyncio.run(scenario())

    assert counter.counts == {}


def test_the_slot_is_free_again_after_the_replan():
    chat, slots, _ = admission(max_running=1)

    async def scenario():
        async with chat.admit("user-1"):
            pass

        async with slots.planning():
            pass

    asyncio.run(scenario())
