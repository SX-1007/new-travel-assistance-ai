"""Turn start, background memory extraction, profiles, factual nodes, registry."""

import asyncio
import inspect
import threading

from langchain_core.messages import HumanMessage

from app.orchestration.nodes import GraphServices, build_nodes
from app.orchestration.nodes.agent import AgentModels
from app.orchestration.nodes import factual as factual_nodes
from app.orchestration.nodes.factual import ROUTER_UNAVAILABLE, factual_answer_node, factual_router_node
from app.orchestration.nodes.turn import MemoryExtractionQueue, begin_chat_turn_node
from app.orchestration.tests.node_fakes import (
    CONFIG,
    FakeDestinations,
    FakeModel,
    Network,
    ProfileStore,
    candidate,
    committed_state,
    profiles,
    reply_services,
    run,
)


def test_turn_start_discards_any_unfinished_transaction():
    state = committed_state(
        HumanMessage(content="hello"),
        planning_outcome="validated",
        candidate_plan=candidate(),
        plan_overrides={"num_people": 3},
        factual_issue_codes=["fact.stale"],
    )

    update = run(begin_chat_turn_node(state, CONFIG))

    assert update["candidate_plan"] is None and update["planning_outcome"] is None
    assert update["plan_overrides"] is None and update["factual_issue_codes"] == []
    # A trip change can't outlive the recommendation it waited on.
    assert update["pending_trip_change"] is None


def test_memory_extraction_does_not_hold_up_the_turn():
    release = threading.Event()
    seen = []

    def extract(user_id, message):
        release.wait(timeout=5)
        seen.append((user_id, message))
        return {"memory_updated": True}

    async def scenario():
        queue = MemoryExtractionQueue(extract)
        await begin_chat_turn_node(committed_state(HumanMessage(content="I'm vegetarian")), CONFIG, memory=queue)
        running = queue.in_flight
        release.set()
        await queue.drain(timeout=5)
        return running

    assert run(scenario()) == 1
    assert seen == [("user-1", "I'm vegetarian")]


def test_extraction_beyond_the_bound_is_skipped_not_queued():
    release = threading.Event()

    async def scenario():
        queue = MemoryExtractionQueue(lambda user, message: release.wait(timeout=5), max_in_flight=1)
        accepted = [queue.submit("u", "first"), queue.submit("u", "second")]
        release.set()
        await queue.drain(timeout=5)
        return accepted

    assert run(scenario()) == [True, False]


def test_saved_memory_refreshes_the_cached_profile():
    loader, store = profiles()

    async def scenario():
        queue = MemoryExtractionQueue(lambda user, message: {"memory_updated": True}, on_saved=loader.invalidate)
        await loader.load("user-1")
        await loader.load("user-1")
        queue.submit("user-1", "I love hiking")
        await queue.drain(timeout=5)
        await loader.load("user-1")

    run(scenario())
    assert store.calls == 2


def test_a_profile_outage_is_not_cached_and_not_fatal():
    loader, store = profiles(ProfileStore(fail=True))

    assert run(loader.load("user-1")) is None
    assert run(loader.load("user-1")) is None
    assert store.calls == 2
    assert run(loader.load(None)) is None


# ── Factual nodes ───────────────────────────────────────────────────


class BrokenFactual:
    def route(self, *args):
        raise RuntimeError("boom")

    def answer(self, *args):
        raise RuntimeError("boom")


def test_router_failure_is_marked_for_the_fail_closed_route():
    update = run(factual_router_node(committed_state(HumanMessage(content="is it open?")), CONFIG, factual=BrokenFactual()))

    assert update["factual_route"] is None
    assert update["factual_issue_codes"] == [ROUTER_UNAVAILABLE]


def test_answer_failure_does_not_guess(monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(factual_nodes, "run_factual_answer", broken)

    update = run(factual_answer_node(committed_state(HumanMessage(content="is it open?")), CONFIG, factual=BrokenFactual()))

    assert update["messages"][0].content.startswith("I couldn't verify that right now")
    assert update["factual_issue_codes"] == ["fact.answer_unavailable"]


# ── Registry ────────────────────────────────────────────────────────


def test_every_node_is_bound_with_the_langgraph_signature():
    network = Network()
    replies, _ = reply_services()
    loader, _ = profiles()
    nodes = build_nodes(
        GraphServices(
            planning=network.planning(),
            replies=replies,
            models=AgentModels(agent=FakeModel(), edit=FakeModel(), budget=FakeModel()),
            edits=network.edits(),
            profiles=loader,
            factual=BrokenFactual(),
            destinations=FakeDestinations(),
        )
    )

    assert set(nodes) == {
        "process_initial_form",
        "prepare_planning_transaction",
        "execute_validated_transaction",
        "generate_reviewed_response",
        "promote_candidate",
        "begin_chat_turn",
        "factual_router",
        "factual_retrieval",
        "factual_answer",
        "agent",
        "budget_decision_agent",
        "assess_chat_budget",
        "resume_accepted_budget",
        "budget_gate_response",
        "post_tool_processing",
        "check_trip_change",
    }

    for name, node in nodes.items():
        if name != "prepare_planning_transaction":
            assert asyncio.iscoroutinefunction(node)
            assert list(inspect.signature(node).parameters) == ["state", "config"]
