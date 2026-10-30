"""The compiled graph end to end: real nodes, planner, router and validator,
an in-memory checkpointer, and fakes only at the network and model edges."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.modules.factual_qa.models import FactualDomain, FactualRoute, RequestType
from app.modules.itinerary.tests.test_planning_end_to_end import _idea
from app.orchestration.graph import GraphRuntime, compile_graph
from app.orchestration.nodes import GraphServices, build_nodes
from app.orchestration.nodes.agent import MAX_TOOL_ROUNDS, AgentModels, build_agent_toolset
from app.orchestration.nodes.replies import CHANGE_NOT_SAVED
from app.orchestration.session_access import SessionAccessService
from app.orchestration.tests.node_fakes import (
    FakeBudget,
    FakeCurrency,
    FakeModel,
    Network,
    assessment,
    profiles,
    reply_services,
    tool_call,
)
from app.shared.exceptions import SessionAccessDenied


class OwnershipStore:
    """In-memory ``SessionOwnershipStore``."""

    def __init__(self):
        self.rows = {}
        self.released = []

    def reserve(self, session_id, user_id):
        if session_id in self.rows:
            return False
        self.rows[session_id] = [user_id, "initializing"]
        return True

    def complete(self, session_id, user_id):
        row = self.rows.get(session_id)
        if row != [user_id, "initializing"]:
            return False
        row[1] = "ready"
        return True

    def reclaim_expired(self, session_id, user_id, lease_seconds):
        return False

    def finalize_expired(self, session_id, user_id, lease_seconds):
        return False

    def release(self, session_id, user_id):
        self.released.append(session_id)
        if self.rows.get(session_id) == [user_id, "initializing"]:
            del self.rows[session_id]

    def backfill(self, session_id, user_id):
        self.rows[session_id] = [user_id, "ready"]
        return True

    def verify(self, session_id, user_id):
        return self.rows.get(session_id) == [user_id, "ready"]


class ChatOnlyFactual:
    """Factual service that routes every message to the agent."""

    def route(self, message, trip):
        return FactualRoute(request_type=RequestType.ACTION, domain=FactualDomain.TRIP_STATE, factual_question=message)


class App:
    def __init__(self, *, agent=(), writer=None):
        self.network = Network()
        self.agent = FakeModel(*agent)
        self.writer = writer or FakeModel(*["Here is your Tokyo trip with Senso-ji and Tokyo Tower."] * 6)
        self.store = OwnershipStore()
        replies, self.semantic = reply_services(writer=self.writer)
        loader, _ = profiles()
        planning = self.network.planning(budget=FakeBudget(), currency=FakeCurrency())
        toolset = build_agent_toolset(
            booking=planning.booking,
            place_search=self.network.edits().place_search,
            amenities=object(),
            currency=FakeCurrency(),
        )
        services = GraphServices(
            planning=planning,
            replies=replies,
            models=AgentModels(agent=self.agent, edit=FakeModel(), budget=FakeModel()),
            edits=self.network.edits(),
            profiles=loader,
            factual=ChatOnlyFactual(),
        )
        self.saver = MemorySaver()
        self.runtime = GraphRuntime(
            workflow=compile_graph(build_nodes(services), toolset, checkpointer=self.saver),
            checkpointer=self.saver,
            sessions=SessionAccessService(store=self.store),
        )

    def new_trip(self, thread="trip-1", user="user-1"):
        initial = {
            "messages": [HumanMessage(content="Plan my trip to Tokyo")],
            "origin_country": "Malaysia",
            "country": "Japan",
            "city": ["Tokyo"],
            "num_people": 1,
            "total_base_budget": 600.0,
            "start_date": "2026-11-01",
            "end_date": "2026-11-03",
            "budget_assessment": assessment(),
        }
        return asyncio.run(self.runtime.invoke_new_trip(initial, thread_id=thread, user_id=user))

    def chat(self, message, thread="trip-1", user="user-1"):
        return asyncio.run(self.runtime.invoke_chat(message, thread_id=thread, user_id=user))


def edit_call(*edits):
    return tool_call("edit_itinerary", edits=list(edits))


def test_a_new_trip_is_planned_reviewed_and_published():
    app = App()

    result = app.new_trip()

    assert result.plan_changed
    state = result.state
    assert state["plan_revision"] == 1 and state["dest_currency_code"] == "JPY"
    assert [day["day"] for day in state["draft_itinerary"]] == [1, 2, 3]
    assert state["messages"][-1].content.startswith("Here is your Tokyo trip")
    assert state["accepted_plan_snapshot"]["plan_revision"] == 1
    assert app.store.rows["trip-1"] == ["user-1", "ready"]


def test_a_chat_edit_runs_agent_tools_review_and_promotion():
    app = App(agent=[edit_call({"day": 1, "action": "remove", "category": "activity", "index": 2})])
    app.new_trip()

    result = app.chat("remove activity 2 on day 1")

    assert result.plan_changed
    state = result.state
    assert state["plan_revision"] == 2
    assert [a["name"] for a in state["draft_itinerary"][0]["activities"]] == ["Senso-ji"]
    assert [entry["revision"] for entry in state["itinerary_history"]] == [1]
    assert state["post_tool_outcome"] is None and state["candidate_plan"] is None


def test_a_date_change_is_replanned_and_published_with_the_plan():
    app = App(agent=[tool_call("update_trip_details", end_date="2026-11-04")])
    app.new_trip()
    app.network.ideas.initial.append(_idea("Ueno Park", 4, 1))

    state = app.chat("make it four days").state

    assert state["end_date"] == "2026-11-04" and state["plan_revision"] == 2
    assert [day["day"] for day in state["draft_itinerary"]] == [1, 2, 3, 4]


def test_a_failed_replan_leaves_the_published_requirements_alone():
    # Tokyo can't be planned as a Korea trip: the transaction fails.
    app = App(agent=[tool_call("update_trip_details", country="Korea")])
    app.new_trip()

    result = app.chat("actually make it Korea")

    assert not result.plan_changed
    state = result.state
    assert state["country"] == "Japan" and state["dest_currency_code"] == "JPY"
    assert state["plan_revision"] == 1 and state["plan_overrides"] is None
    assert state["messages"][-1].content == CHANGE_NOT_SAVED


def test_the_tool_loop_is_bounded_and_still_answers():
    searches = [tool_call("search_places", f"c{n}", category="attraction", query="museum") for n in range(10)]
    app = App(agent=searches)
    app.new_trip()

    result = app.chat("find me a museum")

    assert app.agent.responses == searches[MAX_TOOL_ROUNDS:]
    assert isinstance(result.state["messages"][-1], AIMessage)
    assert not result.plan_changed


def test_a_finished_unavailable_budget_check_does_not_lock_the_chat():
    app = App(agent=["Day 2 has Tokyo Tower."], writer=FakeModel("Here is your Tokyo trip.", "Day 2 has Tokyo Tower."))
    app.new_trip()
    asyncio.run(
        app.runtime._workflow.aupdate_state(
            {"configurable": {"thread_id": "trip-1"}},
            {"budget_gate_outcome": "budget_check_unavailable"},
        )
    )

    state = app.chat("what's on day 2?").state

    assert state["messages"][-1].content == "Day 2 has Tokyo Tower."
    assert state["budget_gate_outcome"] is None


def test_sessions_are_private_to_their_owner():
    app = App()
    app.new_trip()

    with pytest.raises(SessionAccessDenied):
        app.chat("hi", user="someone-else")

    with pytest.raises(SessionAccessDenied):
        app.chat("hi", thread="no-such-trip")

    # An existing id can't be planned over, not even by its owner.
    with pytest.raises(SessionAccessDenied):
        app.new_trip()
