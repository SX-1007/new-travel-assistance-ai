"""The tool-calling agent node, the budget interpreter and the tool choice."""

from langchain_core.messages import AIMessage, HumanMessage

from app.modules.output_review.policies import claims_unconfirmed_success
from app.orchestration.edit_intent import DAY_CLARIFICATION_MARKER
from app.orchestration.nodes.agent import (
    AgentModels,
    agent_node,
    budget_decision_agent_node,
    build_agent_toolset,
)
from app.orchestration.nodes.replies import (
    AGENT_UNAVAILABLE_TEXT,
    BUDGET_DECISION_UNAVAILABLE_TEXT,
    EDIT_NOT_SAVED,
    GENERIC_UNVERIFIED,
    MUTATION_TARGET_REQUIRED,
    MUTATION_TARGET_REQUIRED_TEXT,
    SERVER_OWNED_KEY,
    mutation_choice_text,
)
from app.orchestration.tests.node_fakes import (
    CONFIG,
    REJECT,
    UNAVAILABLE,
    FakeModel,
    Network,
    committed_state,
    profiles,
    reply_services,
    run,
    tool_call,
    tool_turn,
)


NO_FACTS = object()  # No factual route this turn, so the service is never used.


def call_agent(state, *, agent=None, edit=None, decisions=(), writer=None, profile_store=None):
    models = AgentModels(agent=agent or FakeModel(), edit=edit or FakeModel(), budget=FakeModel())
    replies, semantic = reply_services(*decisions, writer=writer)
    loader, store = profiles(profile_store)
    update = run(agent_node(state, CONFIG, models=models, replies=replies, profiles=loader, factual=NO_FACTS))
    return update, models, semantic, store


def test_a_numbered_place_edit_without_a_day_is_asked_before_any_model_call():
    update, models, semantic, store = call_agent(committed_state(HumanMessage(content="replace activity 2 please")))

    [message] = update["messages"]
    assert message.additional_kwargs[SERVER_OWNED_KEY] == DAY_CLARIFICATION_MARKER
    assert update["pending_itinerary_mutation"]["source_request"] == "replace activity 2 please"
    assert models.agent.calls == [] and semantic.calls == 0 and store.calls == 0


def test_a_tool_call_is_returned_for_the_tool_node():
    call = tool_call("search_places", query="ramen")
    update, _, semantic, _ = call_agent(committed_state(HumanMessage(content="find ramen")), agent=FakeModel(call))

    assert update == {"messages": [call]}
    assert semantic.calls == 0


def test_a_model_outage_gets_fixed_text_and_no_paid_review():
    update, _, semantic, _ = call_agent(
        committed_state(HumanMessage(content="what's on day 2?")),
        agent=FakeModel(RuntimeError("quota")),
    )

    assert update["messages"][0].content == AGENT_UNAVAILABLE_TEXT
    assert update["output_review_issue_codes"] == ["model.unavailable"]
    assert semantic.calls == 0


def test_a_prose_only_edit_gets_one_retry_toward_the_tools():
    call = tool_call("search_places", query="Ueno Park")
    update, models, _, _ = call_agent(
        committed_state(HumanMessage(content="add Ueno Park on day 2")),
        agent=FakeModel("Done, I added it!", call),
    )

    assert update == {"messages": [call]}
    assert "Internal execution requirement" in models.agent.calls[1][-1].content


def test_after_a_search_the_edit_only_model_must_call_the_edit_tool():
    messages = tool_turn("add Ueno Park on day 2", "search_places", {"results": [{"name": "Ueno Park"}]}, query="Ueno Park")
    update, models, _, _ = call_agent(
        committed_state(*messages),
        agent=FakeModel("Added Ueno Park."),
        edit=FakeModel("Sure, which slot?"),
    )

    [message] = update["messages"]
    assert message.content == mutation_choice_text(["Ueno Park"], day=2)
    assert message.additional_kwargs[SERVER_OWNED_KEY] == MUTATION_TARGET_REQUIRED
    assert len(models.edit.calls) == 1


def test_an_open_ended_swap_the_model_did_not_apply_offers_the_unplanned_results():
    # The live case (B12): the user left the new place open, the search
    # found options, and the edit model still made no call.
    results = [{"name": name} for name in ("Tokyo Tower", "Senso-ji", "Tokyo Skytree", "Ueno Zoo", "Ginza Six")]
    messages = tool_turn(
        "Please remove the first activity on day 2 and replace it with something else.",
        "search_places",
        {"results": results},
        query="attraction in Tokyo",
    )
    update, _, _, _ = call_agent(
        committed_state(*messages),
        agent=FakeModel("Which one would you like?"),
        edit=FakeModel("Which one would you like?"),
    )

    reply = update["messages"][0].content
    # Places already in the plan are not offered; at most three are.
    assert reply.startswith("I found Tokyo Skytree, Ueno Zoo or Ginza Six,")
    assert "Tokyo Tower" not in reply and "Senso-ji" not in reply
    # The example names the user's day, so replying with it authorizes the edit.
    assert '"On day 2, replace activity 1 with Tokyo Skytree."' in reply
    assert update["output_review_issue_codes"] == ["mutation.tool_call_missing"]


def test_without_new_options_the_target_question_is_generic():
    messages = tool_turn(
        "replace the first activity on day 2 with something else",
        "search_places",
        {"results": [{"name": "Tokyo Tower"}]},
        query="attraction",
    )
    update, _, _, _ = call_agent(
        committed_state(*messages),
        agent=FakeModel("Which one?"),
        edit=FakeModel("Which one?"),
    )

    assert update["messages"][0].content == MUTATION_TARGET_REQUIRED_TEXT


def test_the_choice_question_passes_the_no_false_success_rule():
    # server_reply swaps in a generic fallback when the deterministic rules
    # reject fixed text; the question must never read as a saved change.
    for day in (None, 3):
        assert not claims_unconfirmed_success(mutation_choice_text(["Ueno Zoo", "Ginza Six"], day=day))

    assert not claims_unconfirmed_success(MUTATION_TARGET_REQUIRED_TEXT)
    assert mutation_choice_text([]) == MUTATION_TARGET_REQUIRED_TEXT
    assert mutation_choice_text(["Ueno Zoo"]).startswith("I found Ueno Zoo, and")


def test_a_rejected_reply_is_rewritten_once_with_the_feedback():
    writer = FakeModel("Day 2 has Tokyo Tower.")
    update, _, semantic, _ = call_agent(
        committed_state(HumanMessage(content="what's on day 2?")),
        agent=FakeModel("Day 2 has Tokyo Tower and I booked it."),
        decisions=(REJECT,),
        writer=writer,
    )

    assert update["messages"] == [AIMessage(content="Day 2 has Tokyo Tower.")]
    assert update["output_review_attempts"] == 2
    assert semantic.calls == 2 and len(writer.calls) == 1


def test_unverified_general_answers_get_the_general_fallback():
    update, _, _, _ = call_agent(
        committed_state(HumanMessage(content="what should I pack?")),
        agent=FakeModel("Pack light."),
        decisions=(UNAVAILABLE,),
    )

    # Not the "historical trip detail" text: that one is for history questions.
    assert update["messages"][0].content == GENERIC_UNVERIFIED


def test_unverified_edit_answers_say_nothing_was_saved():
    update, _, _, _ = call_agent(
        committed_state(HumanMessage(content="swap the hotel please")),
        agent=FakeModel("Which hotel?", "Which hotel would you like?"),
        decisions=(UNAVAILABLE,),
    )

    assert update["messages"][0].content == EDIT_NOT_SAVED


def test_the_profile_is_fetched_once_across_agent_passes():
    loader_state = committed_state(HumanMessage(content="find ramen"))
    models = AgentModels(agent=FakeModel(tool_call("search_places"), tool_call("search_places")), edit=FakeModel(), budget=FakeModel())
    replies, _ = reply_services()
    loader, store = profiles()

    for _ in range(2):
        run(agent_node(loader_state, CONFIG, models=models, replies=replies, profiles=loader, factual=NO_FACTS))

    assert store.calls == 1


# ── Budget interpreter ──────────────────────────────────────────────


def test_budget_interpreter_outage_gets_fixed_text():
    models = AgentModels(agent=FakeModel(), edit=FakeModel(), budget=FakeModel(RuntimeError("down")))
    replies, semantic = reply_services()
    state = committed_state(HumanMessage(content="yes"), pending_budget_confirmation={"budget_assessment_id": "a"})

    update = run(budget_decision_agent_node(state, CONFIG, models=models, replies=replies))

    assert update["messages"][0].content == BUDGET_DECISION_UNAVAILABLE_TEXT
    assert semantic.calls == 0


def test_budget_interpreter_tool_call_passes_through():
    call = tool_call("confirm_recommended_budget")
    models = AgentModels(agent=FakeModel(), edit=FakeModel(), budget=FakeModel(call))
    replies, _ = reply_services()
    state = committed_state(HumanMessage(content="yes"), pending_budget_confirmation={"budget_assessment_id": "a"})

    assert run(budget_decision_agent_node(state, CONFIG, models=models, replies=replies)) == {"messages": [call]}


# ── Tool choice ─────────────────────────────────────────────────────


def test_bound_tools_exclude_the_gate_bypass_and_the_duplicate_edit_path():
    network = Network()
    toolset = build_agent_toolset(
        booking=network.planning().booking,
        place_search=network.edits().place_search,
        amenities=object(),
        currency=object(),
    )
    names = {tool.name for tool in toolset.agent}

    assert "update_total_budget" not in names and "modify_existing_booking" not in names
    assert {"edit_itinerary", "propose_budget_change", "update_trip_details"} <= names
    assert [tool.name for tool in toolset.edit] == ["edit_itinerary"]
    assert {tool.name for tool in toolset.budget_decision} == {"propose_budget_change", "confirm_recommended_budget"}

    models = AgentModels.bind(FakeModel(), toolset)
    assert set(models.agent.tools) == names and models.edit.tools == ["edit_itinerary"]
