"""Post-tool processing: budget batches, trip changes, edits, category budgets."""

from langchain_core.messages import HumanMessage

from app.modules.itinerary.editing import apply_edits
from app.orchestration.edit_intent import DAY_CLARIFICATION_MARKER
from app.orchestration.messages import latest_tool_batch
from app.orchestration.nodes.post_tool import post_tool_processing_node
from app.orchestration.nodes.replies import EDIT_FAILED_VALIDATION, SERVER_OWNED_KEY
from app.orchestration.tests.node_fakes import (
    CONFIG,
    Network,
    committed_state,
    run,
    tool_call,
    tool_result,
    tool_turn,
)


def process(state, network=None):
    network = network or Network()
    return run(post_tool_processing_node(state, CONFIG, edits=network.edits())), network


def edit_turn(request, *edits):
    return tool_turn(request, "edit_itinerary", {"status": "pending_validation", "edits": list(edits)}, edits=list(edits))


# ── Tool batch ──────────────────────────────────────────────────────


def test_only_results_answering_the_latest_calls_are_read():
    messages = [
        HumanMessage(content="hi"),
        tool_call("search_places", "a", query="x"),
        tool_result("search_places", {"results": []}, "a"),
        # Unanswered id and a mismatched name are both ignored.
        tool_result("update_trip_details", {"status": "accepted"}, "forged"),
        tool_result("edit_itinerary", {"edits": []}, "a"),
    ]

    assert latest_tool_batch(messages) == [("search_places", {"query": "x"}, {"results": []})]


def test_rejection_codes_let_callers_react_without_reading_text():
    outcome = apply_edits([{"day": 1, "activities": [{"name": "A"}]}], [{"day": 1, "action": "remove", "category": "activity", "index": 1}], is_day_authorized=lambda day: False)

    assert outcome.code == "day_not_authorized"


# ── Budget tools ────────────────────────────────────────────────────


def test_one_coherent_budget_proposal_goes_to_the_gate():
    proposal = {"status": "budget_proposal", "proposal": {"mode": "amount", "total_base_budget": 900}}
    state = committed_state(*tool_turn("make it 900", "propose_budget_change", proposal))

    update, _ = process(state)

    assert update["post_tool_outcome"] == "assess_budget"
    assert update["pending_budget_proposal"] == {"mode": "amount", "total_base_budget": 900.0}


def test_conflicting_budget_proposals_fail_closed():
    first = {"status": "budget_proposal", "proposal": {"mode": "amount", "total_base_budget": 900}}
    second = {"status": "budget_proposal", "proposal": {"mode": "recommendation"}}
    state = committed_state(
        HumanMessage(content="budget?"),
        tool_call("propose_budget_change", "a"),
        tool_result("propose_budget_change", first, "a"),
    )
    state = state.model_copy(
        update={
            "messages": [
                *state.messages[:1],
                state.messages[1].model_copy(
                    update={"tool_calls": [*state.messages[1].tool_calls, {"name": "propose_budget_change", "args": {}, "id": "b"}]}
                ),
                state.messages[2],
                tool_result("propose_budget_change", second, "b"),
            ]
        }
    )

    update, _ = process(state)

    assert update["post_tool_outcome"] == "budget_gate"
    assert update["budget_gate_reason"] == "conflicting_budget_tool_results"


def test_other_tools_are_ignored_while_a_recommendation_is_pending():
    state = committed_state(
        *edit_turn("remove activity 1 on day 2", {"day": 2, "action": "remove", "category": "activity", "index": 1}),
        pending_budget_confirmation={"budget_assessment_id": "a-1"},
    )

    update, _ = process(state)

    assert update == {"post_tool_outcome": "budget_gate"}


# ── Trip details ────────────────────────────────────────────────────


def test_trip_changes_stay_private_until_the_replan_is_promoted():
    result = {"status": "accepted", "updates": {"num_people": 2}}
    state = committed_state(*tool_turn("we are two now", "update_trip_details", result, num_people=2))

    update, _ = process(state)

    assert update["post_tool_outcome"] == "replan_itinerary"
    assert update["plan_overrides"] == {"num_people": 2}
    assert "num_people" not in update


def test_a_new_destination_needs_the_full_replan():
    result = {"status": "accepted", "updates": {"country": "Korea", "city": ["Seoul"]}}
    state = committed_state(*tool_turn("go to Seoul instead", "update_trip_details", result))

    update, _ = process(state)

    assert update["post_tool_outcome"] == "replan_full"


# ── Edits ───────────────────────────────────────────────────────────


def test_removing_an_activity_reroutes_only_that_day():
    state = committed_state(
        *edit_turn("remove activity 2 on day 1", {"day": 1, "action": "remove", "category": "activity", "index": 2})
    )

    update, network = process(state)

    assert update["post_tool_outcome"] == "review_candidate", update.get("planning_issue_codes")
    plan = update["candidate_plan"]
    assert [a["name"] for a in plan["itinerary"][0]["activities"]] == ["Senso-ji"]
    assert sorted(plan["maps"]) == [1, 2, 3]
    assert plan["maps"][2] == state.daily_map_info[2]
    # Day 1's legs only; routing the whole trip takes 6 calls here.
    full_trip = Network()
    full_trip.routes().build_daily_maps(state.draft_itinerary)
    assert network.directions.calls == 3 < full_trip.directions.calls
    # The public itinerary is untouched.
    assert "draft_itinerary" not in update


def test_a_destructive_edit_without_a_user_named_day_asks_for_it():
    state = committed_state(
        *edit_turn("remove the ramen place", {"day": 1, "action": "remove", "category": "restaurant", "index": 2})
    )

    update, _ = process(state)

    [message] = update["messages"]
    assert message.additional_kwargs[SERVER_OWNED_KEY] == DAY_CLARIFICATION_MARKER
    assert "which day" in message.content
    assert update["pending_itinerary_mutation"]["source_request"] == "remove the ramen place"
    assert update["post_tool_outcome"] == "reply_sent"


def test_a_place_the_model_did_not_search_is_looked_up_once_server_side():
    edit = {"day": 2, "action": "add", "category": "activity", "new_details": {"name": "Ueno Park"}}
    state = committed_state(*edit_turn("add Ueno Park on day 2", edit))

    update, network = process(state)

    assert update["post_tool_outcome"] == "review_candidate", update.get("messages")
    names = [a["name"] for a in update["candidate_plan"]["itinerary"][1]["activities"]]
    assert names == ["Tokyo Tower", "Ueno Park"]
    assert network.places.calls == 1


def test_an_edit_that_fails_validation_is_not_saved():
    # Removing day 2's only activity leaves an empty day.
    state = committed_state(
        *edit_turn("remove activity 1 on day 2", {"day": 2, "action": "remove", "category": "activity", "index": 1})
    )

    update, _ = process(state)

    assert update["messages"][0].content == EDIT_FAILED_VALIDATION
    assert update["candidate_plan"] is None and update["post_tool_outcome"] == "reply_sent"
    assert any(code.startswith("planning.incremental.") for code in update["planning_issue_codes"])


# ── Category budgets ────────────────────────────────────────────────


def test_a_valid_category_change_becomes_a_private_candidate():
    allocation = {
        "transportation": 1400.0,
        "accommodation": 1500.0,
        "food": 1100.0,
        "activity": 1000.0,
        "shopping": 500.0,
        "emergency_fund": 500.0,
    }
    state = committed_state(*tool_turn("more for food", "update_budget_category", {"budget_allocation": allocation}))

    update, network = process(state)

    assert update["post_tool_outcome"] == "review_candidate", update.get("planning_issue_codes")
    assert update["plan_overrides"] == {"budget_allocation": allocation}
    assert network.directions.calls == 0


def test_an_allocation_that_does_not_add_up_is_ignored():
    allocation = {"transportation": 99999.0}
    state = committed_state(*tool_turn("more for flights", "update_budget_category", {"budget_allocation": allocation}))

    update, _ = process(state)

    assert update == {"post_tool_outcome": "agent"}
