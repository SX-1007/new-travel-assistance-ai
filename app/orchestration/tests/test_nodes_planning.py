"""Planning nodes: currency setup, transaction, reviewed reply, promotion."""

import pytest
from langchain_core.messages import HumanMessage

from app.modules.itinerary.service import PLANNING_MAX_ATTEMPTS
from app.modules.output_review.service import OutputReviewService
from app.orchestration.nodes.planning import (
    execute_validated_transaction_node,
    generate_reviewed_response_node,
    process_initial_form_node,
    promote_candidate_node,
)
from app.orchestration.nodes.replies import (
    CHANGE_NOT_SAVED,
    EDIT_NOT_VERIFIED,
    NEW_PLAN_UNAVAILABLE,
    QUALIFIED_FALLBACKS,
    SERVER_OWNED_KEY,
)
from app.orchestration.nodes.transaction import PlanKind
from app.orchestration.tests.node_fakes import (
    CONFIG,
    REJECT,
    UNAVAILABLE,
    FakeBudget,
    FakeCurrency,
    FakeModel,
    Network,
    ScriptedReviewer,
    assessment,
    candidate,
    committed_state,
    profiles,
    reply_services,
    run,
    tool_turn,
    trip_state,
)


def validated(state, plan=None, **overrides):
    return state.model_copy(
        update={"planning_outcome": "validated", "candidate_plan": plan or candidate(), "planning_attempts": 1, **overrides}
    )


def approved(plan, reply="Your 3-day Tokyo trip is planned with Senso-ji and Tokyo Tower."):
    return {**plan, "reply": reply, "review": {"approved": True, "issue_codes": [], "feedback": ""}}


def reviewer():
    return OutputReviewService(reviewer=ScriptedReviewer())


# ── process_initial_form ────────────────────────────────────────────


def test_new_trip_reuses_the_preflight_rate_without_a_provider_call():
    currency = FakeCurrency()
    state = trip_state(budget_assessment=assessment(), base_currency_code=None, dest_currency_code=None)

    update = run(process_initial_form_node(state, CONFIG, planning=Network().planning(currency=currency)))

    assert currency.calls == 0
    assert (update["base_currency_code"], update["dest_currency_code"]) == ("MYR", "JPY")
    assert update["total_convert_budget"] == 600.0 * 32.0
    assert round(sum(update["budget_allocation"].values()), 2) == 19200.0
    assert update["currency_fetched_at"].endswith("Z")
    assert update["planning_outcome"] is None and update["candidate_plan"] is None
    assert "plan_overrides" not in update


def test_new_destination_on_a_committed_trip_stays_private():
    currency = FakeCurrency()
    state = committed_state(plan_overrides={"country": "Korea", "city": ["Seoul"]})

    update = run(
        process_initial_form_node(state, CONFIG, planning=Network().planning(currency=currency, budget=FakeBudget(matches=False)))
    )

    assert currency.calls == 2
    assert "dest_currency_code" not in update and "budget_allocation" not in update
    overrides = update["plan_overrides"]
    assert overrides["country"] == "Korea" and overrides["dest_currency_code"] == "KRW"
    assert overrides["total_convert_budget"] == 6000.0


def test_currency_failure_is_a_planning_outcome_not_a_crash():
    state = trip_state(base_currency_code=None)

    update = run(
        process_initial_form_node(state, CONFIG, planning=Network().planning(currency=FakeCurrency(fail=True), budget=FakeBudget(matches=False)))
    )

    assert update["planning_outcome"] == "unavailable"
    assert update["planning_attempts"] == PLANNING_MAX_ATTEMPTS
    assert update["planning_issue_codes"] == ["currency.unavailable"]


# ── execute_validated_transaction ───────────────────────────────────


def test_transaction_plans_against_the_private_overrides_and_loads_the_profile_once():
    loader, store = profiles()
    # Public trip is 2 days; the pending change makes it 3.
    state = trip_state(end_date="2026-11-02", plan_overrides={"end_date": "2026-11-03"})

    update = run(execute_validated_transaction_node(state, CONFIG, planning=Network().planning(), profiles=loader))
    run(execute_validated_transaction_node(state, CONFIG, planning=Network().planning(), profiles=loader))

    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]
    assert [day["day"] for day in update["candidate_plan"]["itinerary"]] == [1, 2, 3]
    assert "end_date" not in update
    assert store.calls == 1


# ── generate_reviewed_response ──────────────────────────────────────


def test_approved_reply_is_stored_on_the_candidate():
    replies, semantic = reply_services(writer=FakeModel("Your Tokyo trip has Senso-ji on day 1."))
    loader, _ = profiles()

    update = run(generate_reviewed_response_node(validated(trip_state()), CONFIG, replies=replies, profiles=loader))

    plan = update["candidate_plan"]
    assert plan["reply"] == "Your Tokyo trip has Senso-ji on day 1."
    assert plan["review"]["approved"] and plan["reply_source"] == "reviewed"
    assert semantic.calls == 1 and update["output_review_attempts"] == 1


def test_reviewer_outage_keeps_the_validated_plan_with_the_fallback_text():
    writer = FakeModel("Your plan is ready.")
    replies, semantic = reply_services(UNAVAILABLE, writer=writer)
    loader, _ = profiles()

    update = run(generate_reviewed_response_node(validated(trip_state()), CONFIG, replies=replies, profiles=loader))

    plan = update["candidate_plan"]
    assert plan["reply"] == QUALIFIED_FALLBACKS[PlanKind.NEW]
    assert plan["reply_source"] == "server_fallback"
    # No rewrite after an outage: that can't fix it.
    assert len(writer.calls) == 1 and semantic.calls == 1


def test_rejected_drafts_are_rewritten_with_the_feedback():
    writer = FakeModel("Everything is booked and paid.", "Your Tokyo trip covers three days.")
    replies, _ = reply_services(REJECT, writer=writer)
    loader, _ = profiles()

    update = run(generate_reviewed_response_node(validated(trip_state()), CONFIG, replies=replies, profiles=loader))

    assert update["candidate_plan"]["reply"] == "Your Tokyo trip covers three days."
    assert "Only state supported facts." in writer.calls[1][-1].content
    assert update["output_review_issue_codes"] == ["reply.unsupported_claim"]


def test_failed_transaction_replies_without_a_model_and_drops_the_overrides():
    writer = FakeModel()
    replies, semantic = reply_services(writer=writer)
    loader, _ = profiles()
    state = committed_state(
        HumanMessage(content="make it 4 days"),
        planning_outcome="unavailable",
        planning_issue_codes=["day.activities.empty"],
        plan_overrides={"end_date": "2026-11-04"},
    )

    update = run(generate_reviewed_response_node(state, CONFIG, replies=replies, profiles=loader))

    [message] = update["messages"]
    assert message.content == CHANGE_NOT_SAVED
    assert message.additional_kwargs[SERVER_OWNED_KEY] == "planning_unavailable"
    assert update["plan_overrides"] is None and update["candidate_plan"] is None
    assert writer.calls == [] and semantic.calls == 0


def test_failed_first_plan_says_nothing_was_saved_yet():
    replies, _ = reply_services()
    loader, _ = profiles()
    state = trip_state(planning_outcome="unavailable")

    update = run(generate_reviewed_response_node(state, CONFIG, replies=replies, profiles=loader))

    assert update["messages"][0].content == NEW_PLAN_UNAVAILABLE


@pytest.mark.parametrize("kind", list(PlanKind))
def test_every_qualified_fallback_passes_the_rules_with_a_candidate(kind):
    from app.orchestration.workflows.output_review import review_context_from_state

    context = review_context_from_state(trip_state(), QUALIFIED_FALLBACKS[kind], candidate=candidate())

    assert reviewer().check(context).approved


# ── promote_candidate ───────────────────────────────────────────────


def test_promotion_commits_plan_and_overrides_together():
    state = validated(
        committed_state(HumanMessage(content="I'd like two travellers")),
        approved(candidate()),
        plan_overrides={"num_people": 2},
    )

    update = run(promote_candidate_node(state, CONFIG, reviewer=reviewer()))

    assert update["plan_revision"] == 2 and update["num_people"] == 2
    assert update["messages"][0].additional_kwargs == {"plan_revision": 2}
    assert [entry["revision"] for entry in update["itinerary_history"]] == [1]
    snapshot = update["accepted_plan_snapshot"]
    assert snapshot["plan_revision"] == 2
    assert snapshot["draft_itinerary"] == update["draft_itinerary"]
    assert snapshot["draft_itinerary"] is not update["draft_itinerary"]
    assert update["plan_overrides"] is None and update["candidate_plan"] is None


def test_promotion_refuses_an_unapproved_reply_and_keeps_the_public_plan():
    plan = {**approved(candidate()), "review": {"approved": False, "issue_codes": ["reply.unrelated"], "feedback": "x"}}
    state = validated(committed_state(HumanMessage(content="plan it")), plan)

    update = run(promote_candidate_node(state, CONFIG, reviewer=reviewer()))

    assert "draft_itinerary" not in update and "plan_revision" not in update
    assert update["messages"][0].content == CHANGE_NOT_SAVED
    assert "promotion.candidate.invalid" in update["planning_issue_codes"]


def test_promotion_revalidates_against_the_overrides_it_would_publish():
    # The plan has 3 days; publishing it with a 4-day range must fail.
    state = validated(
        committed_state(HumanMessage(content="add a day")),
        approved(candidate()),
        plan_overrides={"end_date": "2026-11-04"},
    )

    update = run(promote_candidate_node(state, CONFIG, reviewer=reviewer()))

    assert "plan_revision" not in update and "end_date" not in update
    assert update["plan_overrides"] is None


def test_failed_edit_promotion_keeps_the_existing_itinerary():
    messages = tool_turn("remove activity 2 on day 1", "edit_itinerary", {"edits": []})
    plan = approved(candidate(), reply="")
    state = validated(committed_state(*messages), plan)

    update = run(promote_candidate_node(state, CONFIG, reviewer=reviewer()))

    assert update["messages"][0].content == EDIT_NOT_VERIFIED
    assert update["messages"][0].additional_kwargs[SERVER_OWNED_KEY] == "incremental_edit_rejected"
    assert all(code.startswith("planning.incremental.") for code in update["planning_issue_codes"])
