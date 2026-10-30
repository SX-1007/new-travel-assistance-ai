"""Output review: deterministic rules, service, Gemini normalisation, workflow."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.infrastructure.output_review.gemini_reviewer import _ReviewerOutput, to_decision
from app.modules.output_review.models import OutputReviewContext, OutputReviewDecision
from app.modules.output_review.policies import (
    claims_unconfirmed_success,
    contains_json_container,
    deterministic_output_review,
    is_read_only_itinerary_history_request,
)
from app.modules.output_review.service import OutputReviewService
from app.orchestration.state import AgentState
from app.orchestration.workflows.output_review import (
    generate_reviewed_reply,
    review_context_from_state,
    rewrite_instruction,
)


def context(reply, *, request="Plan my trip", candidate=None):
    return OutputReviewContext(
        latest_user_request=request,
        proposed_reply=reply,
        candidate_summary=candidate,
    )


def codes(decision):
    return decision.issue_codes


# ── Deterministic rules ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply",
    [
        '{"day": 1}',
        'Here you go: {"draft_itinerary": []} enjoy',
        "```json\n[1, 2]\n```",
        "[1, 2, 3]",
        "Your draft_itinerary is saved.",
        "I put it in the working itinerary summary.",
    ],
)
def test_raw_or_internal_data_is_rejected(reply):
    assert codes(deterministic_output_review(context(reply))) == ("reply.raw_internal_data",)


def test_single_number_citation_is_allowed():
    assert deterministic_output_review(context("Senso-ji opens at 6am [1].")).approved


def test_deeply_nested_brackets_do_not_crash():
    assert contains_json_container("[" * 5000) is False


def test_empty_reply_is_rejected():
    assert codes(deterministic_output_review(context("   "))) == ("reply.requirements_missing",)


@pytest.mark.parametrize(
    "reply",
    [
        "Your 5-day trip to Tokyo, Japan, is ready!",
        "I've updated your hotel for day 2.",
        "Don't worry, I've updated your hotel.",
        "I couldn't change the flight, but I replaced the hotel on day 3.",
        "The museum on day 2 has been swapped.",
    ],
)
def test_success_claims_need_a_qualified_candidate(reply):
    assert codes(deterministic_output_review(context(reply))) == ("reply.unsupported_claim",)
    assert deterministic_output_review(context(reply, candidate={"itinerary": []})).approved


@pytest.mark.parametrize(
    "reply",
    [
        "I couldn't update your itinerary because no hotel was available.",
        "Nothing was changed in your plan.",
        "I wasn't able to finalize the itinerary yet.",
        "Would you like me to add Tokyo Tower to day 2?",
    ],
)
def test_failures_and_offers_are_not_success_claims(reply):
    assert not claims_unconfirmed_success(reply)


def test_history_questions_may_describe_past_changes():
    request = "What did you change compared with the original plan?"
    reply = "Earlier I replaced the hotel on day 2."

    assert is_read_only_itinerary_history_request(request)
    assert deterministic_output_review(context(reply, request=request)).approved


def test_explicit_edit_wins_over_history_words():
    assert not is_read_only_itinerary_history_request("Replace day 2 with the previous activity")


# ── Service ─────────────────────────────────────────────────────────


class Reviewer:
    def __init__(self, result):
        self.result = result
        self.payloads = []

    def review(self, payload):
        self.payloads.append(payload)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_deterministic_failure_skips_the_model():
    reviewer = Reviewer(OutputReviewDecision.approve())

    decision = OutputReviewService(reviewer=reviewer).review(context('{"a": 1}'))

    assert codes(decision) == ("reply.raw_internal_data",)
    assert reviewer.payloads == []


def test_semantic_rejection_is_returned():
    rejection = OutputReviewDecision.reject("reply.destination_mismatch", feedback="Wrong city.")
    reviewer = Reviewer(rejection)

    decision = OutputReviewService(reviewer=reviewer).review(context("Kyoto is lovely."))

    assert decision == rejection
    assert reviewer.payloads[0]["has_qualified_candidate"] is False


@pytest.mark.parametrize(
    "result",
    [
        RuntimeError("down"),
        OutputReviewDecision.reject("review.unavailable", feedback="x"),
        "not a decision",
    ],
)
def test_reviewer_failure_or_misuse_is_unavailable(result):
    decision = OutputReviewService(reviewer=Reviewer(result)).review(context("Hello there."))

    assert codes(decision) == ("review.unavailable",)
    assert decision.unavailable


# ── Gemini normalisation ────────────────────────────────────────────


def test_model_output_is_normalised_fail_closed():
    assert to_decision(_ReviewerOutput(approved=True)).approved

    contradiction = to_decision(_ReviewerOutput(approved=True, issue_codes=["reply.unrelated"]))
    assert codes(contradiction) == ("reply.unrelated",)
    assert contradiction.feedback

    with pytest.raises(ValueError):
        to_decision(_ReviewerOutput(approved=False))


# ── Workflow: context ───────────────────────────────────────────────


def _day(n, name):
    return {
        "day": n,
        "date": f"2026-11-0{n}",
        "hotel": {"hotel_name": "Park Hotel", "price_per_night": 100.0, "image": "x.jpg",
                  "location": {"lat": 1.0, "lng": 2.0}},
        "activities": [{"name": name, "type": "attraction", "order": 1, "estimated_cost": 5.0,
                        "thumbnail": "t.png", "location": {"latitude": 1.0, "longitude": 2.0}}],
        "day_total_cost": 105.0,
    }


def state(**overrides):
    values = dict(
        messages=[
            HumanMessage(content="old question"),
            ToolMessage(content='{"stale": true}', tool_call_id="0", name="old_tool"),
            HumanMessage(content="Find me a museum"),
            AIMessage(content="", tool_calls=[{"name": "search_places", "args": {}, "id": "1"}]),
            ToolMessage(
                content=json.dumps({"places": [{"name": "Mori Museum", "thumbnail": "t.png",
                                                "gps_coordinates": {"lat": 1}}]}),
                tool_call_id="1",
                name="search_places",
            ),
        ],
        country="Japan",
        city=["Tokyo"],
        plan_revision=2,
        draft_itinerary=[_day(1, "Draft Place")],
        accepted_plan_snapshot={"plan_revision": 2, "draft_itinerary": [_day(1, "Senso-ji")]},
        itinerary_history=[{"plan_revision": 1, "draft_itinerary": [_day(1, "Old Place")]}],
        total_convert_budget=5000.0,
    )
    values.update(overrides)
    return AgentState(**values)


def test_review_context_is_compact_and_current():
    ctx = review_context_from_state(state(), "reply", candidate=None)

    assert ctx.latest_user_request == "Find me a museum"
    assert ctx.tool_evidence == [
        {"name": "search_places", "result": {"places": [{"name": "Mori Museum"}]}}
    ]
    current = ctx.trusted_requirements["current_itinerary"]
    assert current[0]["activities"] == [
        {"name": "Senso-ji", "type": "attraction", "order": 1, "estimated_cost": 5.0}
    ]
    assert current[0]["hotel"] == {"hotel_name": "Park Hotel", "price_per_night": 100.0}
    # History is only sent for history questions.
    assert "itinerary_history" not in ctx.trusted_requirements
    assert not ctx.has_qualified_candidate


def test_history_and_candidate_financials_are_included_when_relevant():
    history_state = state(messages=[HumanMessage(content="What was the original plan?")])
    candidate = {"itinerary": [_day(1, "New Place")], "financials": {"total_convert_budget": 9000.0,
                                                                    "planning_outcome": "x"}}

    ctx = review_context_from_state(history_state, "reply", candidate=candidate)

    assert ctx.trusted_requirements["itinerary_history"][0]["revision"] == 1
    assert ctx.trusted_requirements["total_convert_budget"] == 9000.0
    assert "planning_outcome" not in ctx.trusted_requirements
    assert ctx.candidate_summary["itinerary"][0]["activities"][0]["name"] == "New Place"


def test_large_tool_results_are_truncated():
    big = ToolMessage(content=json.dumps({"text": "x" * 20_000}), tool_call_id="1", name="big")
    ctx = review_context_from_state(state(messages=[HumanMessage(content="hi"), big]), "reply")

    result = ctx.tool_evidence[0]["result"]
    assert isinstance(result, str) and result.endswith("(truncated)") and len(result) < 6_100


# ── Workflow: rewrite loop ──────────────────────────────────────────


class ScriptedReview:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.seen = []

    def review(self, ctx, *, log_context=None):
        self.seen.append(ctx.proposed_reply)
        return self.decisions.pop(0)


def run_loop(reviewer, generate, **kwargs):
    return asyncio.run(
        generate_reviewed_reply(
            generate=generate,
            build_context=lambda text, attempt: context(text),
            reviewer=reviewer,
            **kwargs,
        )
    )


def test_rejected_reply_is_rewritten_with_feedback():
    rejection = OutputReviewDecision.reject("reply.unrelated", feedback="Answer the question.")
    reviewer = ScriptedReview(rejection, OutputReviewDecision.approve())
    seen_feedback = []

    async def generate(previous):
        seen_feedback.append(previous)
        return f"draft {len(seen_feedback)}"

    result = run_loop(reviewer, generate)

    assert result.reply == "draft 2" and result.attempts == 2
    assert seen_feedback == [None, rejection]
    assert result.issue_codes == ("reply.unrelated",)
    assert "reply.unrelated" in rewrite_instruction(rejection)


def test_unavailable_review_stops_at_once():
    reviewer = ScriptedReview(OutputReviewDecision.reject("review.unavailable", feedback="x"))

    async def generate(previous):
        return "draft"

    result = run_loop(reviewer, generate)

    assert result.reply is None and result.attempts == 1 and result.decision.unavailable


def test_generation_failure_stops_without_review():
    reviewer = ScriptedReview()

    async def generate(previous):
        raise RuntimeError("model down")

    result = run_loop(reviewer, generate)

    assert result.issue_codes == ("generation.unavailable",)
    assert reviewer.seen == []


def test_initial_reply_is_reviewed_first_and_attempts_are_capped():
    rejection = OutputReviewDecision.reject("reply.unrelated", feedback="x")
    reviewer = ScriptedReview(rejection, rejection)

    async def generate(previous):
        return "rewrite"

    result = run_loop(reviewer, generate, initial_reply="first", max_attempts=2)

    assert reviewer.seen == ["first", "rewrite"]
    assert result.reply is None and result.attempts == 2

    with pytest.raises(ValueError):
        run_loop(reviewer, generate, max_attempts=4)
