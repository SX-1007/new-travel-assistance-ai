"""Replies when a paid search provider's quota is used up: honest text,
never "try again shortly", and accepted by the deterministic reply rules."""

import pytest

from app.modules.output_review.service import OutputReviewService
from app.orchestration.nodes.replies import (
    QUOTA_BUDGET_TEXT,
    QUOTA_CHANGE_TEXT,
    QUOTA_NEW_PLAN_TEXT,
    planning_failure_text,
    server_reply,
)
from app.orchestration.nodes.transaction import PlanKind
from app.orchestration.state import AgentState


def test_a_new_trip_stopped_by_the_quota_says_so():
    state = AgentState(planning_issue_codes=["provider.quota_exhausted"])

    assert planning_failure_text(state, PlanKind.NEW) == QUOTA_NEW_PLAN_TEXT


def test_a_replan_stopped_by_the_quota_says_the_plan_is_kept():
    state = AgentState(plan_revision=1, planning_issue_codes=["provider.quota_exhausted"])

    assert planning_failure_text(state, PlanKind.REQUIREMENTS) == QUOTA_CHANGE_TEXT


@pytest.mark.parametrize("text", [QUOTA_NEW_PLAN_TEXT, QUOTA_CHANGE_TEXT, QUOTA_BUDGET_TEXT])
def test_the_quota_texts_pass_the_reply_rules(text):
    reviewer = OutputReviewService(reviewer=object())

    message = server_reply(AgentState(), text, marker="test", reviewer=reviewer, fallback="FALLBACK")

    assert message.content == text
    assert "shortly" not in text and "later" in text
