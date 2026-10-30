"""Gemini adapters for factual Q&A: classifier, answer writer, grounding reviewer.

Each adapter owns its own model client; none is built at import time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.infrastructure.llm.gemini import build_gemini_chat_model
from app.modules.factual_qa.models import GroundingReview, RequestClassification
from app.modules.factual_qa.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CLASSIFIER_SYSTEM_PROMPT,
    GROUNDING_SYSTEM_PROMPT,
)
from app.shared.text import content_text


def _messages(system: str, payload: Mapping[str, Any]) -> list[Any]:
    return [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]


class GeminiRequestClassifier:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._llm = build_gemini_chat_model(
            api_key=api_key,
            model=model,
            temperature=0.0,
            # Rules take over on failure, so don't wait on retries.
            max_retries=1,
            thinking_budget=0,
        ).with_structured_output(RequestClassification)

    def classify(self, message: str, trip: Mapping[str, Any]) -> RequestClassification:
        raw = self._llm.invoke(
            _messages(CLASSIFIER_SYSTEM_PROMPT, {"message": message, "trip_context": dict(trip)})
        )
        return RequestClassification.model_validate(raw)


class GeminiFactualAnswerWriter:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._llm = build_gemini_chat_model(
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_retries=2,
            thinking_budget=0,
        )

    def write(self, payload: Mapping[str, Any]) -> str:
        response = self._llm.invoke(_messages(ANSWER_SYSTEM_PROMPT, payload))
        return content_text(getattr(response, "content", "")).strip()


class GeminiGroundingReviewer:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._llm = build_gemini_chat_model(
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_retries=1,
            thinking_budget=0,
        ).with_structured_output(GroundingReview)

    def review(self, payload: Mapping[str, Any]) -> GroundingReview:
        raw = self._llm.invoke(_messages(GROUNDING_SYSTEM_PROMPT, payload))

        if raw is None:
            raise RuntimeError("grounding reviewer returned no decision")

        return GroundingReview.model_validate(raw)


__all__ = [
    "GeminiFactualAnswerWriter",
    "GeminiGroundingReviewer",
    "GeminiRequestClassifier",
]
