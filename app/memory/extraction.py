"""Long-term memory extraction from one chat message.

A model reads the user's latest message and returns only lasting traits
(home, travel pacing, diet, interests, lodging style). They are saved in two
places at once: the profile row (``user_profiles``) and the embedded
preference store (``mem0_vector``, searched by meaning).

Greetings and one-word replies are skipped without a model call. A failed
call is retried with backoff, at most ``MAX_LLM_RETRIES`` calls per message.
The chat model is injected (``build_extraction_chain``); nothing here reads
settings.

Known gap (B4): a ``home_country`` learned from chat is saved without the
country check that onboarding applies.
"""

from __future__ import annotations

import re
import time
import random
import logging
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from app.memory.observability import log_memory_event
from app.memory.service import MemoryService

__all__ = [
    "LongTermMemory",
    "build_extraction_chain",
    "analyse_and_extract_traits",
    "run_memory_extraction_task",
]

logger = logging.getLogger(__name__)

# Total extraction attempts per message. The injected chat model should be
# built with max_retries=1 so attempts are not multiplied by client retries.
MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_WAIT = 1.5
LLM_RETRY_MAX_WAIT = 10.0
_MAX_MESSAGE_LENGTH = 8_192

# Greetings and acknowledgements: nothing to learn, so no model call.
_TRIVIAL_RE = re.compile(
    r"^(hi|hello|hey|ok|okay|thanks|thank\s+you|yes|no|sure|cool|nice|great|bye|gotcha|alright)\s*[!.?]*$",
    re.IGNORECASE,
)


class LongTermMemory(BaseModel):
    """Permanent user traits stored in the relational profile and vector DB."""

    home_country: Optional[str] = Field(
        default=None,
        description="The permanent home country of the user.",
    )
    home_state: Optional[str] = Field(
        default=None,
        description="The permanent home state or province of the user.",
    )
    travel_pacing: Optional[str] = Field(
        default=None,
        description="Permanent travel speed preference (e.g., 'relaxed', 'packed schedule', 'moderate').",
    )
    dietary_restrictions: Optional[List[str]] = Field(
        default=None,
        description="Permanent food allergies or diets (e.g., 'User is vegan', 'User has a peanut allergy').",
    )
    interests: Optional[List[str]] = Field(
        default=None,
        description="Lifelong broad interests (e.g., 'User loves history', 'User enjoys hiking').",
    )
    accommodation_preferences: Optional[List[str]] = Field(
        default=None,
        description="Permanent hotel/lodging styles (e.g., 'User prefers 5-star hotels').",
    )


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Mem0 Context Management Engine for an AI Travel Agent.
Your job is to read the user's latest message and extract ONLY permanent, long-term memory points.

RULES:
1. ONLY extract permanent traits. IGNORE short-term trip requests (e.g., "I want a pool for this trip").
2. Sort extracted traits into their specific categories. Do not mix them.
3. Format traits as clear, standalone factual sentences.

EXAMPLES:
User: "I'm allergic to peanuts, I love history, and I only stay in 5-star hotels."
→ dietary_restrictions: ["User has a peanut allergy."]
→ interests: ["User loves history."]
→ accommodation_preferences: ["User only stays in 5-star hotels."]

User: "I live in Selangor, Malaysia, and I like a relaxed pace when traveling."
→ home_country: "Malaysia", home_state: "Selangor", travel_pacing: "relaxed"

User: "Can you find a hotel with a gym for next week?"
→ All fields null.
""",
        ),
        ("human", "{message}"),
    ]
)


def build_extraction_chain(llm: BaseChatModel) -> Runnable:
    """The extraction chain around an injected chat model (a fake one in tests)."""

    # Force the LongTermMemory tool call. langchain-google-genai 2.1 doesn't
    # recognise Gemini 3 model names as able to force structured output, so
    # without an explicit tool_choice Gemini may answer in prose; the parser
    # then returns None and real preferences would be lost as "no traits".
    forced_llm = llm.bind_tools(
        [LongTermMemory],
        tool_choice=LongTermMemory.__name__,
    )

    parser = PydanticToolsParser(
        tools=[LongTermMemory],
        first_tool_only=True,
    )

    chain = _EXTRACTION_PROMPT | forced_llm | parser

    log_memory_event(
        logger,
        "memory.extractor_chain_ready",
    )

    return chain


def _is_trivial(message: str) -> bool:
    """Very short messages and greetings/acknowledgements."""
    stripped = message.strip().lower()
    return len(stripped) < 4 or bool(_TRIVIAL_RE.match(stripped))


def _calculate_backoff(attempt: int) -> float:
    """Exponential backoff plus up to 10% jitter, so retries from many requests don't land at once."""
    base = min(LLM_RETRY_BASE_WAIT * (2 ** (attempt - 1)), LLM_RETRY_MAX_WAIT)
    jitter = random.uniform(0, base * 0.1)
    return base + jitter


def _has_long_term_traits(
    memory: LongTermMemory,
) -> bool:
    """Whether at least one trait was extracted."""

    return any(
        (
            bool(memory.home_country),
            bool(memory.home_state),
            bool(memory.travel_pacing),
            bool(memory.dietary_restrictions),
            bool(memory.interests),
            bool(memory.accommodation_preferences),
        )
    )


def _update_relational_db(
    memory_service: MemoryService,
    user_id: str,
    updates: Dict[str, Any],
) -> bool:
    """Save to the profile row; False (logged) on failure."""

    try:
        return memory_service.update_profile(
            user_id,
            updates,
        )

    except Exception:
        log_memory_event(
            logger,
            "memory.extractor.relational_failed",
        )
        return False


def _update_vector_db(
    memory_service: MemoryService,
    user_id: str,
    categories: Dict[str, List[str]],
) -> bool:
    """Save to the preference store; False (logged) on failure."""

    try:
        return (
            memory_service
            .insert_preferences(
                user_id,
                categories,
            )
        )

    except Exception:
        log_memory_event(
            logger,
            "memory.extractor.vector_failed",
        )
        return False
    

def analyse_and_extract_traits(
    user_message: str,
    *,
    chain: Runnable,
) -> Optional[LongTermMemory]:
    """The traits in ``user_message``; None when there are none, the message
    is trivial, or every attempt failed (failures are retried with backoff).
    """
    if not user_message or not user_message.strip():
        return None

    if _is_trivial(user_message):
        log_memory_event(logger, "memory.extractor.trivial")
        return None

    # Bound the input: a long paste adds cost and latency, not traits.
    message = user_message.strip()[:_MAX_MESSAGE_LENGTH]

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            result = chain.invoke(
                {
                    "message": message,
                }
            )

            # Missing structured output is an extraction failure.
            # It does NOT mean the user has no long-term preferences.
            if not isinstance(result, LongTermMemory):
                raise RuntimeError(
                    "Memory extractor returned no structured "
                    "LongTermMemory output."
                )

            # A valid LongTermMemory object with every field empty really
            # means that this message contains no persistent user traits.
            if not _has_long_term_traits(result):
                log_memory_event(
                    logger,
                    "memory.extractor.no_traits",
                )
                return None

            log_memory_event(
                logger,
                "memory.extractor.completed",
            )

            return result

        except Exception:
            if attempt == MAX_LLM_RETRIES:
                log_memory_event(
                    logger,
                    "memory.extractor.failed",
                )
                return None

            wait_time = _calculate_backoff(attempt)

            log_memory_event(
                logger,
                "memory.extractor.retry",
            )

            time.sleep(wait_time)

    return None


def run_memory_extraction_task(
    user_id: str,
    user_message: str,
    *,
    chain: Runnable,
    memory_service: MemoryService,
) -> Dict[str, Any]:
    """Extract traits from one message and save them.

    Runs in the background (the graph's memory queue). Returns
    ``{"memory_updated": bool}``: true when either store was updated.
    """
    log_memory_event(logger, "memory.extractor.started")

    if not user_id:
        return {"memory_updated": False}

    extracted_data = analyse_and_extract_traits(user_message, chain=chain)

    if not extracted_data:
        return {"memory_updated": False}

    # Every trait goes to the profile row.
    relational_updates: Dict[str, Any] = {}

    if extracted_data.home_country:
        relational_updates["home_country"] = extracted_data.home_country

    if extracted_data.home_state:
        relational_updates["home_state"] = extracted_data.home_state

    if extracted_data.travel_pacing:
        relational_updates["travel_pacing"] = extracted_data.travel_pacing

    if extracted_data.dietary_restrictions:
        relational_updates["dietary_restrictions"] = (
            extracted_data.dietary_restrictions
        )

    if extracted_data.interests:
        relational_updates["interests"] = extracted_data.interests

    if extracted_data.accommodation_preferences:
        relational_updates["accommodation_preferences"] = (
            extracted_data.accommodation_preferences
        )


    # The list traits also go to the preference store, as sentences that can
    # be searched by meaning.
    vector_categories: Dict[str, Optional[List[str]]] = {
        "dietary": extracted_data.dietary_restrictions,
        "interest": extracted_data.interests,
        "accommodation": extracted_data.accommodation_preferences,

        # A second copy of pacing, besides the profile column.
        "travel_pacing": (
            [extracted_data.travel_pacing]
            if extracted_data.travel_pacing
            else None
        ),
    }

    valid_categories: Dict[str, List[str]] = {
        cat: prefs
        for cat, prefs in vector_categories.items()
        if prefs
    }

    if not relational_updates and not valid_categories:
        return {"memory_updated": False}

    # Both writes at once; either success counts as an update.
    futures: Dict[Future, str] = {}
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem0-db")

    if relational_updates:
        log_memory_event(
            logger,
            "memory.extractor.relational_queued",
        )

        future = executor.submit(
            _update_relational_db,
            memory_service,
            user_id,
            relational_updates,
        )

        futures[future] = "relational"

    if valid_categories:
        log_memory_event(
            logger,
            "memory.extractor.vector_queued",
        )

        future = executor.submit(
            _update_vector_db,
            memory_service,
            user_id,
            valid_categories,
        )

        futures[future] = "vector"

    memory_updated = False
    try:
        for future in as_completed(futures):
            try:
                if future.result():
                    memory_updated = True
                    log_memory_event(logger, "memory.extractor.persistence_succeeded")
                else:
                    log_memory_event(logger, "memory.extractor.persistence_failed")
            except Exception:
                log_memory_event(logger, "memory.extractor.persistence_raised")
    finally:
        executor.shutdown(wait=False)

    return {"memory_updated": memory_updated}
