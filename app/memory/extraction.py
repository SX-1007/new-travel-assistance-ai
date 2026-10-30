"""
Memory Extraction Module — Mem0 Tiered Extraction Engine

Extracts long-term user traits from conversation messages and routes them
to appropriate storage tiers (relational profile + vector database).

Design:
  • The chat model is injected (``build_extraction_chain``); this module
    reads no settings and creates no clients at import time
  • Parallel relational + vector writes via a short-lived thread pool
  • Exponential backoff with jitter for LLM retries (max 3 calls/message)
  • Trivial-message pre-filtering to skip unnecessary LLM calls
  • Module-level prompt template (compiled once at import)
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

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
# Total extraction attempts per message. The injected chat model should be
# built with max_retries=1 so attempts are not multiplied by client retries.
MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_WAIT = 1.5
LLM_RETRY_MAX_WAIT = 10.0
_MAX_MESSAGE_LENGTH = 8_192  # Guard against oversized inputs

# Pre-compiled regex for trivial-message detection (O(1) match)
_TRIVIAL_RE = re.compile(
    r"^(hi|hello|hey|ok|okay|thanks|thank\s+you|yes|no|sure|cool|nice|great|bye|gotcha|alright)\s*[!.?]*$",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Prompt Template (compiled once at module load)
# ──────────────────────────────────────────────
_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [  # FIX #2: from_messages (plural)
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


# ──────────────────────────────────────────────
# Chain Construction (LLM injected by the composition root)
# ──────────────────────────────────────────────
def build_extraction_chain(llm: BaseChatModel) -> Runnable:
    """
    Build the extraction chain around an injected chat model.

    The model is created by the composition root, so this module needs no
    credentials and can be tested with a fake chat model.
    """

    # Force the memory schema tool explicitly.
    #
    # The project's older langchain-google-genai integration does not
    # automatically recognise Gemini 3.x model names when deciding whether
    # structured output can force a tool call. Without an explicit tool_choice,
    # Gemini may return a normal text response and the Pydantic parser returns
    # None, causing valid preferences to be treated as "no traits".
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


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def _is_trivial(message: str) -> bool:
    """
    Check if a message is trivial (greeting / acknowledgement).
    These messages are skipped to avoid unnecessary LLM calls.
    """
    stripped = message.strip().lower()
    return len(stripped) < 4 or bool(_TRIVIAL_RE.match(stripped))


def _calculate_backoff(attempt: int) -> float:
    """
    Calculate exponential backoff with jitter.
    Jitter prevents thundering-herd when the LLM service recovers.
    """
    base = min(LLM_RETRY_BASE_WAIT * (2 ** (attempt - 1)), LLM_RETRY_MAX_WAIT)
    jitter = random.uniform(0, base * 0.1)  # Up to 10% jitter
    return base + jitter


def _has_long_term_traits(
    memory: LongTermMemory,
) -> bool:
    """Return True only when at least one persistent trait was extracted."""

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
    """Update relational long-term memory."""

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
    """Update semantic long-term memory."""

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
    

# ──────────────────────────────────────────────
# Core Extraction Logic
# ──────────────────────────────────────────────
def analyse_and_extract_traits(
    user_message: str,
    *,
    chain: Runnable,
) -> Optional[LongTermMemory]:
    """
    Send the user message to the LLM and extract long-term memory traits.

    Implements exponential backoff with jitter for transient LLM failures.
    Pre-filters trivial messages to avoid unnecessary LLM calls.

    Args:
        user_message: The user's latest chat message.
        chain: Extraction chain from ``build_extraction_chain``.

    Returns:
        LongTermMemory instance if traits were found, None otherwise.
    """
    if not user_message or not user_message.strip():
        return None

    if _is_trivial(user_message):
        log_memory_event(logger, "memory.extractor.trivial")
        return None

    # Guard against excessively long inputs (Google flash handles 1M tokens,
    # but truncating saves bandwidth and reduces latency)
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


# ──────────────────────────────────────────────
# Graph Node — Parallel DB Routing
# ──────────────────────────────────────────────
def run_memory_extraction_task(
    user_id: str,
    user_message: str,
    *,
    chain: Runnable,
    memory_service: MemoryService,
) -> Dict[str, Any]:
    """
    LangGraph-compatible node function.

    Extracts long-term traits from the user message and routes them to
    the appropriate storage tier **in parallel**:
      • Relational DB (home_country, home_state, travel_pacing)
      • Vector DB (dietary, interests, accommodation)

    Relational and vector writes run concurrently in a short-lived
    two-thread pool, reducing total latency from (DB1 + DB2) to
    max(DB1, DB2) without leaving a global executor to shut down.

    Args:
        user_id: The authenticated user's ID.
        user_message: The user's latest chat message.
        chain: Extraction chain from ``build_extraction_chain``.
        memory_service: Long-term memory use cases.

    Returns:
        Dict with key ``"memory_updated"`` (bool) indicating whether any
        persistence operation succeeded.
    """
    log_memory_event(logger, "memory.extractor.started")

    if not user_id:
        return {"memory_updated": False}

    # ── Step 1: Extract traits via LLM ──────────────────────
    extracted_data = analyse_and_extract_traits(user_message, chain=chain)

    if not extracted_data:
        return {"memory_updated": False}

    # ── Step 2: Prepare DB updates ──────────────────────────
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


    # Keep a semantic/vector copy as the second memory tier.
    vector_categories: Dict[str, Optional[List[str]]] = {
        "dietary": extracted_data.dietary_restrictions,
        "interest": extracted_data.interests,
        "accommodation": extracted_data.accommodation_preferences,

        # Keep a Mem0 fallback for pacing too.
        # This prevents the preference from disappearing if an older
        # user_profiles schema does not contain travel_pacing.
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

    # ── Step 3: Parallel DB Writes ──────────────────────────
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

    # Collect results as they complete (non-blocking on the slower one)
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
