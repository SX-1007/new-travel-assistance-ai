"""Memory extraction with injected fakes. No LLM, database or settings."""

import pytest

from app.memory import extraction
from app.memory.extraction import (
    LongTermMemory,
    analyse_and_extract_traits,
    run_memory_extraction_task,
)


class FakeChain:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def invoke(self, _inputs):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeMemoryService:
    def __init__(self):
        self.relational = []
        self.vector = []

    def update_profile(self, user_id, updates):
        self.relational.append(updates)
        return True

    def insert_preferences(self, user_id, categories):
        self.vector.append(categories)
        return True


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(extraction.time, "sleep", lambda _seconds: None)


def test_failures_stop_after_three_llm_calls():
    chain = FakeChain([RuntimeError("boom")] * 5)

    assert analyse_and_extract_traits("I am vegan and love museums", chain=chain) is None
    assert chain.calls == 3


def test_trivial_message_skips_llm():
    chain = FakeChain([])

    assert analyse_and_extract_traits("thanks!", chain=chain) is None
    assert chain.calls == 0


def test_traits_are_routed_to_both_stores():
    memory = LongTermMemory(dietary_restrictions=["User is vegan."])
    service = FakeMemoryService()

    result = run_memory_extraction_task(
        "user-1",
        "I'm vegan and live in Malaysia",
        chain=FakeChain([memory]),
        memory_service=service,
    )

    assert result == {"memory_updated": True}
    assert service.relational == [{"dietary_restrictions": ["User is vegan."]}]
    assert service.vector == [{"dietary": ["User is vegan."]}]


def _run(memory, service):
    return run_memory_extraction_task(
        "user-1",
        "message",
        chain=FakeChain([memory]),
        memory_service=service,
    )


def test_the_extractor_is_never_asked_for_a_home_location():
    # Every new trip flies from the profile's home: only the profile screen
    # may change it, never a remark in chat.
    assert not {"home_country", "home_state"} & set(LongTermMemory.model_fields)
    assert "home_country" not in extraction._EXTRACTION_PROMPT.messages[0].prompt.template


def test_a_home_location_in_the_model_output_is_never_saved():
    service = FakeMemoryService()
    memory = LongTermMemory.model_validate(
        {"home_country": "Singapore", "home_state": "Central", "travel_pacing": "relaxed"}
    )

    result = _run(memory, service)

    assert result == {"memory_updated": True}
    assert service.relational == [{"travel_pacing": "relaxed"}]


def test_a_message_with_only_a_home_location_saves_nothing():
    service = FakeMemoryService()

    result = _run(LongTermMemory.model_validate({"home_country": "Singapore"}), service)

    assert result == {"memory_updated": False}
    assert service.relational == [] and service.vector == []
