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
    memory = LongTermMemory(
        home_country="Malaysia",
        dietary_restrictions=["User is vegan."],
    )
    service = FakeMemoryService()

    result = run_memory_extraction_task(
        "user-1",
        "I'm vegan and live in Malaysia",
        chain=FakeChain([memory]),
        memory_service=service,
    )

    assert result == {"memory_updated": True}
    assert service.relational == [
        {"home_country": "Malaysia", "dietary_restrictions": ["User is vegan."]}
    ]
    assert service.vector == [{"dietary": ["User is vegan."]}]


def _run(memory, service):
    return run_memory_extraction_task(
        "user-1",
        "message",
        chain=FakeChain([memory]),
        memory_service=service,
    )


@pytest.mark.parametrize("country", ["Malaysia", "MY", "MYS", "UK", "  Japan  "])
def test_recognised_home_country_is_saved(country):
    service = FakeMemoryService()

    result = _run(LongTermMemory(home_country=country, home_state="Selangor"), service)

    assert result == {"memory_updated": True}
    assert service.relational == [
        {"home_country": country.strip(), "home_state": "Selangor"}
    ]


@pytest.mark.parametrize("country", ["Narnia", "Mars", "Malasia", "   "])
def test_unknown_home_country_is_not_saved(country):
    service = FakeMemoryService()

    result = _run(LongTermMemory(home_country=country), service)

    assert result == {"memory_updated": False}
    assert service.relational == []
    assert service.vector == []


def test_unknown_home_country_drops_its_state_but_keeps_other_traits(caplog):
    service = FakeMemoryService()
    memory = LongTermMemory(
        home_country="Narnia",
        home_state="Lantern Waste",
        travel_pacing="relaxed",
    )

    with caplog.at_level("WARNING"):
        result = _run(memory, service)

    assert result == {"memory_updated": True}
    assert service.relational == [{"travel_pacing": "relaxed"}]
    assert service.vector == [{"travel_pacing": ["relaxed"]}]
    assert "memory.extractor.home_country_rejected" in caplog.text
    assert "Narnia" not in caplog.text


def test_state_without_a_country_is_still_saved():
    service = FakeMemoryService()

    _run(LongTermMemory(home_state="  Selangor "), service)

    assert service.relational == [{"home_state": "Selangor"}]
