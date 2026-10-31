"""Each task's backend chain and limiters, from the composition root."""

import pytest

from app.bootstrap import container
from app.infrastructure.config.settings import Settings
from app.infrastructure.llm.rate_limit import SlidingWindowRateLimiter


CHAT, UTILITY, ROUTER_MODEL = "chat-model", "utility-model", "google/gemma-test:free"

CLIENT_GETTERS = (
    container.get_output_reviewer,
    container.get_factual_request_classifier,
    container.get_factual_answer_writer,
    container.get_grounding_reviewer,
    container.get_activity_idea_generator,
    container.get_destination_city_picker,
)
GETTERS = (
    *CLIENT_GETTERS,
    container.get_chat_model,
    container.get_gemini_rate_limiter,
    container.get_openrouter_rate_limiter,
    container.get_backend_health,
)


@pytest.fixture
def configure(monkeypatch):
    """Point the container at in-memory settings and recording clients."""

    for name in (
        "GeminiOutputReviewer",
        "GeminiRequestClassifier",
        "GeminiFactualAnswerWriter",
        "GeminiGroundingReviewer",
        "GeminiActivityIdeaGenerator",
        "GeminiPrimaryCityPicker",
    ):
        monkeypatch.setattr(container, name, lambda **kwargs: kwargs["backends"])

    monkeypatch.setattr(container, "build_chat_model", lambda backends, **kwargs: backends)

    def apply(**values):
        settings = Settings.model_construct(
            GEMINI_API_KEY="main-key",
            GEMINI_CHAT_MODEL=CHAT,
            GEMINI_UTILITY_MODEL=UTILITY,
            OPENROUTER_MODEL=ROUTER_MODEL,
            **values,
        )
        monkeypatch.setattr(container, "_settings", lambda: settings)

        for getter in GETTERS:
            getter.cache_clear()

    yield apply

    for getter in GETTERS:
        getter.cache_clear()


def chain(backends):
    return [(backend.provider, backend.api_key, backend.model) for backend in backends]


def test_every_task_tries_its_key_then_the_second_key_then_openrouter(configure):
    configure(GEMINI_API_KEY_2="backup-key", OPENROUTER_API_KEY="router-key")

    assert chain(container.get_chat_model()) == [
        ("gemini", "main-key", CHAT),
        ("gemini", "backup-key", CHAT),
        ("openrouter", "router-key", ROUTER_MODEL),
    ]

    for getter in CLIENT_GETTERS:
        providers = [(provider, key) for provider, key, _ in chain(getter())]
        assert providers == [("gemini", "main-key"), ("gemini", "backup-key"), ("openrouter", "router-key")]


def test_a_task_with_its_own_key_still_gets_the_backups(configure):
    configure(GEMINI_DESTINATION_API="destination-key", GEMINI_API_KEY_2="backup-key")

    assert chain(container.get_destination_city_picker()) == [
        ("gemini", "destination-key", UTILITY),
        ("gemini", "backup-key", UTILITY),
    ]


def test_without_backups_each_task_has_only_its_own_key(configure):
    configure()

    assert chain(container.get_chat_model()) == [("gemini", "main-key", CHAT)]


def test_a_backup_key_equal_to_the_main_key_is_not_tried_twice(configure):
    configure(GEMINI_API_KEY_2="main-key")

    assert chain(container.get_output_reviewer()) == [("gemini", "main-key", CHAT)]


def test_limiters_are_shared_per_quota(configure):
    configure(GEMINI_API_KEY_2="backup-key", OPENROUTER_API_KEY="router-key")

    reviewer = container.get_output_reviewer()
    writer = container.get_factual_answer_writer()
    classifier = container.get_factual_request_classifier()

    main_chat = reviewer[0].rate_limiter
    assert isinstance(main_chat, SlidingWindowRateLimiter) and main_chat.max_requests == 15
    # Same key and model share Google's quota, whatever the task.
    assert writer[0].rate_limiter is main_chat
    # Another model or another key has its own quota.
    assert classifier[0].rate_limiter is not main_chat
    assert reviewer[1].rate_limiter is not main_chat
    # OpenRouter's free limit is per account, across models and tasks.
    assert reviewer[2].rate_limiter is classifier[2].rate_limiter
    assert reviewer[2].rate_limiter.max_requests == 20


def test_zero_turns_pacing_off(configure):
    configure(GEMINI_REQUESTS_PER_MINUTE=0, OPENROUTER_REQUESTS_PER_MINUTE=0, OPENROUTER_API_KEY="router-key")

    assert [backend.rate_limiter for backend in container.get_chat_model()] == [None, None]


def test_health_is_shared_per_key_and_model_across_tasks(configure):
    configure(GEMINI_API_KEY_2="backup-key", OPENROUTER_API_KEY="router-key")

    reviewer = container.get_output_reviewer()
    writer = container.get_factual_answer_writer()
    classifier = container.get_factual_request_classifier()

    # When the main key's quota for a model runs out, every task skips it.
    assert writer[0].health is reviewer[0].health
    assert classifier[0].health is not reviewer[0].health
    assert reviewer[1].health is not reviewer[0].health
    assert reviewer[2].health is classifier[2].health
    assert (reviewer[2].health.provider, reviewer[2].health.model) == ("openrouter", ROUTER_MODEL)
