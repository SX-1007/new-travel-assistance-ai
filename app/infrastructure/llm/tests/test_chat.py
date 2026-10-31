"""Chat models with ordered backups: construction and failover, offline."""

import asyncio

import pytest
from google.api_core import exceptions as google_errors
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app.infrastructure.llm.chat import (
    OPENROUTER_BASE_URL,
    ChatBackend,
    ChatGemini,
    ChatOpenRouter,
    build_chat_model,
)
from app.infrastructure.llm.health import BackendCoolingDown, BackendHealth
from app.infrastructure.llm.rate_limit import SlidingWindowRateLimiter


MAIN = ChatBackend("gemini", "main-key", "gemini-test")
BACKUP = ChatBackend("gemini", "backup-key", "gemini-test")
ROUTER = ChatBackend("openrouter", "router-key", "google/gemma-test:free")


@tool
def search_places(query: str) -> str:
    """Search places."""
    return "[]"


class Verdict(BaseModel):
    approved: bool


def answer(text):
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


@pytest.fixture
def providers(monkeypatch):
    """Scripted provider outcomes, keyed by API key; records every call."""

    outcomes, calls = {}, []

    def generate(self, messages, *args, **kwargs):
        key = self.google_api_key if isinstance(self, ChatGoogleGenerativeAI) else self.openai_api_key
        key = key.get_secret_value()
        calls.append(key)
        outcome = outcomes[key]

        if isinstance(outcome, Exception):
            raise outcome

        return answer(outcome)

    async def agenerate(self, messages, *args, **kwargs):
        return generate(self, messages)

    for cls in (ChatGoogleGenerativeAI, ChatOpenRouter):
        monkeypatch.setattr(cls, "_generate", generate)
        monkeypatch.setattr(cls, "_agenerate", agenerate)

    return outcomes, calls


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


class CountingLimiter(SlidingWindowRateLimiter):
    def __init__(self):
        super().__init__(max_requests=100)
        self.acquired = 0

    def acquire(self, *, blocking=True):
        self.acquired += 1
        return super().acquire(blocking=blocking)


DAILY_QUOTA = google_errors.ResourceExhausted("429 quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
MINUTE_QUOTA = google_errors.ResourceExhausted("429 quota exceeded. Please retry in 30s.")


def guarded_chain(clock):
    """Main and backup Gemini keys, each with its own health record."""

    main_health = BackendHealth(provider="gemini", model="gemini-test", clock=clock)
    backup_health = BackendHealth(provider="gemini", model="gemini-test", clock=clock)
    limiter = CountingLimiter()
    chain = build_chat_model(
        [
            ChatBackend("gemini", "main-key", "gemini-test", limiter, main_health),
            ChatBackend("gemini", "backup-key", "gemini-test", None, backup_health),
        ]
    )
    return chain, limiter, main_health


def test_one_backend_is_a_plain_model_with_every_attempt():
    pacer = SlidingWindowRateLimiter(max_requests=15)
    model = build_chat_model([ChatBackend("gemini", "main-key", "gemini-test", pacer)], max_retries=3)

    assert isinstance(model, ChatGemini)
    assert model.max_retries == 3 and model.rate_limiter is pacer


def test_backups_follow_in_order_and_only_the_last_one_retries():
    chain = build_chat_model([MAIN, BACKUP, ROUTER], max_retries=3, thinking_budget=0)

    assert isinstance(chain, RunnableWithFallbacks)
    main, backup, router = chain.runnable, *chain.fallbacks
    # Earlier backends get one attempt, so a failing account is left at once.
    assert (main.max_retries, backup.max_retries) == (1, 1)
    assert isinstance(router, ChatOpenRouter)
    # 3 attempts on OpenRouter = 1 call + 2 retries.
    assert router.max_retries == 2
    assert router.openai_api_base == OPENROUTER_BASE_URL
    # Gemini-only options never reach the OpenAI-compatible client.
    assert main.thinking_budget == 0 and backup.thinking_budget == 0


def test_a_healthy_main_key_is_the_only_one_called(providers):
    outcomes, calls = providers
    outcomes.update({"main-key": "from main", "backup-key": "from backup", "router-key": "from router"})

    assert build_chat_model([MAIN, BACKUP, ROUTER]).invoke("hi").content == "from main"
    assert calls == ["main-key"]


def test_a_failing_main_key_falls_back_to_the_second_key(providers):
    outcomes, calls = providers
    outcomes.update({"main-key": RuntimeError("429 quota"), "backup-key": "from backup"})

    assert build_chat_model([MAIN, BACKUP, ROUTER]).invoke("hi").content == "from backup"
    assert calls == ["main-key", "backup-key"]


def test_openrouter_answers_when_both_gemini_keys_fail(providers):
    outcomes, calls = providers
    outcomes.update({"main-key": RuntimeError("down"), "backup-key": RuntimeError("down"), "router-key": "from router"})

    assert build_chat_model([MAIN, BACKUP, ROUTER]).invoke("hi").content == "from router"
    assert calls == ["main-key", "backup-key", "router-key"]


def test_when_every_backend_fails_the_first_error_is_raised(providers):
    outcomes, _ = providers
    outcomes.update({"main-key": RuntimeError("main down"), "backup-key": RuntimeError("backup down")})

    with pytest.raises(RuntimeError, match="main down"):
        build_chat_model([MAIN, BACKUP]).invoke("hi")


def test_tools_and_structured_output_apply_to_every_backend():
    chain = build_chat_model([MAIN, ROUTER])

    with_tools = chain.bind_tools([search_places])
    assert isinstance(with_tools, RunnableWithFallbacks)
    assert with_tools.fallbacks[0].kwargs["tools"][0]["function"]["name"] == "search_places"

    structured = chain.with_structured_output(Verdict)
    assert isinstance(structured, RunnableWithFallbacks) and len(structured.fallbacks) == 1


def test_openrouter_structured_output_uses_a_forced_tool_call():
    router = build_chat_model([ROUTER])
    bound = router.with_structured_output(Verdict).first

    # Free OpenRouter models support tools but not always strict JSON schemas.
    assert bound.kwargs["tools"][0]["function"]["name"] == "Verdict"
    assert "response_format" not in bound.kwargs


@pytest.mark.parametrize(
    "make",
    [
        lambda: build_chat_model([]),
        lambda: build_chat_model([MAIN], max_retries=0),
        lambda: ChatBackend("gemini", " ", "gemini-test"),
        lambda: ChatBackend("gemini", "key", ""),
        lambda: ChatBackend("anthropic", "key", "model"),
    ],
)
def test_invalid_configurations_are_refused(make):
    with pytest.raises(ValueError):
        make()


# ── Skipping a backend that can't answer ───────────────────────────


def test_a_used_up_main_key_is_skipped_without_a_request_or_a_pacing_slot(providers):
    outcomes, calls = providers
    outcomes.update({"main-key": DAILY_QUOTA, "backup-key": "from backup"})
    chain, limiter, _ = guarded_chain(Clock())

    assert chain.invoke("first").content == "from backup"
    assert (calls, limiter.acquired) == (["main-key", "backup-key"], 1)

    calls.clear()
    assert chain.invoke("second").content == "from backup"
    assert chain.invoke("third").content == "from backup"
    # The main key is not asked again, and pacing isn't spent on it.
    assert calls == ["backup-key", "backup-key"]
    assert limiter.acquired == 1


def test_the_main_key_is_tried_again_after_its_cooldown(providers):
    outcomes, calls = providers
    outcomes.update({"main-key": MINUTE_QUOTA, "backup-key": "from backup"})
    clock = Clock()
    chain, _, main_health = guarded_chain(clock)

    chain.invoke("first")
    clock.now += 31
    outcomes["main-key"] = "from main"
    calls.clear()

    assert chain.invoke("later").content == "from main"
    assert calls == ["main-key"]
    assert main_health.seconds_until_available() == 0


def test_a_transient_failure_does_not_skip_the_main_key_next_time(providers):
    outcomes, calls = providers
    outcomes.update({"main-key": TimeoutError("read timed out"), "backup-key": "from backup"})
    chain, _, _ = guarded_chain(Clock())

    chain.invoke("first")
    calls.clear()
    chain.invoke("second")

    assert calls == ["main-key", "backup-key"]


def test_the_async_path_skips_a_cooling_key_too(providers):
    outcomes, calls = providers
    outcomes.update({"main-key": DAILY_QUOTA, "backup-key": "from backup"})
    chain, _, _ = guarded_chain(Clock())

    async def run():
        first = await chain.ainvoke("first")
        calls.clear()
        second = await chain.bind_tools([search_places]).ainvoke("second")
        return first, second

    first, second = asyncio.run(run())

    assert first.content == second.content == "from backup"
    assert calls == ["backup-key"]


def test_a_cooling_key_with_no_backup_fails_fast_with_a_clear_error(providers):
    outcomes, calls = providers
    outcomes["main-key"] = DAILY_QUOTA
    health = BackendHealth(provider="gemini", model="gemini-test", clock=Clock())
    model = build_chat_model([ChatBackend("gemini", "main-key", "gemini-test", None, health)])

    with pytest.raises(google_errors.ResourceExhausted):
        model.invoke("first")

    with pytest.raises(BackendCoolingDown):
        model.invoke("second")

    assert calls == ["main-key"]
