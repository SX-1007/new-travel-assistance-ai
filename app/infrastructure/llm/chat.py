"""Chat models with ordered backups.

Each task gets a chain of backends: its own Gemini key first, then the
backups (the second Gemini key, then a free OpenRouter model). A failed call,
including one that returns malformed structured output, is retried once on
the next backend, so one account's outage or exhausted quota does not fail
the user's turn. A backend whose quota is used up, or whose key is rejected,
is skipped until its cooldown ends (``health.py``), so the backup answers at
once instead of after another doomed request.

``bind_tools`` and ``with_structured_output`` on the returned model apply to
every backend, so callers use it like a single chat model. Nothing here reads
application settings or creates clients at import time.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import Field

from app.infrastructure.llm.health import BackendHealth


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Free OpenRouter models answer slowly when busy; a hung backup must not
# hold the user's turn.
OPENROUTER_TIMEOUT_SECONDS = 60.0

Provider = Literal["gemini", "openrouter"]


@dataclass(frozen=True, slots=True)
class ChatBackend:
    """One provider account and model a task may call."""

    provider: Provider
    api_key: str
    model: str
    rate_limiter: BaseRateLimiter | None = None
    # Shared by every client of this provider, key and model.
    health: BackendHealth | None = None

    def __post_init__(self) -> None:
        if self.provider not in ("gemini", "openrouter"):
            raise ValueError(f"unknown chat provider: {self.provider!r}")

        if not (self.api_key or "").strip():
            raise ValueError("api_key must not be empty")

        if not (self.model or "").strip():
            raise ValueError("model must not be empty")


def _guarded(health: BackendHealth | None, call: Callable[[], LLMResult]) -> LLMResult:
    if health is None:
        return call()

    token = health.ensure_available()

    try:
        result = call()
    except Exception as error:
        health.record_failure(error)
        raise

    health.record_success(token)
    return result


async def _aguarded(health: BackendHealth | None, call: Callable[[], Awaitable[LLMResult]]) -> LLMResult:
    if health is None:
        return await call()

    token = health.ensure_available()

    try:
        result = await call()
    except Exception as error:
        health.record_failure(error)
        raise

    health.record_success(token)
    return result


# ``generate``/``agenerate`` are what invoke, ainvoke and batch call, and they
# run before the rate limiter: a backend that is cooling down is refused
# without taking a pacing slot or making a request.


class ChatGemini(ChatGoogleGenerativeAI):
    """``ChatGoogleGenerativeAI`` that skips the key while it is cooling down."""

    backend_health: BackendHealth | None = Field(default=None, exclude=True)

    def generate(self, messages: Any, *args: Any, **kwargs: Any) -> LLMResult:
        return _guarded(self.backend_health, lambda: super(ChatGemini, self).generate(messages, *args, **kwargs))

    async def agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> LLMResult:
        return await _aguarded(
            self.backend_health, lambda: super(ChatGemini, self).agenerate(messages, *args, **kwargs)
        )


class ChatOpenRouter(ChatOpenAI):
    """``ChatOpenAI`` against OpenRouter's OpenAI-compatible API.

    Structured output goes through a forced tool call by default: free
    OpenRouter models support tools but not always strict JSON schemas, which
    is ``ChatOpenAI``'s default method. Like ``ChatGemini``, it skips the
    account while it is cooling down.
    """

    backend_health: BackendHealth | None = Field(default=None, exclude=True)

    def generate(self, messages: Any, *args: Any, **kwargs: Any) -> LLMResult:
        return _guarded(self.backend_health, lambda: super(ChatOpenRouter, self).generate(messages, *args, **kwargs))

    async def agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> LLMResult:
        return await _aguarded(
            self.backend_health, lambda: super(ChatOpenRouter, self).agenerate(messages, *args, **kwargs)
        )

    def with_structured_output(
        self,
        schema: Any = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        **kwargs: Any,
    ) -> Runnable:
        return super().with_structured_output(schema, method=method, **kwargs)


def build_chat_model(
    backends: Sequence[ChatBackend],
    *,
    temperature: float = 0.0,
    max_retries: int = 2,
    **gemini_options: Any,
) -> BaseChatModel | Runnable:
    """The first backend, falling back to the others in order.

    ``max_retries`` is the total number of attempts on the LAST backend.
    Earlier backends get one attempt each, so a failure moves on at once
    instead of waiting through retries on an account that is down.
    ``gemini_options`` (for example ``thinking_budget``) apply only to Gemini
    backends.
    """

    if not backends:
        raise ValueError("at least one chat backend is required")

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    last = len(backends) - 1
    models = [
        _build_one(
            backend,
            temperature=temperature,
            attempts=max_retries if index == last else 1,
            gemini_options=gemini_options,
        )
        for index, backend in enumerate(backends)
    ]

    if len(models) == 1:
        return models[0]

    return models[0].with_fallbacks(models[1:])


def _build_one(
    backend: ChatBackend,
    *,
    temperature: float,
    attempts: int,
    gemini_options: dict[str, Any],
) -> BaseChatModel:
    if backend.provider == "gemini":
        return ChatGemini(
            model=backend.model.strip(),
            temperature=temperature,
            # The Gemini client counts attempts, not retries.
            max_retries=attempts,
            google_api_key=backend.api_key.strip(),
            rate_limiter=backend.rate_limiter,
            backend_health=backend.health,
            **gemini_options,
        )

    return ChatOpenRouter(
        model=backend.model.strip(),
        temperature=temperature,
        # The OpenAI client counts retries after the first attempt.
        max_retries=attempts - 1,
        api_key=backend.api_key.strip(),
        base_url=OPENROUTER_BASE_URL,
        timeout=OPENROUTER_TIMEOUT_SECONDS,
        rate_limiter=backend.rate_limiter,
        backend_health=backend.health,
    )


__all__ = [
    "OPENROUTER_BASE_URL",
    "ChatBackend",
    "ChatGemini",
    "ChatOpenRouter",
    "Provider",
    "build_chat_model",
]
