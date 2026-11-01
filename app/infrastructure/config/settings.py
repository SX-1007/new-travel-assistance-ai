"""Application settings from the environment and ``.env``.

Validated on first use: a missing required value stops the app at startup
rather than failing a request later.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings — validates all required env vars on first access.

    Only the composition root (``app.bootstrap.container``) may read settings.
    Infrastructure classes receive the values they need through constructors,
    so any module can be imported and tested without a complete ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ── Gemini API keys ──────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(
        ..., min_length=1, description="Primary Gemini API key (required)"
    )
    # Backup key: every task falls back to it, on the same model, when its
    # own key fails (outage, exhausted quota, malformed answer).
    GEMINI_API_KEY_2: str = ""
    # Optional per-task keys; each falls back to GEMINI_API_KEY (see the
    # ``effective_*`` properties below).
    GEMINI_IATA_API: str = ""
    GEMINI_FLIGHTS_HOTELS: str = ""
    GEMINI_EXTRACTOR_API: str = ""
    GEMINI_EMBED_API: str = ""
    GEMINI_DESTINATION_API: str = ""
    GEMINI_ACTIVITIES_API: str = ""

    # ── Models ───────────────────────────────────────────────────────
    GEMINI_CHAT_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_UTILITY_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_EMBED_MODEL: str = "gemini-embedding-2"
    # Calls allowed per minute for each key and chat model, matching
    # Google's quota (15 on the free tier). Extra calls wait for a free slot
    # instead of failing with 429. 0 turns pacing off.
    GEMINI_REQUESTS_PER_MINUTE: int = Field(default=15, ge=0)

    # ── OpenRouter (last-resort backup) ─────────────────────────────
    # Tried after the Gemini keys. Free models are shared and often busy,
    # and allow 50 requests a day without purchased credits.
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "google/gemma-4-31b-it:free"
    # OpenRouter's free models allow 20 per minute per account.
    OPENROUTER_REQUESTS_PER_MINUTE: int = Field(default=20, ge=0)

    # ── PostgreSQL ───────────────────────────────────────────────────
    # Profiles, checkpoints, session ownership and pgvector tables. Use a
    # direct connection: the app pools its own connections and sets a
    # per-connection statement timeout, which a transaction pooler drops.
    DATABASE_URL: str = Field(
        ..., min_length=1, description="Direct PostgreSQL connection string"
    )

    # ── Firebase (sign-in) ───────────────────────────────────────────
    FIREBASE_CREDENTIAL_JSON: str = Field(
        ..., min_length=1, description="Path to JSON file or raw JSON string"
    )

    # ── Sign-in ──────────────────────────────────────────────────────
    # Users sign in with Firebase Authentication (same project as the
    # credential above). Development only: true accepts "Bearer <user id>"
    # with no check, so anyone can act as any user. Never set it on a
    # server others can reach.
    AUTH_INSECURE_DEV_TOKENS: bool = False

    # ── Cache (Upstash Redis, REST API) ─────────────────────────────
    # Flight, hotel and place results and budget assessments. Optional:
    # without them nothing is cached (more paid searches), and a budget
    # below the minimum can't be confirmed in a later request.
    UPSTASH_REDIS_REST_URL: str = ""
    UPSTASH_REDIS_REST_TOKEN: str = ""

    # ── External APIs ────────────────────────────────────────────────
    MAPBOX_TOKEN: str = Field(..., min_length=1)
    SERPAPI_FLIGHT: str = Field(..., min_length=1)
    SERPAPI_HOTEL: str = Field(..., min_length=1)
    SERPAPI_PLACE: str = Field(..., min_length=1)
    # Optional key for factual web search; falls back to SERPAPI_PLACE.
    SERPAPI_FACTUAL: str = ""
    # Optional. When set, factual web search tries Exa first and uses
    # SerpAPI only when Exa fails, finds nothing or is out of credit.
    EXA_API_KEY: str = ""
    FREECURRENCY_API: str = Field(..., min_length=1)

    # ── Effective keys: a task's own key, else the fallback ──────────
    @property
    def effective_embed_api_key(self) -> str:
        return self.GEMINI_EMBED_API or self.GEMINI_API_KEY

    @property
    def effective_flights_hotels_api_key(self) -> str:
        return self.GEMINI_FLIGHTS_HOTELS or self.GEMINI_API_KEY

    @property
    def effective_iata_api_key(
        self,
    ) -> str:
        """IATA key, else the flights/hotels key, else the primary key."""
        return (
            self.GEMINI_IATA_API
            or self.effective_flights_hotels_api_key
        )

    @property
    def effective_destination_api_key(self) -> str:
        return self.GEMINI_DESTINATION_API or self.GEMINI_API_KEY

    @property
    def effective_activities_api_key(self) -> str:
        return self.GEMINI_ACTIVITIES_API or self.GEMINI_API_KEY

    @property
    def effective_extractor_api_key(self) -> str:
        return self.GEMINI_EXTRACTOR_API or self.GEMINI_API_KEY

    @property
    def effective_factual_serpapi_key(self) -> str:
        return self.SERPAPI_FACTUAL or self.SERPAPI_PLACE


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings once, on first use rather than at import."""
    return Settings()
