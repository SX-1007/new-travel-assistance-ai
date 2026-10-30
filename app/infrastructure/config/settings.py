"""
Application configuration — loaded from environment / .env file.

Uses Pydantic BaseSettings for:
  • Type validation at startup (fail-fast)
  • Automatic .env loading
  • Computed fallback properties for optional Gemini sub-keys
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
    GEMINI_IATA_API: str = ""
    GEMINI_FLIGHTS_HOTELS: str = ""
    GEMINI_EXTRACTOR_API: str = ""
    GEMINI_EMBED_API: str = ""
    GEMINI_DESTINATION_API: str = ""
    GEMINI_ACTIVITIES_API: str = ""

    # ------------------- Model Configuration ------------------
    GEMINI_CHAT_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_UTILITY_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_EMBED_MODEL: str = "gemini-embedding-2"

    # ── PostgreSQL ───────────────────────────────────────────────────
    # Profiles, checkpoints, session ownership and pgvector tables. Use a
    # direct connection: the app pools its own connections and sets a
    # per-connection statement timeout, which a transaction pooler drops.
    DATABASE_URL: str = Field(
        ..., min_length=1, description="Direct PostgreSQL connection string"
    )

    # ── Firebase ─────────────────────────────────────────────────────
    FIREBASE_CREDENTIAL_JSON: str = Field(
        ..., min_length=1, description="Path to JSON file or raw JSON string"
    )

    # ── External APIs ────────────────────────────────────────────────
    MAPBOX_TOKEN: str = Field(..., min_length=1)
    SERPAPI_FLIGHT: str = Field(..., min_length=1)
    SERPAPI_HOTEL: str = Field(..., min_length=1)
    SERPAPI_PLACE: str = Field(..., min_length=1)
    # Optional dedicated key for factual web retrieval. Falls back to the
    # existing SerpAPI place key so current deployments keep working.
    SERPAPI_FACTUAL: str = ""
    FREECURRENCY_API: str = Field(..., min_length=1)

    # ── Computed helpers (fallback to primary Gemini key) ────────────
    @property
    def effective_embed_api_key(self) -> str:
        """Embedding API key — falls back to primary Gemini key."""
        return self.GEMINI_EMBED_API or self.GEMINI_API_KEY

    @property
    def effective_flights_hotels_api_key(self) -> str:
        return self.GEMINI_FLIGHTS_HOTELS or self.GEMINI_API_KEY

    @property
    def effective_iata_api_key(
        self,
    ) -> str:
        """
        Dedicated IATA resolver key with
        booking/primary fallback.
        """

        return (
            self.GEMINI_IATA_API
            or self.effective_flights_hotels_api_key
        )

    @property
    def effective_destination_api_key(self) -> str:
        """Primary-city selection key — falls back to primary Gemini key."""
        return self.GEMINI_DESTINATION_API or self.GEMINI_API_KEY

    @property
    def effective_activities_api_key(self) -> str:
        """Activity-idea generation key — falls back to primary Gemini key."""
        return self.GEMINI_ACTIVITIES_API or self.GEMINI_API_KEY

    @property
    def effective_extractor_api_key(self) -> str:
        return self.GEMINI_EXTRACTOR_API or self.GEMINI_API_KEY

    @property
    def effective_factual_serpapi_key(self) -> str:
        """Dedicated factual-search key, falling back to the existing place key."""
        return self.SERPAPI_FACTUAL or self.SERPAPI_PLACE


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings once, on first use rather than at import."""
    return Settings()
