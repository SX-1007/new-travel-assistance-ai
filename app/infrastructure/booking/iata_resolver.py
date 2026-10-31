"""Gemini-backed IATA resolution for flight search."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from threading import RLock
from types import MappingProxyType
from typing import Final

import pycountry
from cachetools import TTLCache
from langchain_core.prompts import (
    PromptTemplate,
)
from pydantic import (
    BaseModel,
    Field,
)

from app.infrastructure.llm.chat import ChatBackend, build_chat_model
from app.modules.booking.flights import (
    normalise_iata_code,
)
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_CACHE_SIZE: Final = 1024
_CACHE_TTL_SECONDS: Final = 86_400


# Common destinations, answered without a model call.
_TOP_DESTINATIONS: Final[
    Mapping[str, str]
] = MappingProxyType(
    {
        "malaysia": "KUL",
        "singapore": "SIN",
        "thailand": "BKK",
        "japan": "HND",
        "south korea": "ICN",
        "indonesia": "CGK",
        "australia": "SYD",
        "united kingdom": "LHR",
        "usa": "JFK",
        "united states": "JFK",
    }
)


_IATA_PROMPT = (
    PromptTemplate.from_template(
        "You are an aviation system. "
        "The destination country is "
        "'{country}'.\n"
        "Return the 3-letter IATA code "
        "of its primary international "
        "gateway.\n"
        "Return ONLY the IATA code. "
        "No explanation and no quotes."
    )
)


class _IataCodeResponse(
    BaseModel
):
    iata_code: str = Field(
        description=(
            "Three-letter IATA code "
            "of the primary "
            "international airport"
        )
    )


class GeminiIataResolver:
    """Resolve validated countries to primary-airport IATA codes."""

    def __init__(
        self,
        *,
        backends: Sequence[ChatBackend],
    ) -> None:
        # ChatBackend validates the key and model.
        llm = build_chat_model(
            backends,
            temperature=0,
            max_retries=3,
        )

        self._chain = (
            _IATA_PROMPT
            | llm.with_structured_output(
                _IataCodeResponse
            )
        )

        self._cache: TTLCache[
            str,
            str | None,
        ] = TTLCache(
            maxsize=_CACHE_SIZE,
            ttl=_CACHE_TTL_SECONDS,
        )

        self._lock = RLock()

    def resolve_iata_code(
        self,
        country_name: str,
    ) -> str | None:

        clean_country = " ".join(
            (country_name or "").split()
        )

        if not clean_country:
            return None

        cache_key = (
            clean_country.casefold()
        )

        with self._lock:
            if cache_key in self._cache:
                return self._cache[
                    cache_key
                ]

        static_code = (
            _TOP_DESTINATIONS.get(
                cache_key
            )
        )

        if static_code is not None:
            with self._lock:
                self._cache[
                    cache_key
                ] = static_code

            return static_code

        try:
            country = (
                pycountry
                .countries
                .search_fuzzy(
                    clean_country
                )[0]
            )

        except (
            LookupError,
            IndexError,
        ):
            # Not a country at all: remember the miss.
            with self._lock:
                self._cache[
                    cache_key
                ] = None

            return None

        try:
            result = (
                self._chain.invoke(
                    {
                        "country":
                            country.name
                    }
                )
            )

        except Exception:
            safe_log(
                logger,
                logging.WARNING,
                "provider.iata.failed",
            )

            # A model failure may be temporary: don't cache it.
            return None

        # The chain returns None when the model answers without the
        # structured-output tool; treat that like an invalid code.
        iata_code = (
            normalise_iata_code(
                getattr(
                    result,
                    "iata_code",
                    None,
                )
            )
        )

        # Not cached: a bad answer may be one-off.
        if iata_code is None:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.iata."
                    "invalid_response"
                ),
            )

            return None

        with self._lock:
            self._cache[
                cache_key
            ] = iata_code

        return iata_code


__all__ = [
    "GeminiIataResolver",
]