"""Concrete country-to-currency and foreign-exchange provider infrastructure.

This module owns HTTP, retries, provider fallback, caching and country/currency
resolution. It intentionally has no dependency on AgentState, LangGraph or the
budget domain.
"""

from __future__ import annotations

import logging
import math
from threading import RLock
from typing import Final

import pycountry
import requests
from babel.numbers import get_territory_currencies
from cachetools import TTLCache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.modules.budget.exceptions import (
    CurrencyCodeUnavailableError,
    CurrencyRateUnavailableError,
)
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_COUNTRY_CACHE_SIZE: Final = 256
_COUNTRY_CACHE_TTL_SECONDS: Final = 86_400

_RATE_CACHE_SIZE: Final = 512
_RATE_CACHE_TTL_SECONDS: Final = 3_600

_REQUEST_TIMEOUT_SECONDS: Final = 5.0

_PRIMARY_RATE_URL: Final = (
    "https://api.freecurrencyapi.com/v1/latest"
)

_FALLBACK_RATE_URL: Final = (
    "https://open.er-api.com/v6/latest/{origin}"
)


_RETRY_STRATEGY = Retry(
    total=3,
    connect=3,
    read=3,
    status=3,
    backoff_factor=0.5,
    status_forcelist=(
        429,
        500,
        502,
        503,
        504,
    ),
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=True,
    raise_on_status=False,
)


def _build_http_session() -> requests.Session:
    session = requests.Session()

    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=_RETRY_STRATEGY,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session


def _normalise_currency_code(
    value: str,
    *,
    field_name: str,
) -> str:
    code = (
        value or ""
    ).strip().upper()

    if (
        len(code) != 3
        or not code.isascii()
        or not code.isalpha()
    ):
        raise ValueError(
            f"{field_name} must be a "
            "3-letter ISO-4217 code"
        )

    return code


def _valid_rate(
    value: object,
) -> float | None:
    try:
        rate = float(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if (
        not math.isfinite(rate)
        or rate <= 0
    ):
        return None

    return rate


class CurrencyProvider:
    """
    Thread-safe cached adapter over the configured
    currency providers.
    """

    def __init__(
        self,
        *,
        api_key: str,
        session: requests.Session | None = None,
    ) -> None:
        key = api_key.strip()

        if not key:
            raise ValueError(
                "api_key must not be empty"
            )

        self._api_key = key

        self._session = (
            session
            or _build_http_session()
        )

        self._owns_session = (
            session is None
        )

        self._country_cache: TTLCache[
            str,
            str,
        ] = TTLCache(
            maxsize=_COUNTRY_CACHE_SIZE,
            ttl=_COUNTRY_CACHE_TTL_SECONDS,
        )

        self._rate_cache: TTLCache[
            tuple[str, str],
            float,
        ] = TTLCache(
            maxsize=_RATE_CACHE_SIZE,
            ttl=_RATE_CACHE_TTL_SECONDS,
        )

        self._country_lock = RLock()
        self._rate_lock = RLock()

    def close(self) -> None:
        """Close the internally-owned HTTP session."""

        if self._owns_session:
            self._session.close()

    def get_currency_code(
        self,
        country: str,
    ) -> str:
        """
        Resolve a country name to its current
        primary ISO-4217 currency code.

        Resolution failures are explicit. Returning
        USD for an unknown country could silently
        corrupt downstream budget calculations.
        """

        normalised_country = " ".join(
            (country or "").split()
        )

        if not normalised_country:
            raise CurrencyCodeUnavailableError(
                "country must not be empty"
            )

        cache_key = (
            normalised_country.casefold()
        )

        with self._country_lock:
            cached = (
                self._country_cache.get(
                    cache_key
                )
            )

        if cached is not None:
            return cached

        try:
            country_data = (
                pycountry.countries.search_fuzzy(
                    normalised_country
                )[0]
            )

            currencies = (
                get_territory_currencies(
                    country_data.alpha_2,
                    tender=True,
                )
            )

            if not currencies:
                raise LookupError(
                    "no active tender currency found"
                )

            code = _normalise_currency_code(
                currencies[0],
                field_name=(
                    "resolved currency code"
                ),
            )

        except (
            LookupError,
            IndexError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.currency."
                    "country_resolution_failed"
                ),
            )

            raise CurrencyCodeUnavailableError(
                "Could not resolve a currency "
                f"for country "
                f"{normalised_country!r}."
            ) from exc

        with self._country_lock:
            self._country_cache[
                cache_key
            ] = code

        return code

    def get_currency_rate(
        self,
        origin_curr_code: str,
        des_curr_code: str,
    ) -> float:
        """
        Return a positive FX rate or raise
        CurrencyRateUnavailableError.
        """

        origin = _normalise_currency_code(
            origin_curr_code,
            field_name="origin currency",
        )

        destination = (
            _normalise_currency_code(
                des_curr_code,
                field_name=(
                    "destination currency"
                ),
            )
        )

        if origin == destination:
            return 1.0

        cached = self._lookup_rate(
            origin,
            destination,
        )

        if cached is not None:
            return cached

        primary_rate = (
            self._fetch_primary_rate(
                origin,
                destination,
            )
        )

        if primary_rate is not None:
            self._cache_rate(
                origin,
                destination,
                primary_rate,
            )

            return primary_rate

        fallback_rate = (
            self._fetch_fallback_rate(
                origin,
                destination,
            )
        )

        if fallback_rate is not None:
            safe_log(
                logger,
                logging.INFO,
                (
                    "provider.currency."
                    "fallback_used"
                ),
            )

            self._cache_rate(
                origin,
                destination,
                fallback_rate,
            )

            return fallback_rate

        safe_log(
            logger,
            logging.ERROR,
            "provider.currency.all_failed",
        )

        raise CurrencyRateUnavailableError(
            "No exchange rate is currently "
            f"available for "
            f"{origin}->{destination}."
        )

    def _lookup_rate(
        self,
        origin: str,
        destination: str,
    ) -> float | None:
        with self._rate_lock:
            return self._rate_cache.get(
                (
                    origin,
                    destination,
                )
            )

    def _cache_rate(
        self,
        origin: str,
        destination: str,
        rate: float,
    ) -> None:
        with self._rate_lock:
            self._rate_cache[
                (
                    origin,
                    destination,
                )
            ] = rate

            self._rate_cache[
                (
                    destination,
                    origin,
                )
            ] = 1.0 / rate

    def _fetch_primary_rate(
        self,
        origin: str,
        destination: str,
    ) -> float | None:
        try:
            response = self._session.get(
                _PRIMARY_RATE_URL,
                params={
                    "apikey": self._api_key,
                    "base_currency": origin,
                    "currencies": destination,
                },
                timeout=(
                    _REQUEST_TIMEOUT_SECONDS
                ),
            )

            response.raise_for_status()

            payload = response.json()

            rate = _valid_rate(
                payload[
                    "data"
                ][destination]
            )

            if rate is not None:
                return rate

        except requests.HTTPError as exc:
            status = getattr(
                exc.response,
                "status_code",
                None,
            )

            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.currency."
                    "primary_http_error"
                ),
                status_code=status,
            )

        except (
            requests.RequestException,
            KeyError,
            TypeError,
            ValueError,
        ):
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.currency."
                    "primary_failed"
                ),
            )

        return None

    def _fetch_fallback_rate(
        self,
        origin: str,
        destination: str,
    ) -> float | None:
        try:
            response = self._session.get(
                _FALLBACK_RATE_URL.format(
                    origin=origin
                ),
                timeout=(
                    _REQUEST_TIMEOUT_SECONDS
                ),
            )

            response.raise_for_status()

            payload = response.json()

            if (
                payload.get("result")
                != "success"
            ):
                return None

            return _valid_rate(
                payload.get(
                    "rates",
                    {},
                ).get(
                    destination
                )
            )

        except (
            requests.RequestException,
            TypeError,
            ValueError,
        ):
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.currency."
                    "fallback_failed"
                ),
            )

            return None


__all__ = [
    "CurrencyProvider",
]