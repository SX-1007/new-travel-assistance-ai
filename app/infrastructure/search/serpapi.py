"""Reusable SerpAPI HTTP client infrastructure."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_SERPAPI_URL: Final = (
    "https://serpapi.com/search"
)

_REQUEST_TIMEOUT_SECONDS: Final = 15.0


_RETRY_STRATEGY = Retry(
    total=3,
    connect=3,
    read=2,
    status=3,
    backoff_factor=1.0,
    status_forcelist=(
        429,
        500,
        502,
        503,
        504,
    ),
    allowed_methods=frozenset(
        {"GET"}
    ),
    respect_retry_after_header=True,
    raise_on_status=False,
)


class SerpApiUnavailableError(
    RuntimeError
):
    """Raised when SerpAPI cannot return a valid response."""


def _build_http_session(
) -> requests.Session:
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


class SerpApiClient:
    """Injected-key HTTP adapter for one SerpAPI credential."""

    def __init__(
        self,
        *,
        api_key: str,
        session: (
            requests.Session | None
        ) = None,
        timeout_seconds: float = (
            _REQUEST_TIMEOUT_SECONDS
        ),
    ) -> None:
        key = api_key.strip()

        if not key:
            raise ValueError(
                "api_key must not be empty"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0"
            )

        self._api_key = key

        self._session = (
            session
            or _build_http_session()
        )

        self._owns_session = (
            session is None
        )

        self._timeout_seconds = float(
            timeout_seconds
        )

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def search(
        self,
        params: Mapping[
            str,
            Any,
        ],
    ) -> dict[str, Any]:

        request_params = dict(
            params
        )

        request_params[
            "api_key"
        ] = self._api_key

        try:
            response = (
                self._session.get(
                    _SERPAPI_URL,
                    params=request_params,
                    timeout=(
                        self._timeout_seconds
                    ),
                )
            )

            response.raise_for_status()

            payload = response.json()

        except requests.HTTPError as exc:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.serpapi."
                    "http_failed"
                ),
                status_code=getattr(
                    exc.response,
                    "status_code",
                    None,
                ),
            )

            raise SerpApiUnavailableError(
                "SerpAPI request failed."
            ) from exc

        except requests.RequestException as exc:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.serpapi."
                    "transport_failed"
                ),
            )

            raise SerpApiUnavailableError(
                "SerpAPI request failed."
            ) from exc

        except ValueError as exc:
            safe_log(
                logger,
                logging.WARNING,
                (
                    "provider.serpapi."
                    "invalid_json"
                ),
            )

            raise SerpApiUnavailableError(
                "SerpAPI returned invalid JSON."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise SerpApiUnavailableError(
                "SerpAPI returned an invalid payload."
            )

        return payload


__all__ = [
    "SerpApiClient",
    "SerpApiUnavailableError",
]