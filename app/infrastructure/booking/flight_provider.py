"""SerpAPI-backed flight-search provider."""

from __future__ import annotations

import math
from collections.abc import Mapping
from concurrent.futures import (
    ThreadPoolExecutor,
)
from datetime import date
from typing import (
    Any,
    Protocol,
)
from urllib.parse import quote_plus

from pydantic import ValidationError

from app.infrastructure.search.serpapi import (
    SerpApiUnavailableError,
)
from app.modules.booking.flights import (
    filter_flights_by_budget,
    flight_matches_query_endpoints,
    normalise_iata_code,
)
from app.modules.booking.models import (
    FlightAirport,
    FlightResult,
)
from app.modules.booking.ports import (
    IataResolverPort,
)
from app.modules.booking.validation import (
    normalise_adults,
    normalise_currency_code,
)


class _SearchClient(Protocol):
    def search(
        self,
        params: Mapping[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        ...


class _SearchCache(Protocol):
    def get(
        self,
        *,
        namespace: str,
        key: Mapping[
            str,
            Any,
        ],
    ) -> Any | None:
        ...

    def set(
        self,
        *,
        namespace: str,
        key: Mapping[
            str,
            Any,
        ],
        payload: Any,
        ttl_hours: int = 1,
    ) -> bool:
        ...


class SerpApiFlightProvider:
    """
    Search and normalize flight results without
    orchestration dependencies.
    """

    def __init__(
        self,
        *,
        client: _SearchClient,
        iata_resolver: (
            IataResolverPort
        ),
        cache: _SearchCache,
    ) -> None:
        self._client = client

        self._iata_resolver = (
            iata_resolver
        )

        self._cache = cache

    def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        currency: str,
        max_budget: (
            float | None
        ) = None,
        adults: int = 1,
    ) -> list[FlightResult]:

        adults_count = (
            normalise_adults(
                adults
            )
        )

        travel_date = (
            _normalise_iso_date(
                departure_date
            )
        )

        currency_code = (
            normalise_currency_code(
                currency
            )
        )

        # IATA resolution may involve two
        # independent LLM calls, so these may
        # safely execute in parallel.
        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            origin_future = (
                executor.submit(
                    self
                    ._iata_resolver
                    .resolve_iata_code,
                    origin,
                )
            )

            destination_future = (
                executor.submit(
                    self
                    ._iata_resolver
                    .resolve_iata_code,
                    destination,
                )
            )

            origin_iata = (
                normalise_iata_code(
                    origin_future.result()
                )
            )

            destination_iata = (
                normalise_iata_code(
                    destination_future
                    .result()
                )
            )

        if (
            origin_iata is None
            or destination_iata is None
        ):
            return []

        # Cache using canonical IATA identity,
        # rather than raw country spelling.
        cache_key = {
            "origin_iata":
                origin_iata,

            "destination_iata":
                destination_iata,

            "date":
                travel_date,

            "currency":
                currency_code,

            "adults":
                adults_count,
        }

        flights = (
            _validated_cached_flights(
                self._cache.get(
                    namespace="flight",
                    key=cache_key,
                ),
                origin_iata=(
                    origin_iata
                ),
                destination_iata=(
                    destination_iata
                ),
            )
        )

        if not flights:
            params = {
                "engine":
                    "google_flights",

                "departure_id":
                    origin_iata,

                "arrival_id":
                    destination_iata,

                "outbound_date":
                    travel_date,

                # one-way search
                "type":
                    "2",

                "adults":
                    str(
                        adults_count
                    ),

                "currency":
                    currency_code,

                "hl":
                    "en",
            }

            booking_url = (
                "https://www.google.com/"
                "travel/flights?q="
                + quote_plus(
                    "flights from "
                    f"{origin_iata} "
                    "to "
                    f"{destination_iata} "
                    f"on {travel_date}"
                )
                + (
                    f"&curr="
                    f"{currency_code}"
                    "&hl=en"
                )
            )

            try:
                payload = (
                    self._client.search(
                        params
                    )
                )

            except (
                SerpApiUnavailableError
            ):
                return []

            if payload.get("error"):
                return []

            raw_flights = [
                *(
                    payload.get(
                        "best_flights"
                    )
                    or []
                ),
                *(
                    payload.get(
                        "other_flights"
                    )
                    or []
                ),
            ]

            flights = []

            for raw_flight in (
                raw_flights
            ):
                parsed = (
                    _parse_flight(
                        raw_flight,
                        booking_url=(
                            booking_url
                        ),
                    )
                )

                if (
                    parsed
                    is not None
                    and
                    flight_matches_query_endpoints(
                        parsed,
                        origin_iata=(
                            origin_iata
                        ),
                        destination_iata=(
                            destination_iata
                        ),
                    )
                ):
                    flights.append(
                        parsed
                    )

            flights.sort(
                key=lambda flight:
                    flight.price
            )

            if flights:
                self._cache.set(
                    namespace="flight",
                    key=cache_key,
                    payload=[
                        flight.model_dump(
                            mode="json"
                        )
                        for flight
                        in flights
                    ],
                )

        return (
            filter_flights_by_budget(
                flights,
                max_budget=max_budget,
            )
        )


def _validated_cached_flights(
    payload: Any,
    *,
    origin_iata: str,
    destination_iata: str,
) -> list[FlightResult]:

    if not isinstance(
        payload,
        list,
    ):
        return []

    flights: list[
        FlightResult
    ] = []

    for item in payload:
        try:
            flight = (
                FlightResult
                .model_validate(
                    item
                )
            )

        except ValidationError:
            continue

        if (
            flight_matches_query_endpoints(
                flight,
                origin_iata=(
                    origin_iata
                ),
                destination_iata=(
                    destination_iata
                ),
            )
        ):
            flights.append(
                flight
            )

    flights.sort(
        key=lambda flight:
            flight.price
    )

    return flights


def _parse_flight(
    raw_flight: Any,
    *,
    booking_url: str,
) -> FlightResult | None:

    if not isinstance(
        raw_flight,
        Mapping,
    ):
        return None

    price = (
        _finite_positive_float(
            raw_flight.get(
                "price"
            )
        )
    )

    segments = raw_flight.get(
        "flights"
    )

    if (
        price is None
        or not isinstance(
            segments,
            list,
        )
        or not segments
        or not all(
            isinstance(
                segment,
                Mapping,
            )
            for segment
            in segments
        )
    ):
        return None

    first = segments[0]
    last = segments[-1]

    departure_airport = (
        _parse_airport(
            first.get(
                "departure_airport"
            )
        )
    )

    arrival_airport = (
        _parse_airport(
            last.get(
                "arrival_airport"
            )
        )
    )

    if (
        departure_airport is None
        or arrival_airport is None
    ):
        return None

    layovers = [
        str(
            layover.get(
                "name"
            )
            or layover.get(
                "id"
            )
            or ""
        ).strip()
        for layover
        in (
            raw_flight.get(
                "layovers"
            )
            or []
        )
        if isinstance(
            layover,
            Mapping,
        )
    ]

    try:
        return FlightResult(
            airline=str(
                first.get(
                    "airline"
                )
                or "unknown"
            ),

            flight_number=str(
                first.get(
                    "flight_number"
                )
                or "N/A"
            ),

            airline_logo=str(
                raw_flight.get(
                    "airline_logo"
                )
                or first.get(
                    "airline_logo"
                )
                or ""
            ),

            travel_class=str(
                first.get(
                    "travel_class"
                )
                or ""
            ),

            airplane=str(
                first.get(
                    "airplane"
                )
                or ""
            ),

            departure_airport=(
                departure_airport
            ),

            arrival_airport=(
                arrival_airport
            ),

            departure_time=str(
                first.get(
                    "departure_time"
                )
                or (
                    departure_airport
                    .time
                )
                or ""
            ),

            arrival_time=str(
                last.get(
                    "arrival_time"
                )
                or (
                    arrival_airport
                    .time
                )
                or ""
            ),

            duration=(
                _non_negative_int(
                    raw_flight.get(
                        "total_duration",
                        first.get(
                            "duration",
                            0,
                        ),
                    )
                )
            ),

            stops=max(
                0,
                len(segments) - 1,
            ),

            layovers=[
                value
                for value
                in layovers
                if value
            ],

            price=price,

            booking_url=(
                booking_url
                or None
            ),
        )

    except ValidationError:
        return None


def _parse_airport(
    raw_airport: Any,
) -> FlightAirport | None:

    if not isinstance(
        raw_airport,
        Mapping,
    ):
        return None

    raw_identities = [
        raw_airport.get(
            field
        )
        for field in (
            "id",
            "code",
        )
        if raw_airport.get(
            field
        )
        not in (
            None,
            "",
        )
    ]

    if not raw_identities:
        return None

    identities = [
        normalise_iata_code(
            value
        )
        for value
        in raw_identities
    ]

    if (
        any(
            identity is None
            for identity
            in identities
        )
        or len(
            set(identities)
        )
        != 1
    ):
        return None

    canonical_iata = (
        identities[0]
    )

    if canonical_iata is None:
        return None

    # Store the one verified identity in
    # FlightAirport.id. No provider-specific
    # "code" field needs to leak into the
    # domain model.
    return FlightAirport(
        id=canonical_iata,

        name=str(
            raw_airport.get(
                "name"
            )
            or ""
        ),

        time=str(
            raw_airport.get(
                "time"
            )
            or ""
        ),
    )


def _normalise_iso_date(
    value: str,
) -> str:
    try:
        parsed = (
            date.fromisoformat(
                value
            )
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "departure_date must "
            "be a valid YYYY-MM-DD date"
        ) from exc

    return parsed.isoformat()


def _finite_positive_float(
    value: Any,
) -> float | None:

    if (
        isinstance(
            value,
            bool,
        )
        or value is None
    ):
        return None

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None

    if (
        not math.isfinite(
            number
        )
        or number <= 0
    ):
        return None

    return number


def _non_negative_int(
    value: Any,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        return 0

    try:
        number = int(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return 0

    return max(
        0,
        number,
    )


__all__ = [
    "SerpApiFlightProvider",
]