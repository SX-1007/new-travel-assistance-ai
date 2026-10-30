"""Pure currency arithmetic used by the budget domain."""

from __future__ import annotations

import math


def convert_amount(
    amount: float,
    rate: float,
) -> float:
    """
    Convert an amount with a positive finite
    FX rate and round to cents.
    """

    if (
        isinstance(amount, bool)
        or not isinstance(
            amount,
            (int, float),
        )
    ):
        raise ValueError(
            "amount must be a finite number"
        )

    if (
        isinstance(rate, bool)
        or not isinstance(
            rate,
            (int, float),
        )
    ):
        raise ValueError(
            "rate must be a positive "
            "finite number"
        )

    numeric_amount = float(amount)
    numeric_rate = float(rate)

    if not math.isfinite(
        numeric_amount
    ):
        raise ValueError(
            "amount must be a finite number"
        )

    if (
        not math.isfinite(
            numeric_rate
        )
        or numeric_rate <= 0
    ):
        raise ValueError(
            "rate must be a positive "
            "finite number"
        )

    return round(
        numeric_amount
        * numeric_rate,
        2,
    )


def home_currency_amount(
    amount: float,
    currency: str,
    *,
    home: str,
    destination: str,
    rate: float,
) -> float:
    """``amount`` in ``currency`` expressed in the home currency.

    ``rate`` is destination units per home unit. Only the trip's two
    currencies are accepted: anything else must be converted first, by a
    provider-backed tool, never guessed.
    """

    code = currency.strip().upper() if isinstance(currency, str) else ""

    if code and code == home.upper():
        return convert_amount(amount, 1.0)

    if code and code == destination.upper():
        return convert_amount(amount, 1.0 / rate if rate else math.nan)

    raise ValueError(
        f"State the amount in {home} or {destination}; convert any other "
        "currency with chat_currency_conversion first."
    )


__all__ = [
    "convert_amount",
    "home_currency_amount",
]