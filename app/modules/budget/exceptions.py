"""Provider-neutral budget and currency failure contracts."""

from __future__ import annotations


class CurrencyProviderError(RuntimeError):
    """Base error for failures of the currency capability."""


class CurrencyCodeUnavailableError(
    CurrencyProviderError
):
    """Country could not be resolved to an ISO-4217 currency."""


class CurrencyRateUnavailableError(
    CurrencyProviderError
):
    """No configured provider could return a valid FX rate."""


class BudgetAssessmentUnavailable(RuntimeError):
    """Required real pricing or exchange-rate evidence is unavailable."""


__all__ = [
    "BudgetAssessmentUnavailable",
    "CurrencyProviderError",
    "CurrencyCodeUnavailableError",
    "CurrencyRateUnavailableError",
]