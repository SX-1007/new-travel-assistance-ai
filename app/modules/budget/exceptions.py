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


class BudgetProviderQuotaExhausted(BudgetAssessmentUnavailable):
    """The flight or hotel search quota is used up, so no price evidence can
    be fetched until it resets (retrying right away can't help)."""


__all__ = [
    "BudgetAssessmentUnavailable",
    "BudgetProviderQuotaExhausted",
    "CurrencyProviderError",
    "CurrencyCodeUnavailableError",
    "CurrencyRateUnavailableError",
]