"""LangChain-facing currency tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import (
    BaseTool,
    tool,
)

from app.modules.budget.exceptions import (
    CurrencyProviderError,
)
from app.modules.budget.currency import (
    convert_amount,
)
from app.modules.budget.ports import (
    CurrencyProviderPort,
)


def build_chat_currency_conversion_tool(
    currency_provider: (
        CurrencyProviderPort
    ),
) -> BaseTool:
    """
    Create the conversational currency tool
    with its provider injected.
    """

    @tool(
        "chat_currency_conversion"
    )
    def chat_currency_conversion(
        from_country: str,
        to_country: str,
        amount: float,
    ) -> dict[str, Any]:
        """
        Convert an amount between currencies
        used by two countries.
        """

        try:
            from_currency = (
                currency_provider
                .get_currency_code(
                    from_country
                )
            )

            to_currency = (
                currency_provider
                .get_currency_code(
                    to_country
                )
            )

            rate = (
                currency_provider
                .get_currency_rate(
                    from_currency,
                    to_currency,
                )
            )

            converted = (
                convert_amount(
                    amount,
                    rate,
                )
            )

        except (
            CurrencyProviderError,
            ValueError,
        ) as exc:
            return {
                "error": str(exc)
            }

        return {
            "from": from_currency,
            "to": to_currency,
            "rate": rate,
            "original_amount": amount,
            "converted_amount":
                converted,
        }

    return chat_currency_conversion


__all__ = [
    "build_chat_currency_conversion_tool",
]