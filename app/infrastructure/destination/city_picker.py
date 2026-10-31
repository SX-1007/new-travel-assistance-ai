"""Gemini-backed primary-city selection for country-only trips."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.infrastructure.llm.chat import ChatBackend, build_chat_model


_DESTINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You select one practical primary travel city for a country-only trip. "
            "The city must be physically inside the requested destination country. "
            "Prefer a major tourism base with broad transport, hotel, and attraction "
            "coverage. For a city-state, return the city-state name. Do not return a "
            "country, airport, attraction, or city from another country. Return only "
            "the required structured city field.",
        ),
        (
            "human",
            "Destination country: {country}\n"
            "Trip dates: {start_date} to {end_date}\n"
            "Travellers: {num_people}\n"
            "Previously rejected candidates: {rejected}",
        ),
    ]
)


class _DestinationChoice(BaseModel):
    """Structured city selected when the user supplies only a country."""

    city: str = Field(min_length=1)


class GeminiPrimaryCityPicker:
    """Ask Gemini for one candidate city; verification happens elsewhere."""

    def __init__(
        self,
        *,
        backends: Sequence[ChatBackend],
    ) -> None:
        llm = build_chat_model(
            backends,
            temperature=0.0,
            max_retries=2,
            thinking_budget=0,
        )

        self._chain = (
            _DESTINATION_PROMPT
            | llm.with_structured_output(_DestinationChoice)
        )

    def propose_primary_city(
        self,
        *,
        country: str,
        start_date: str,
        end_date: str,
        num_people: int,
        rejected: Sequence[str],
    ) -> str:
        """Return Gemini's candidate. Provider errors propagate."""

        proposal = self._chain.invoke(
            {
                "country": country,
                "start_date": start_date,
                "end_date": end_date,
                "num_people": num_people,
                "rejected": ", ".join(rejected) if rejected else "None",
            }
        )

        city = getattr(proposal, "city", "")

        return city if isinstance(city, str) else ""


__all__ = [
    "GeminiPrimaryCityPicker",
]
