"""Gemini IATA resolver against a fake structured-output chain."""

import logging

import pytest

from app.infrastructure.booking.iata_resolver import (
    GeminiIataResolver,
    _IataCodeResponse,
)
from app.infrastructure.llm.chat import ChatBackend


class FakeChain:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def resolver(*results):
    iata = GeminiIataResolver(
        backends=[ChatBackend(provider="gemini", api_key="key", model="model")]
    )
    chain = FakeChain(*results)
    iata._chain = chain
    return iata, chain


def test_top_destination_needs_no_model_call():
    iata, chain = resolver()

    assert iata.resolve_iata_code("  Japan ") == "HND"
    assert chain.calls == []


def test_model_code_is_normalised_and_cached():
    iata, chain = resolver(_IataCodeResponse(iata_code=" cdg "))

    assert iata.resolve_iata_code("France") == "CDG"
    assert iata.resolve_iata_code("france") == "CDG"
    assert chain.calls == [{"country": "France"}]


def test_unknown_country_is_cached_without_a_model_call():
    iata, chain = resolver()

    assert iata.resolve_iata_code("Narnia") is None
    assert iata.resolve_iata_code("Narnia") is None
    assert chain.calls == []


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_country_returns_none(blank):
    iata, chain = resolver()

    assert iata.resolve_iata_code(blank) is None
    assert chain.calls == []


def test_empty_model_answer_returns_none_and_is_not_cached(caplog):
    # B18: no structured-output tool call -> the chain returns None.
    iata, chain = resolver(None, _IataCodeResponse(iata_code="CDG"))

    with caplog.at_level(logging.WARNING):
        assert iata.resolve_iata_code("France") is None

    assert "provider.iata.invalid_response" in caplog.text
    # The miss was not cached, so a retry asks the model again.
    assert iata.resolve_iata_code("France") == "CDG"
    assert len(chain.calls) == 2


@pytest.mark.parametrize("answer", ["PARIS", "C1G", "", "ÇDG"])
def test_invalid_code_returns_none_and_is_not_cached(answer):
    iata, chain = resolver(
        _IataCodeResponse(iata_code=answer),
        _IataCodeResponse(iata_code="CDG"),
    )

    assert iata.resolve_iata_code("France") is None
    assert iata.resolve_iata_code("France") == "CDG"


def test_unexpected_answer_shape_returns_none():
    iata, _ = resolver({"iata_code": "CDG"})

    assert iata.resolve_iata_code("France") is None


def test_model_failure_returns_none_and_is_not_cached(caplog):
    iata, chain = resolver(
        RuntimeError("quota"),
        _IataCodeResponse(iata_code="CDG"),
    )

    with caplog.at_level(logging.WARNING):
        assert iata.resolve_iata_code("France") is None

    assert "provider.iata.failed" in caplog.text
    assert "quota" not in caplog.text
    assert iata.resolve_iata_code("France") == "CDG"
