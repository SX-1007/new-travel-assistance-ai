"""Factual web search from the composition root: Exa first only with a key."""

import pytest

from app.bootstrap import container
from app.infrastructure.config.settings import Settings
from app.infrastructure.factual_qa.exa_search import ExaWebSearch
from app.infrastructure.factual_qa.fallback_search import FallbackWebSearch
from app.infrastructure.factual_qa.web_search import SerpApiWebSearch


GETTERS = (container.get_factual_web_search, container.get_factual_exa_client, container.get_factual_search_client)


@pytest.fixture
def configure(monkeypatch):
    def apply(**values):
        settings = Settings.model_construct(SERPAPI_PLACE="place-key", SERPAPI_FACTUAL="", **values)
        monkeypatch.setattr(container, "_settings", lambda: settings)

        for getter in GETTERS:
            getter.cache_clear()

    yield apply

    for getter in GETTERS:
        if getter is not container.get_factual_web_search and getter.cache_info().currsize:
            getter().close()
        getter.cache_clear()


@pytest.mark.parametrize("key", ["", "   "])
def test_without_an_exa_key_serpapi_answers_alone(configure, key):
    configure(EXA_API_KEY=key)

    search = container.get_factual_web_search()

    assert isinstance(search, SerpApiWebSearch)
    # No Exa client exists, so shutdown has nothing of Exa's to close.
    assert container.get_factual_exa_client.cache_info().currsize == 0


def test_with_an_exa_key_exa_is_tried_before_serpapi(configure):
    configure(EXA_API_KEY="exa-key")

    search = container.get_factual_web_search()

    assert isinstance(search, FallbackWebSearch)
    assert isinstance(search._primary, ExaWebSearch)
    assert isinstance(search._backup, SerpApiWebSearch)
    # Exa has its own client; SerpAPI keeps the factual search client.
    assert search._primary._client is container.get_factual_exa_client()
    assert search._backup._client is container.get_factual_search_client()


def test_shutdown_releases_the_exa_client():
    assert container.get_factual_exa_client in container._ALL_GETTERS
