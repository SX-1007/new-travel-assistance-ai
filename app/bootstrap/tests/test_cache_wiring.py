"""Cache stores from the composition root: one per user, off without settings."""

import pytest

from app.bootstrap import container
from app.infrastructure.cache.upstash import UpstashRestClient
from app.infrastructure.config.settings import Settings


STORE_GETTERS = (
    container.get_flight_cache_store,
    container.get_hotel_cache_store,
    container.get_place_cache_store,
    container.get_pin_cache_store,
    container.get_budget_cache_store,
)


@pytest.fixture
def configure(monkeypatch):
    def apply(**values):
        settings = Settings.model_construct(**values)
        monkeypatch.setattr(container, "_settings", lambda: settings)

        for getter in STORE_GETTERS:
            getter.cache_clear()

    yield apply

    for getter in STORE_GETTERS:
        if getter.cache_info().currsize:
            getter().close()
        getter.cache_clear()


def test_each_user_of_the_cache_has_its_own_client_and_session(configure):
    configure(UPSTASH_REDIS_REST_URL="https://db.upstash.io", UPSTASH_REDIS_REST_TOKEN="token")

    stores = [getter() for getter in STORE_GETTERS]
    clients = [store._client for store in stores]

    assert all(store.enabled for store in stores)
    assert all(isinstance(client, UpstashRestClient) for client in clients)
    assert len({id(client) for client in clients}) == len(clients)
    assert len({id(client._session) for client in clients}) == len(clients)


@pytest.mark.parametrize(
    "values",
    [
        {"UPSTASH_REDIS_REST_URL": "", "UPSTASH_REDIS_REST_TOKEN": ""},
        {"UPSTASH_REDIS_REST_URL": "https://db.upstash.io", "UPSTASH_REDIS_REST_TOKEN": "  "},
        {"UPSTASH_REDIS_REST_URL": " ", "UPSTASH_REDIS_REST_TOKEN": "token"},
    ],
)
def test_without_upstash_settings_the_cache_is_off(configure, values):
    configure(**values)

    assert not any(getter().enabled for getter in STORE_GETTERS)


def test_shutdown_closes_and_releases_every_store():
    for getter in STORE_GETTERS:
        assert getter in container._ALL_GETTERS
