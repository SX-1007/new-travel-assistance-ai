"""PostgresUserProfileStore: input checks without a database, then the SQL
against a real PostgreSQL when ``TEST_DATABASE_URL`` is set.

Each database test runs in its own throwaway schema, so it never touches the
real ``user_profiles`` table and leaves nothing behind.
"""

import os
import uuid
from contextlib import contextmanager

import psycopg
import pytest

from app.infrastructure.database.user_profiles import PostgresUserProfileStore


ONBOARDING = {"name": "Aina", "home_country": "Malaysia", "home_state": "Selangor"}


@contextmanager
def _unreachable():
    raise psycopg.OperationalError("database down")
    yield  # pragma: no cover


def _store_without_database():
    def connect():
        raise AssertionError("must not touch the database")

    return PostgresUserProfileStore(connect=connect)


# ── No database ─────────────────────────────────────────────────────


def test_unknown_column_is_refused_before_any_query():
    store = _store_without_database()

    assert store.create_profile("u1", {"name": "A", "password": "x"}) is False
    assert store.update_profile("u1", {"is_admin": True}) is False


def test_empty_input_is_refused_before_any_query():
    store = _store_without_database()

    assert store.create_profile("", ONBOARDING) is False
    assert store.create_profile("u1", {}) is False
    assert store.update_profile("u1", {}) is False


def test_fetch_raises_on_outage_unless_told_to_fail_soft():
    store = PostgresUserProfileStore(connect=_unreachable)

    with pytest.raises(psycopg.OperationalError):
        store.fetch_profile("u1")

    assert store.fetch_profile("u1", raise_on_error=False) == {}


def test_writes_report_failure_instead_of_raising():
    store = PostgresUserProfileStore(connect=_unreachable)

    assert store.create_profile("u1", ONBOARDING) is False
    assert store.update_profile("u1", {"interests": ["food"]}) is False


# ── Real PostgreSQL ─────────────────────────────────────────────────

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture
def store():
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set")

    schema = f"test_profiles_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')

    @contextmanager
    def connect():
        with psycopg.connect(DATABASE_URL, options=f"-c search_path={schema}") as conn:
            yield conn

    profiles = PostgresUserProfileStore(connect=connect)
    profiles.ensure_schema()
    try:
        yield profiles
    finally:
        with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_missing_user_has_an_empty_profile(store):
    assert store.fetch_profile("nobody") == {}


def test_onboarding_round_trip_returns_plain_fields(store):
    assert store.create_profile("u1", ONBOARDING) is True

    assert store.fetch_profile("u1") == {
        "id": "u1",
        **ONBOARDING,
        "travel_pacing": None,
        "dietary_restrictions": None,
        "interests": None,
        "accommodation_preferences": None,
    }


def test_learned_lists_are_stored_as_lists(store):
    store.create_profile("u1", ONBOARDING)

    assert store.update_profile("u1", {"interests": ["food", "museums"], "travel_pacing": "relaxed"}) is True

    profile = store.fetch_profile("u1")
    assert profile["interests"] == ["food", "museums"]
    assert profile["travel_pacing"] == "relaxed"


def test_re_onboarding_keeps_learned_columns(store):
    store.create_profile("u1", ONBOARDING)
    store.update_profile("u1", {"interests": ["food"]})

    assert store.create_profile("u1", {**ONBOARDING, "home_state": "Penang"}) is True

    profile = store.fetch_profile("u1")
    assert profile["home_state"] == "Penang"
    assert profile["interests"] == ["food"]


def test_update_without_a_row_reports_false(store):
    assert store.update_profile("nobody", {"interests": ["food"]}) is False
    assert store.fetch_profile("nobody") == {}


def test_schema_creation_is_repeatable(store):
    store.ensure_schema()
    store.create_profile("u1", ONBOARDING)
    store.ensure_schema()

    assert store.fetch_profile("u1")["name"] == "Aina"
