"""PostgresPreferenceStore: input checks and failure handling without a
database, then the SQL against a real PostgreSQL when ``TEST_DATABASE_URL``
is set (each test in its own throwaway schema)."""

import os
import uuid
from contextlib import contextmanager

import psycopg
import pytest

from app.infrastructure.database.user_preferences import PostgresPreferenceStore


@contextmanager
def _unreachable():
    raise psycopg.OperationalError("database down")
    yield  # pragma: no cover


def _store_without_database():
    def connect():
        raise AssertionError("must not touch the database")

    return PostgresPreferenceStore(connect=connect)


# ── No database ─────────────────────────────────────────────────────


def test_nothing_to_save_never_reaches_the_database():
    store = _store_without_database()

    assert store.insert_preferences("", {"interest": ["food"]}) is False
    assert store.insert_preferences("u1", {}) is False
    assert store.insert_preferences("u1", {"interest": ["  ", ""]}) is False
    assert store.fetch_preferences("") == {}


def test_an_outage_is_reported_not_raised_on_write():
    assert PostgresPreferenceStore(connect=_unreachable).insert_preferences("u1", {"interest": ["food"]}) is False


def test_an_outage_on_read_fails_soft_unless_asked_to_raise():
    store = PostgresPreferenceStore(connect=_unreachable)

    assert store.fetch_preferences("u1") == {}
    with pytest.raises(psycopg.OperationalError):
        store.fetch_preferences("u1", raise_on_error=True)


# ── Real PostgreSQL ─────────────────────────────────────────────────

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture
def schema():
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set")

    name = f"test_prefs_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{name}"')

    @contextmanager
    def connect():
        with psycopg.connect(DATABASE_URL, options=f"-c search_path={name}") as conn:
            yield conn

    try:
        yield connect
    finally:
        with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{name}" CASCADE')


def test_round_trip_without_duplicates(schema):
    store = PostgresPreferenceStore(connect=schema)
    store.ensure_schema()

    assert store.insert_preferences("u1", {"interest": ["Loves history.", "loves HISTORY."], "dietary": ["Vegan."]})
    assert store.insert_preferences("u1", {"interest": ["Loves history."]})

    assert store.fetch_preferences("u1") == {"interest": ["Loves history."], "dietary": ["Vegan."]}
    assert store.fetch_preferences("u2") == {}


def test_a_table_from_the_embedding_version_accepts_new_rows(schema):
    # Before this version every row needed an embedding; old rows keep theirs.
    with schema() as conn:
        conn.execute(
            """
            CREATE TABLE mem0_vector (
                id UUID PRIMARY KEY, user_id TEXT NOT NULL, category TEXT NOT NULL,
                preference_text TEXT NOT NULL, embedding REAL[] NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, category, preference_text)
            )
            """
        )
        conn.execute(
            "INSERT INTO mem0_vector (id, user_id, category, preference_text, embedding) "
            "VALUES (%s, 'u1', 'interest', 'Old one.', ARRAY[0.5])",
            (str(uuid.uuid4()),),
        )

    store = PostgresPreferenceStore(connect=schema)
    store.ensure_schema()
    store.ensure_schema()

    assert store.insert_preferences("u1", {"interest": ["New one."]})
    assert sorted(store.fetch_preferences("u1")["interest"]) == ["New one.", "Old one."]

    with schema() as conn:
        kept = conn.execute("SELECT embedding FROM mem0_vector WHERE preference_text = 'Old one.'").fetchone()
    assert kept == ([0.5],)
