"""SessionOwnershipRepository against a real PostgreSQL when
``TEST_DATABASE_URL`` is set (skipped otherwise), in a throwaway schema:
the chat-history index, ``touch`` and the one-time backfill of sessions
saved before ownership rows existed."""

import json
import os
import uuid
from contextlib import contextmanager

import psycopg
import pytest

from app.infrastructure.database.session_ownership import SessionOwnershipRepository


DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

# The columns of LangGraph's ``checkpoints`` table that the backfill reads.
_CHECKPOINTS_SQL = """
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)
"""

# The ownership table as it was before ``updated_at``.
_OLD_OWNERS_SQL = """
CREATE TABLE trip_session_owners (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('initializing', 'ready')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


@pytest.fixture
def database():
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set")

    schema = f"test_owners_{uuid.uuid4().hex[:12]}"

    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')

    @contextmanager
    def connect():
        with psycopg.connect(DATABASE_URL, options=f"-c search_path={schema}") as conn:
            yield conn

    try:
        yield connect
    finally:
        with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


def _checkpoint(connect, thread_id, checkpoint_id, ts, user_id):
    with connect() as conn:
        conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint, metadata) VALUES (%s, %s, %s, %s)",
            (thread_id, checkpoint_id, json.dumps({"ts": ts}), json.dumps({"user_id": user_id} if user_id else {})),
        )


def test_older_sessions_are_backfilled_once_in_their_own_order(database):
    with database() as conn:
        conn.execute(_CHECKPOINTS_SQL)
        conn.execute(_OLD_OWNERS_SQL)
        # A session that already had a row before updated_at existed.
        conn.execute("INSERT INTO trip_session_owners (session_id, user_id, state) VALUES ('owned', 'u1', 'ready')")

    _checkpoint(database, "legacy-old", "1", "2026-09-01T10:00:00+00:00", "u1")
    _checkpoint(database, "legacy-new", "1", "2026-09-05T10:00:00+00:00", "u1")
    _checkpoint(database, "legacy-new", "2", "2026-10-01T10:00:00+00:00", "u1")
    _checkpoint(database, "owned", "1", "2026-09-03T10:00:00+00:00", "u1")
    _checkpoint(database, "someone-else", "1", "2026-10-02T10:00:00+00:00", "u2")
    _checkpoint(database, "no-owner", "1", "2026-10-03T10:00:00+00:00", None)

    repository = SessionOwnershipRepository(connect=database)

    assert repository.backfill_history() == 3
    assert repository.backfill_history() == 0  # recorded: never runs twice

    # Newest checkpoint first; the existing row got its checkpoint's time.
    assert repository.recent_sessions("u1", 10) == ["legacy-new", "owned", "legacy-old"]
    assert repository.recent_sessions("u2", 10) == ["someone-else"]
    assert repository.verify("legacy-old", "u1") and not repository.verify("legacy-old", "u2")


def test_a_chat_turn_moves_its_session_to_the_top(database):
    repository = SessionOwnershipRepository(connect=database)

    for session_id in ("a", "b", "c"):
        assert repository.reserve(session_id, "u1")
        assert repository.complete(session_id, "u1")

    assert repository.recent_sessions("u1", 10)[0] == "c"

    assert repository.touch("a", "u1") is True
    assert repository.touch("a", "intruder") is False

    assert repository.recent_sessions("u1", 2) == ["a", "c"]


def test_a_fresh_database_needs_no_checkpoints_table(database):
    repository = SessionOwnershipRepository(connect=database)

    assert repository.backfill_history() == 0
    assert repository.recent_sessions("u1", 10) == []


def test_the_history_index_exists(database):
    repository = SessionOwnershipRepository(connect=database)
    repository.ensure_schema()

    with database() as conn:
        indexes = [row[0] for row in conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'trip_session_owners'"
        )]

    assert any("(user_id, updated_at DESC NULLS LAST)" in definition for definition in indexes)
