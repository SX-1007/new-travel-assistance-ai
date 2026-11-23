"""Chat history from the ownership index, and the touch after each turn."""

import asyncio

from app.orchestration.checkpoints import MAX_HISTORY_SESSIONS, fetch_chat_history_checkpoints
from app.orchestration.tests.test_graph_runtime import Checkpoints, Workflow, chat, checkpoint, runtime


class Sessions:
    def __init__(self, ids):
        self.ids = ids
        self.asked = []

    def recent_sessions(self, user_id, limit):
        self.asked.append((user_id, limit))
        return self.ids[:limit]


def load(session_id):
    if session_id == "gone":
        return None
    return {"session_id": session_id, "updated_at": "", "state": {}}


def test_history_lists_the_users_sessions_newest_first_and_loads_only_those():
    sessions = Sessions(["newest", "gone", "older", "oldest"])
    loaded = []

    def recording_load(session_id):
        loaded.append(session_id)
        return load(session_id)

    records = fetch_chat_history_checkpoints("u1", 3, sessions=sessions, load=recording_load)

    assert [record["session_id"] for record in records] == ["newest", "older"]
    assert sorted(loaded) == ["gone", "newest", "older"]
    assert sessions.asked == [("u1", 3)]


def test_history_limits_are_bounded_and_need_a_user():
    sessions = Sessions([])

    fetch_chat_history_checkpoints("u1", 500, sessions=sessions, load=load)
    fetch_chat_history_checkpoints("u1", 0, sessions=sessions, load=load)

    assert sessions.asked == [("u1", MAX_HISTORY_SESSIONS), ("u1", 1)]
    assert fetch_chat_history_checkpoints("", 5, sessions=sessions, load=load) == []


def test_a_chat_turn_touches_its_session():
    app, store = runtime(checkpoints=Checkpoints(checkpoint({"plan_revision": 1})))
    store.rows["t"] = ["user-1", "ready"]

    asyncio.run(chat(app))

    assert store.touched == ["t"]


def test_a_failed_touch_never_fails_the_turn():
    app, store = runtime(workflow=Workflow({"plan_revision": 1}), checkpoints=Checkpoints(checkpoint({"plan_revision": 1})))
    store.rows["t"] = ["user-1", "ready"]

    def broken(session_id, user_id):
        raise ConnectionError("database down")

    store.touch = broken

    result = asyncio.run(chat(app))

    assert result.state == {"plan_revision": 1}
