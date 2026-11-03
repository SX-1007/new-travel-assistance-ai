"""GraphRuntime edge cases (scripted workflow) and the checkpointer's resilience."""

import asyncio
from types import SimpleNamespace

import pytest
from psycopg import OperationalError
from psycopg.errors import DuplicateColumn

from app.orchestration.checkpoints import ResilientAsyncPostgresSaver, setup_checkpointer
from app.orchestration.graph import GraphRuntime
from app.orchestration.session_access import SessionAccessService
from app.orchestration.tests.test_graph import OwnershipStore
from app.shared.exceptions import SessionAccessDenied, SessionBusy


class Workflow:
    def __init__(self, result=None, *, gate=None, error=None):
        self.result = result if result is not None else {"plan_revision": 1}
        self.gate = gate
        self.error = error
        self.calls = []

    async def ainvoke(self, turn_input, config=None):
        self.calls.append((turn_input, config))

        if self.gate is not None:
            await self.gate.wait()

        if self.error is not None:
            raise self.error

        return self.result


class Checkpoints:
    def __init__(self, saved=None, history=()):
        self.saved = saved
        self.history = list(history)
        self.scans = 0
        self.deleted = []

    async def aget_tuple(self, config):
        return self.saved

    async def alist(self, config, *, limit=None):
        self.scans += 1

        for item in self.history:
            yield item

    async def adelete_thread(self, thread_id):
        self.deleted.append(thread_id)
        self.saved = None


def checkpoint(values, owner="user-1"):
    return SimpleNamespace(checkpoint={"channel_values": values}, metadata={"user_id": owner})


def runtime(workflow=None, checkpoints=None, store=None, **kwargs):
    store = store or OwnershipStore()
    return (
        GraphRuntime(
            workflow=workflow or Workflow(),
            checkpointer=checkpoints or Checkpoints(),
            sessions=SessionAccessService(store=store),
            **kwargs,
        ),
        store,
    )


def new_trip(app, **kwargs):
    return app.invoke_new_trip({"messages": []}, thread_id="t", user_id="user-1", **kwargs)


def chat(app, message="hi", **kwargs):
    return app.invoke_chat(message, thread_id="t", user_id="user-1", **kwargs)


def test_config_carries_the_owner_for_the_nodes_and_the_checkpoint():
    workflow = Workflow()
    app, _ = runtime(workflow)

    asyncio.run(new_trip(app, request_id="req-9"))

    config = workflow.calls[0][1]
    assert config["configurable"] == {"thread_id": "t", "__user_id": "user-1", "request_id": "req-9"}
    assert config["metadata"]["user_id"] == "user-1"
    assert config["recursion_limit"] == 50


@pytest.mark.parametrize(
    ("call", "kwargs"),
    [
        (new_trip, {}),
        (chat, {"message": "  "}),
        (chat, {"budget_action": "accept_anything"}),
    ],
)
def test_invalid_input_is_refused_before_any_io(call, kwargs):
    workflow = Workflow()
    app, _ = runtime(workflow)

    with pytest.raises(ValueError):
        if call is new_trip:
            asyncio.run(app.invoke_new_trip({}, thread_id="", user_id="user-1"))
        else:
            asyncio.run(call(app, **kwargs))

    assert workflow.calls == []


def test_a_second_turn_on_the_same_session_is_refused_not_interleaved():
    async def scenario():
        gate = asyncio.Event()
        app, _ = runtime(Workflow(gate=gate), Checkpoints(saved=checkpoint({"plan_revision": 1})))
        app._sessions._store.backfill("t", "user-1")
        first = asyncio.create_task(chat(app))
        await asyncio.sleep(0)

        with pytest.raises(SessionBusy):
            await chat(app)

        gate.set()
        return await first

    assert not asyncio.run(scenario()).plan_changed


def test_a_failed_or_timed_out_initialisation_releases_its_reservation():
    app, store = runtime(Workflow(gate=asyncio.Event()), new_trip_timeout=0.01)

    with pytest.raises(TimeoutError):
        asyncio.run(new_trip(app))

    assert store.released == ["t"] and "t" not in store.rows

    app, store = runtime(Workflow(error=RuntimeError("planner crashed")))

    with pytest.raises(RuntimeError):
        asyncio.run(new_trip(app))

    assert store.released == ["t"]


def test_an_interrupted_initialisation_returns_its_saved_plan_once():
    store = OwnershipStore()
    store.finalize_expired = lambda session_id, user_id, lease: True
    app, _ = runtime(Workflow(), Checkpoints(saved=checkpoint({"plan_revision": 1})), store)

    result = asyncio.run(new_trip(app))

    assert result.state == {"plan_revision": 1} and result.plan_changed
    assert app._workflow.calls == []


def test_a_legacy_session_owner_is_adopted_from_checkpoint_metadata():
    app, store = runtime(Workflow({"plan_revision": 2}), Checkpoints(saved=checkpoint({"plan_revision": 1})))

    result = asyncio.run(chat(app))

    assert store.rows["t"] == ["user-1", "ready"] and result.plan_changed

    stranger, _ = runtime(checkpoints=Checkpoints(saved=checkpoint({}, owner="someone-else")))

    with pytest.raises(SessionAccessDenied):
        asyncio.run(chat(stranger))


def test_legacy_history_is_scanned_once_per_session():
    old = checkpoint({"plan_revision": 1, "draft_itinerary": [{"day": 1, "activities": []}]})
    checkpoints = Checkpoints(saved=checkpoint({"plan_revision": 2}), history=[old])
    workflow = Workflow({"plan_revision": 2})
    app, store = runtime(workflow, checkpoints)
    store.backfill("t", "user-1")

    asyncio.run(chat(app))
    asyncio.run(chat(app))

    assert checkpoints.scans == 1
    assert [entry["revision"] for entry in workflow.calls[0][0]["itinerary_history"]] == [1]
    assert "itinerary_history" not in workflow.calls[1][0]


# ── Delete ──────────────────────────────────────────────────────────


def delete(app, user_id="user-1"):
    return app.delete_session("t", user_id=user_id)


def test_the_owner_deletes_every_checkpoint_and_the_ownership_row():
    checkpoints = Checkpoints(saved=checkpoint({"plan_revision": 1}))
    app, store = runtime(checkpoints=checkpoints)
    store.backfill("t", "user-1")

    asyncio.run(delete(app))

    assert checkpoints.deleted == ["t"] and "t" not in store.rows


def test_another_user_cannot_delete_a_session():
    checkpoints = Checkpoints(saved=checkpoint({"plan_revision": 1}))
    app, store = runtime(checkpoints=checkpoints)
    store.backfill("t", "user-1")

    with pytest.raises(SessionAccessDenied):
        asyncio.run(delete(app, user_id="intruder"))

    assert checkpoints.deleted == [] and store.rows["t"] == ["user-1", "ready"]


def test_a_legacy_session_is_deleted_by_its_checkpoint_owner_and_a_missing_one_is_not_found():
    checkpoints = Checkpoints(saved=checkpoint({"plan_revision": 1}))
    app, store = runtime(checkpoints=checkpoints)

    asyncio.run(delete(app))

    assert checkpoints.deleted == ["t"] and store.rows == {}

    with pytest.raises(SessionAccessDenied):
        asyncio.run(delete(app))


def test_a_half_finished_delete_can_be_retried():
    # The checkpoints went but the ownership row didn't: the owner check
    # still passes, so the retry removes the row.
    checkpoints = Checkpoints(saved=None)
    app, store = runtime(checkpoints=checkpoints)
    store.backfill("t", "user-1")

    asyncio.run(delete(app))

    assert "t" not in store.rows


def test_a_session_is_not_deleted_while_a_turn_is_running():
    async def scenario():
        gate = asyncio.Event()
        checkpoints = Checkpoints(saved=checkpoint({"plan_revision": 1}))
        app, store = runtime(Workflow(gate=gate), checkpoints)
        store.backfill("t", "user-1")
        turn = asyncio.create_task(chat(app))
        await asyncio.sleep(0)

        with pytest.raises(SessionBusy):
            await delete(app)

        gate.set()
        await turn
        return checkpoints

    assert asyncio.run(scenario()).deleted == []


# ── Checkpointer ────────────────────────────────────────────────────


def test_a_dropped_connection_is_retried_then_surfaces():
    calls = []

    async def flaky(value):
        calls.append(value)
        if len(calls) < 3:
            raise OperationalError("SSL error: bad length")
        return value

    saver = SimpleNamespace(retry_attempts=3, retry_base_delay_seconds=0)

    assert asyncio.run(ResilientAsyncPostgresSaver._retry(saver, flaky, "ok")) == "ok"

    async def dead(value):
        raise OperationalError("gone")

    with pytest.raises(OperationalError):
        asyncio.run(ResilientAsyncPostgresSaver._retry(saver, dead, "x"))


class _Cursor:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def execute(self, sql, params=None):
        if sql.startswith("INSERT"):
            self.db.recorded.append(params[0])

    async def fetchone(self):
        return {"v": self.db.version}


class _Connection(_Cursor):
    def cursor(self):
        return _Cursor(self.db)


class _Pool:
    def __init__(self, db):
        self.db = db

    def connection(self):
        return _Connection(self.db)


class _Saver:
    MIGRATIONS = ["m0", "m1", "m2"]

    def __init__(self, failures):
        self.failures = failures
        self.version = 0
        self.recorded = []
        self.conn = _Pool(self)

    async def setup(self):
        if self.failures:
            self.failures -= 1
            raise DuplicateColumn("column task_path already exists")


def test_setup_heals_a_migration_recorded_too_late():
    saver = _Saver(failures=1)

    asyncio.run(setup_checkpointer(saver))

    assert saver.recorded == [1]


def test_setup_does_not_mask_a_real_schema_problem():
    saver = _Saver(failures=5)
    saver.version = 2  # nothing left to record

    with pytest.raises(DuplicateColumn):
        asyncio.run(setup_checkpointer(saver))
