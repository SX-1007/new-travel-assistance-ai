"""A cancelled planning run stops its worker thread before the next paid
search, and the coroutine returns only once the thread has stopped, so a
planning slot held around it is never freed while searches still run."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.modules.itinerary.tests.test_planning_end_to_end import FakePlaceSearch
from app.modules.places.policies import SearchQuota
from app.orchestration.tests.node_fakes import Network, trip_state
from app.orchestration.workflows.planning import run_planning_transaction
from app.shared.cancellation import CancellationToken, OperationCancelled


class GatedPlaces(FakePlaceSearch):
    """Place search that holds every call until released; counts the
    searches that got past the quota (the paid ones)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.paid = 0
        self._lock = threading.Lock()

    def search(self, query, *, near=None, zoom=11, quota=None):
        self.entered.set()
        self.release.wait(10)
        result = super().search(query, near=near, zoom=zoom, quota=quota)

        with self._lock:
            self.paid += 1

        return result


def gated_network():
    network = Network()
    network.places = GatedPlaces()
    return network


def start(network, pool, **kwargs):
    planning = network.planning()
    return asyncio.ensure_future(
        run_planning_transaction(
            trip_state(),
            planner=planning.planner,
            booking=planning.booking,
            activities=planning.activities,
            routes=planning.routes,
            executor=pool,
            **kwargs,
        )
    )


# ── The token ───────────────────────────────────────────────────────


def test_a_token_raises_only_after_cancel():
    token = CancellationToken()
    token.raise_if_cancelled()

    token.cancel()

    assert token.cancelled
    with pytest.raises(OperationCancelled):
        token.raise_if_cancelled()


def test_cancellation_is_not_an_ordinary_exception():
    # ``except Exception`` blocks that turn provider failures into "no
    # result" must not swallow it.
    assert not issubclass(OperationCancelled, Exception)


def test_a_cancelled_quota_refuses_the_next_paid_search():
    token = CancellationToken()
    quota = SearchQuota(5, cancellation=token)
    assert quota.try_acquire()

    token.cancel()

    with pytest.raises(OperationCancelled):
        quota.try_acquire()
    assert quota.used == 1


# ── A running plan ──────────────────────────────────────────────────


def test_cancelling_a_plan_stops_its_thread_before_another_paid_search():
    network = gated_network()

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = start(network, pool)
            assert await asyncio.to_thread(network.places.entered.wait, 5)

            task.cancel()
            await asyncio.sleep(0.05)
            # Still waiting for the thread: whoever holds a planning slot
            # around this coroutine keeps holding it.
            assert not task.done()

            network.places.release.set()

            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())

    assert network.places.paid == 0


def test_a_cancelled_plan_releases_its_thread_for_the_next_plan():
    network = gated_network()

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = start(network, pool)
            assert await asyncio.to_thread(network.places.entered.wait, 5)
            task.cancel()
            network.places.release.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            # The one planning thread is free again: a new plan runs at once.
            update = await asyncio.wait_for(start(Network(), pool), timeout=10)

        return update

    update = asyncio.run(scenario())

    assert update["planning_outcome"] == "validated", update["planning_issue_codes"]


def test_a_thread_that_overruns_the_grace_period_is_logged_not_awaited(caplog):
    network = gated_network()

    async def scenario():
        pool = ThreadPoolExecutor(max_workers=1)

        try:
            task = start(network, pool, stop_grace_seconds=0.05)
            assert await asyncio.to_thread(network.places.entered.wait, 5)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        finally:
            network.places.release.set()
            pool.shutdown(wait=True)

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())

    assert "transaction.stop_overdue" in caplog.text
    assert network.places.paid == 0


def test_a_request_time_limit_stops_the_plan_too():
    # GraphRuntime runs turns under asyncio.wait_for; its timeout cancels
    # the run like any other cancellation.
    network = gated_network()

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as pool:
            planning = network.planning()
            run = run_planning_transaction(
                trip_state(),
                planner=planning.planner,
                booking=planning.booking,
                activities=planning.activities,
                routes=planning.routes,
                executor=pool,
            )
            threading.Timer(0.3, network.places.release.set).start()

            with pytest.raises(TimeoutError):
                await asyncio.wait_for(run, timeout=0.1)

    asyncio.run(scenario())

    assert network.places.paid == 0
