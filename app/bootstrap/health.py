"""Readiness of the storage the API needs: Postgres, the vector table and
the graph's checkpoint pool."""

from __future__ import annotations

import logging

from app.bootstrap.container import graph_ready, verify_resources
from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

_UNAVAILABLE_EVENTS = {
    "postgres": "bootstrap.health.postgres_unavailable",
    "travel_vectors": "bootstrap.health.vector_store_unavailable",
    "checkpoints": "bootstrap.health.checkpoints_unavailable",
}


def readiness() -> dict[str, bool]:
    """Each resource → ready. Blocking (database round trip)."""

    try:
        results = dict(verify_resources())
    except Exception:
        safe_log(logger, logging.ERROR, "bootstrap.health.check_failed")
        results = {"postgres": False, "travel_vectors": False}

    results["checkpoints"] = graph_ready()
    return results


def verify_startup_health() -> bool:
    """True only when every resource is ready; logs each one that is not."""

    results = readiness()

    for resource, ready in results.items():
        if not ready:
            safe_log(
                logger,
                logging.ERROR,
                _UNAVAILABLE_EVENTS.get(resource, "bootstrap.health.resource_unavailable"),
            )

    healthy = all(results.values())

    if healthy:
        safe_log(logger, logging.INFO, "bootstrap.health.ready")

    return healthy


__all__ = [
    "readiness",
    "verify_startup_health",
]
