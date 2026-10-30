"""FastAPI dependencies that hand routes their application services.

Each one only asks the container (the composition root) for an object, so
tests replace any of them with ``app.dependency_overrides`` and routes never
import infrastructure. ``dependencies.py`` stays HTTP-only (auth, request id).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends

from app.api.errors import ServiceNotReady
from app.bootstrap import container
from app.modules.output_review.service import OutputReviewService
from app.modules.places.pins import PlacePinLocator
from app.modules.profile.service import OnboardingService
from app.orchestration.checkpoints import fetch_chat_history_checkpoints
from app.orchestration.graph import GraphRuntime
from app.orchestration.trip_intake import NewTripIntake


HistoryLoader = Callable[[str, int], list[dict[str, Any]]]


def graph_runtime() -> GraphRuntime:
    try:
        return container.get_graph_runtime()
    except RuntimeError as exc:
        raise ServiceNotReady from exc


def new_trip_intake() -> NewTripIntake:
    return container.get_new_trip_intake()


def output_review() -> OutputReviewService:
    return container.get_output_review_service()


def onboarding() -> OnboardingService:
    return container.get_onboarding_service()


def place_pin_locator() -> PlacePinLocator:
    return container.get_place_pin_locator()


def chat_history_loader() -> HistoryLoader:
    """Newest checkpoint per session for a user (blocking; run in a thread)."""

    return fetch_chat_history_checkpoints


Runtime = Annotated[GraphRuntime, Depends(graph_runtime)]
TripIntake = Annotated[NewTripIntake, Depends(new_trip_intake)]
Reviewer = Annotated[OutputReviewService, Depends(output_review)]
ChatHistoryLoader = Annotated[HistoryLoader, Depends(chat_history_loader)]
Onboarding = Annotated[OnboardingService, Depends(onboarding)]
PinLocator = Annotated[PlacePinLocator, Depends(place_pin_locator)]


__all__ = [
    "ChatHistoryLoader",
    "HistoryLoader",
    "Onboarding",
    "PinLocator",
    "Reviewer",
    "Runtime",
    "TripIntake",
    "chat_history_loader",
    "graph_runtime",
    "new_trip_intake",
    "onboarding",
    "output_review",
    "place_pin_locator",
]
