"""Onboarding profile: GET tells the client whether onboarding is done,
POST saves name and home country/state (the trip form reads its origin
from here). Store failures answer 503 (``ProfileUnavailable``)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response

from app.api.dependencies import CurrentUser
from app.api.providers import Onboarding
from app.api.schemas.profile import (
    OnboardingRequest,
    ProfileFields,
    ProfileResponse,
    ProfileSavedResponse,
)
from app.modules.profile.models import OnboardingProfile


router = APIRouter(tags=["profile"])


@router.get(
    "/",
    response_model=ProfileResponse,
    summary="Fetch the onboarding profile (to gate app usage)",
)
async def get_profile(
    user_id: CurrentUser,
    response: Response,
    onboarding: Onboarding,
) -> ProfileResponse:
    # Per-user data: never cached by the browser or a shared proxy.
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Vary"] = "X-User-ID"

    profile = await asyncio.to_thread(onboarding.get, user_id)

    if profile is None:
        return ProfileResponse(onboarded=False)

    return ProfileResponse(onboarded=True, profile=ProfileFields(**profile.as_record()))


@router.post(
    "/",
    response_model=ProfileSavedResponse,
    summary="Create or update the onboarding profile",
)
async def save_profile(
    request: OnboardingRequest,
    user_id: CurrentUser,
    onboarding: Onboarding,
) -> ProfileSavedResponse:
    await asyncio.to_thread(
        onboarding.save,
        user_id,
        OnboardingProfile(
            name=request.name,
            home_country=request.origin_country,
            home_state=request.origin_state,
        ),
    )
    return ProfileSavedResponse(user_id=user_id)
