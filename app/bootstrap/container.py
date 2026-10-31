"""Application dependency composition.

This module is the composition root responsible for connecting application
services to their concrete infrastructure implementations.

It is the ONLY module that reads ``Settings``. Every infrastructure class
receives its credentials through its constructor, so each one can be imported,
replaced or tested on its own without a complete ``.env``. Getters are lazy:
nothing is constructed until first requested.

Each provider gets its own client and HTTP session (and, where configured,
its own key), so a slow or failing feature can't hold up another one.
"""

from __future__ import annotations

import asyncio
import logging

from functools import lru_cache, partial
from typing import Any

from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import Runnable

from app.infrastructure.activities.idea_generator import (
    GeminiActivityIdeaGenerator,
)
from app.infrastructure.budget.assessment_cache import (
    FirestoreBudgetAssessmentCache,
)
from app.infrastructure.booking.cache import (
    FirebaseBookingSearchCache,
)
from app.infrastructure.booking.flight_provider import (
    SerpApiFlightProvider,
)
from app.infrastructure.booking.hotel_provider import (
    SerpApiHotelProvider,
)
from app.infrastructure.booking.iata_resolver import (
    GeminiIataResolver,
)
from app.infrastructure.config.settings import (
    Settings,
    get_settings,
)
from app.infrastructure.currency.provider import (
    CurrencyProvider,
)
from app.infrastructure.destination.city_picker import (
    GeminiPrimaryCityPicker,
)
from app.infrastructure.database.firebase import (
    FirestoreCacheStore,
)
from app.infrastructure.database.checkpoint_pool import (
    open_checkpoint_pool,
)
from app.infrastructure.database.postgres import (
    close_connection_pool,
    init_connection_pool,
    verify_connection_pool,
)
from app.infrastructure.database.session_ownership import (
    SessionOwnershipRepository,
)
from app.infrastructure.database.user_profiles import (
    PostgresUserProfileStore,
)
from app.infrastructure.factual_qa.gemini import (
    GeminiFactualAnswerWriter,
    GeminiGroundingReviewer,
    GeminiRequestClassifier,
)
from app.infrastructure.factual_qa.web_search import (
    SerpApiWebSearch,
)
from app.infrastructure.llm.chat import (
    ChatBackend,
    build_chat_model,
)
from app.infrastructure.llm.gemini import (
    GeminiEmbedder,
)
from app.infrastructure.llm.health import (
    BackendHealth,
)
from app.infrastructure.llm.rate_limit import (
    SlidingWindowRateLimiter,
)
from app.infrastructure.maps.directions import (
    MapboxDirectionsClient,
)
from app.infrastructure.maps.mapbox import (
    MapboxForwardGeocoder,
)
from app.infrastructure.maps.reverse_geocoder import (
    MapboxReverseGeocoder,
)
from app.infrastructure.output_review.gemini_reviewer import (
    GeminiOutputReviewer,
)
from app.infrastructure.places.cache import (
    FirestorePlaceSearchCache,
)
from app.infrastructure.places.serpapi_maps import (
    SerpApiMapsPlaceSearch,
)
from app.infrastructure.search.serpapi import (
    SerpApiClient,
)
from app.infrastructure.vector.semantic_preferences import (
    PgVectorSemanticPreferenceStore,
)
from app.infrastructure.vector.travel_options import (
    PgVectorTravelOptionStore,
)
from app.memory.extraction import (
    build_extraction_chain,
    run_memory_extraction_task,
)
from app.memory.service import (
    MemoryService,
)
from app.modules.activities.service import (
    ActivityPlanningService,
    ActivitySearchService,
)
from app.modules.amenities.service import (
    NearbyAmenityService,
)
from app.modules.budget.service import (
    BudgetAssessmentService,
)
from app.modules.booking.ports import (
    FlightSearchPort,
    HotelSearchPort,
)
from app.modules.booking.service import (
    BookingPlanningService,
)
from app.modules.destination.service import (
    DestinationResolutionService,
)
from app.modules.factual_qa.service import (
    FactualQAService,
)
from app.modules.itinerary.service import (
    ItineraryPlanningService,
)
from app.modules.output_review.service import (
    OutputReviewService,
)
from app.modules.places.service import (
    VerifiedPlaceSearch,
)
from app.modules.routing.service import (
    DailyRouteService,
)
from app.orchestration.checkpoints import (
    ResilientAsyncPostgresSaver,
    setup_checkpointer,
)
from app.orchestration.graph import (
    GraphRuntime,
    compile_graph,
)
from app.orchestration.nodes import (
    AgentModels,
    AgentToolset,
    EditServices,
    GraphServices,
    MemoryExtractionQueue,
    PlanningServices,
    ProfileLoader,
    ReplyServices,
    build_agent_toolset,
    build_nodes,
)
from app.orchestration.session_access import SessionAccessService
from app.orchestration.trip_intake import NewTripIntake
from app.modules.places.pins import PlacePinLocator
from app.modules.profile.service import OnboardingService

logger = logging.getLogger(__name__)


def _settings() -> Settings:
    return get_settings()


# ── Gemini pacing ───────────────────────────────────────────────────


@lru_cache(maxsize=None)
def get_gemini_rate_limiter(api_key: str, model: str) -> BaseRateLimiter | None:
    """The one limiter for a key and model.

    Google counts the quota per project and model, so every client that uses
    the same key and model must share one limiter, whatever task it serves.
    """

    requests_per_minute = _settings().GEMINI_REQUESTS_PER_MINUTE

    if requests_per_minute == 0:
        return None

    return SlidingWindowRateLimiter(max_requests=requests_per_minute)


@lru_cache(maxsize=1)
def get_openrouter_rate_limiter() -> BaseRateLimiter | None:
    """One limiter for the OpenRouter account: its free-model limit is per
    account, across models."""

    requests_per_minute = _settings().OPENROUTER_REQUESTS_PER_MINUTE

    if requests_per_minute == 0:
        return None

    return SlidingWindowRateLimiter(max_requests=requests_per_minute)


@lru_cache(maxsize=None)
def get_backend_health(provider: str, api_key: str, model: str) -> BackendHealth:
    """One cooldown record per provider, key and model, shared by every task:
    when a quota runs out, it runs out for all of them."""

    return BackendHealth(provider=provider, model=model)


def _chat_backends(api_key: str, model: str) -> tuple[ChatBackend, ...]:
    """A task's own Gemini key, then the backups in order: the second Gemini
    key on the same model, then OpenRouter. Unset backups are skipped."""

    settings = _settings()
    keys = [api_key]

    if settings.GEMINI_API_KEY_2 and settings.GEMINI_API_KEY_2 != api_key:
        keys.append(settings.GEMINI_API_KEY_2)

    backends = [
        ChatBackend(
            "gemini",
            key,
            model,
            get_gemini_rate_limiter(key, model),
            get_backend_health("gemini", key, model),
        )
        for key in keys
    ]

    if settings.OPENROUTER_API_KEY:
        backends.append(
            ChatBackend(
                "openrouter",
                settings.OPENROUTER_API_KEY,
                settings.OPENROUTER_MODEL,
                get_openrouter_rate_limiter(),
                get_backend_health("openrouter", settings.OPENROUTER_API_KEY, settings.OPENROUTER_MODEL),
            )
        )

    return tuple(backends)


# ── Storage ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_firestore_cache_store() -> FirestoreCacheStore:
    return FirestoreCacheStore(
        credential_json=_settings().FIREBASE_CREDENTIAL_JSON,
    )


@lru_cache(maxsize=1)
def get_user_profile_store() -> PostgresUserProfileStore:
    return PostgresUserProfileStore()


@lru_cache(maxsize=1)
def get_session_access_service() -> SessionAccessService:
    repository = SessionOwnershipRepository()

    return SessionAccessService(
        store=repository,
    )


# ── Embeddings / vector stores ──────────────────────────────────────


@lru_cache(maxsize=1)
def get_embedder() -> GeminiEmbedder:
    settings = _settings()

    return GeminiEmbedder(
        api_key=settings.effective_embed_api_key,
        model=settings.GEMINI_EMBED_MODEL,
    )


@lru_cache(maxsize=1)
def get_travel_option_store() -> PgVectorTravelOptionStore:
    return PgVectorTravelOptionStore(
        embedder=get_embedder(),
    )


@lru_cache(maxsize=1)
def get_semantic_preference_store() -> PgVectorSemanticPreferenceStore:
    return PgVectorSemanticPreferenceStore(
        embedder=get_embedder(),
    )


# ── Memory ──────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_memory_service() -> MemoryService:
    return MemoryService(
        profile_store=get_user_profile_store(),
        semantic_store=get_semantic_preference_store(),
    )


@lru_cache(maxsize=1)
def get_memory_extraction_chain() -> Runnable:
    settings = _settings()

    return build_extraction_chain(
        build_chat_model(
            _chat_backends(settings.effective_extractor_api_key, settings.GEMINI_UTILITY_MODEL),
            temperature=0.0,
            # extraction.py owns the retry loop; one attempt per loop pass.
            max_retries=1,
        )
    )


# ── External providers ──────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_currency_provider(
) -> CurrencyProvider:
    return CurrencyProvider(
        api_key=(
            _settings().FREECURRENCY_API
        ),
    )


@lru_cache(maxsize=1)
def get_mapbox_reverse_geocoder(
) -> MapboxReverseGeocoder:
    return MapboxReverseGeocoder(
        access_token=(
            _settings().MAPBOX_TOKEN
        ),
    )


@lru_cache(maxsize=1)
def get_mapbox_forward_geocoder(
) -> MapboxForwardGeocoder:
    """Separate session and cache from the reverse geocoder."""

    return MapboxForwardGeocoder(
        access_token=(
            _settings().MAPBOX_TOKEN
        ),
    )


@lru_cache(maxsize=1)
def get_mapbox_directions_client(
) -> MapboxDirectionsClient:
    """Separate session from both geocoders."""

    return MapboxDirectionsClient(
        access_token=(
            _settings().MAPBOX_TOKEN
        ),
    )


@lru_cache(maxsize=1)
def get_flight_search_client(
) -> SerpApiClient:
    return SerpApiClient(
        api_key=_settings().SERPAPI_FLIGHT,
    )


@lru_cache(maxsize=1)
def get_hotel_search_client() -> SerpApiClient:
    return SerpApiClient(api_key=_settings().SERPAPI_HOTEL)


@lru_cache(maxsize=1)
def get_place_search_client() -> SerpApiClient:
    return SerpApiClient(api_key=_settings().SERPAPI_PLACE)


@lru_cache(maxsize=1)
def get_flight_search_provider(
) -> FlightSearchPort:
    settings = _settings()

    return SerpApiFlightProvider(
        client=get_flight_search_client(),

        iata_resolver=GeminiIataResolver(
            backends=_chat_backends(settings.effective_iata_api_key, settings.GEMINI_UTILITY_MODEL),
        ),

        cache=FirebaseBookingSearchCache(
            store=get_firestore_cache_store(),
        ),
    )


@lru_cache(maxsize=1)
def get_hotel_search_provider() -> HotelSearchPort:
    """Hotel search with its own cache and an injected geocoder."""
    return SerpApiHotelProvider(
        client=get_hotel_search_client(),
        cache=FirebaseBookingSearchCache(
            store=get_firestore_cache_store(),
        ),
        location_verifier=get_mapbox_reverse_geocoder(),
    )


# ── Destination ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_destination_city_picker() -> GeminiPrimaryCityPicker:
    settings = _settings()

    return GeminiPrimaryCityPicker(
        backends=_chat_backends(settings.effective_destination_api_key, settings.GEMINI_UTILITY_MODEL),
    )


@lru_cache(maxsize=1)
def get_destination_resolution_service() -> DestinationResolutionService:
    """Country + optional cities → country-verified planning cities."""

    return DestinationResolutionService(
        geocoder=get_mapbox_forward_geocoder(),
        country_locator=get_mapbox_reverse_geocoder(),
        city_picker=get_destination_city_picker(),
    )


# ── Booking planning ────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_booking_planning_service() -> BookingPlanningService:
    """Trip flights + hotel skeleton and chat alternatives."""

    return BookingPlanningService(
        flights=get_flight_search_provider(),
        hotels=get_hotel_search_provider(),
    )


# ── Routing ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_daily_route_service() -> DailyRouteService:
    """Itinerary days → multi-profile routes and public GeoJSON maps."""

    return DailyRouteService(
        directions=get_mapbox_directions_client(),
        geocoder=get_mapbox_forward_geocoder(),
    )


# ── Planning transaction ────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_itinerary_planning_service() -> ItineraryPlanningService:
    """Booking → activities → routes → validation, with targeted retries.

    Stateless: the per-trip stages are bound by
    ``orchestration.workflows.planning`` on each call.
    """

    return ItineraryPlanningService()


# ── Places / activities / amenities ─────────────────────────────────


@lru_cache(maxsize=1)
def get_place_search_provider() -> SerpApiMapsPlaceSearch:
    """Cached, unverified Google Maps place search."""

    return SerpApiMapsPlaceSearch(
        client=get_place_search_client(),
        cache=FirestorePlaceSearchCache(
            store=get_firestore_cache_store(),
        ),
    )


@lru_cache(maxsize=1)
def get_pin_search_client() -> SerpApiClient:
    """Own client and HTTP session for chat map pins (same credential as
    place search), so pin traffic never shares a session with planning."""

    return SerpApiClient(api_key=_settings().SERPAPI_PLACE)


@lru_cache(maxsize=1)
def get_place_pin_locator() -> PlacePinLocator:
    """Exact coordinates for places named in chat replies."""

    return PlacePinLocator(
        search=SerpApiMapsPlaceSearch(
            client=get_pin_search_client(),
            cache=FirestorePlaceSearchCache(
                store=get_firestore_cache_store(),
            ),
        ),
    )


@lru_cache(maxsize=1)
def get_verified_place_search() -> VerifiedPlaceSearch:
    """Place search that keeps only destination-verified results."""

    return VerifiedPlaceSearch(
        search=get_place_search_provider(),
        geocoder=get_mapbox_forward_geocoder(),
        locator=get_mapbox_reverse_geocoder(),
    )


@lru_cache(maxsize=1)
def get_activity_idea_generator() -> GeminiActivityIdeaGenerator:
    """Gemini brainstorm/rescue idea generator."""

    settings = _settings()

    return GeminiActivityIdeaGenerator(
        backends=_chat_backends(settings.effective_activities_api_key, settings.GEMINI_CHAT_MODEL),
    )


@lru_cache(maxsize=1)
def get_activity_planning_service() -> ActivityPlanningService:
    """Itinerary skeleton → verified attractions and restaurants."""

    return ActivityPlanningService(
        ideas=get_activity_idea_generator(),
        places=get_verified_place_search(),
    )


@lru_cache(maxsize=1)
def get_activity_search_service() -> ActivitySearchService:
    """Chat-time attraction/restaurant search."""

    return ActivitySearchService(
        places=get_verified_place_search(),
    )


@lru_cache(maxsize=1)
def get_nearby_amenity_service() -> NearbyAmenityService:
    """Nearest police/hospital/pharmacy around a trusted origin."""

    return NearbyAmenityService(
        search=get_place_search_provider(),
        places=get_verified_place_search(),
        routes=get_daily_route_service(),
    )


# ── Budget ──────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_budget_assessment_service() -> BudgetAssessmentService:
    """Provider-grounded minimum budgets with a shared Firestore lease."""

    return BudgetAssessmentService(
        currency_provider=get_currency_provider(),
        flight_search=get_flight_search_provider(),
        hotel_search=get_hotel_search_provider(),
        cache=FirestoreBudgetAssessmentCache(
            store=get_firestore_cache_store(),
        ),
    )


# ── Profile ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_onboarding_service() -> OnboardingService:
    """Onboarding profile reads/writes; a save drops the planner's cached
    copy of that user's profile."""

    return OnboardingService(
        store=get_user_profile_store(),
        on_saved=_invalidate_cached_profile,
    )


def _invalidate_cached_profile(user_id: str) -> None:
    # Only a loader that already exists can hold a stale copy.
    if get_profile_loader.cache_info().currsize:
        get_profile_loader().invalidate(user_id)


# ── New-trip intake ─────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_new_trip_intake() -> NewTripIntake:
    """Profile origin, verified cities and the budget gate before planning."""

    return NewTripIntake(
        profiles=get_onboarding_service(),
        destinations=get_destination_resolution_service(),
        budget=get_budget_assessment_service(),
    )


# ── Lifecycle ───────────────────────────────────────────────────────


def init_resources() -> None:
    """Open storage and create schemas at application startup."""

    init_connection_pool(_settings().DATABASE_URL)

    get_user_profile_store().ensure_schema()
    get_travel_option_store().ensure_schema()
    get_semantic_preference_store().ensure_schema()


def verify_resources() -> dict[str, bool]:
    """Return readiness of each required startup resource."""

    database_ready = verify_connection_pool()

    return {
        "postgres": database_ready,
        "travel_vectors": (
            database_ready
            and get_travel_option_store().verify()
        ),
    }


# ── Output review ───────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_output_reviewer() -> GeminiOutputReviewer:
    """Tool-less Gemini reviewer for proposed public replies."""

    settings = _settings()

    return GeminiOutputReviewer(
        backends=_chat_backends(settings.GEMINI_API_KEY, settings.GEMINI_CHAT_MODEL),
    )


@lru_cache(maxsize=1)
def get_output_review_service() -> OutputReviewService:
    """Deterministic rules first, then the semantic reviewer."""

    return OutputReviewService(reviewer=get_output_reviewer())


# ── Factual Q&A ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_factual_search_client() -> SerpApiClient:
    """Own SerpAPI client and session for web evidence, even when the key
    falls back to the place-search key."""

    return SerpApiClient(api_key=_settings().effective_factual_serpapi_key)


@lru_cache(maxsize=1)
def get_factual_web_search() -> SerpApiWebSearch:
    return SerpApiWebSearch(client=get_factual_search_client())


@lru_cache(maxsize=1)
def get_factual_request_classifier() -> GeminiRequestClassifier:
    settings = _settings()

    return GeminiRequestClassifier(
        backends=_chat_backends(settings.GEMINI_API_KEY, settings.GEMINI_UTILITY_MODEL),
    )


@lru_cache(maxsize=1)
def get_factual_answer_writer() -> GeminiFactualAnswerWriter:
    settings = _settings()

    return GeminiFactualAnswerWriter(
        backends=_chat_backends(settings.GEMINI_API_KEY, settings.GEMINI_CHAT_MODEL),
    )


@lru_cache(maxsize=1)
def get_grounding_reviewer() -> GeminiGroundingReviewer:
    settings = _settings()

    return GeminiGroundingReviewer(
        backends=_chat_backends(settings.GEMINI_API_KEY, settings.GEMINI_UTILITY_MODEL),
    )


@lru_cache(maxsize=1)
def get_factual_qa_service() -> FactualQAService:
    """Route, retrieve and answer factual questions behind evidence gates."""

    return FactualQAService(
        classifier=get_factual_request_classifier(),
        search=get_factual_web_search(),
        writer=get_factual_answer_writer(),
        reviewer=get_grounding_reviewer(),
    )


# ── Graph ───────────────────────────────────────────────────────────

# Seconds to let background work (memory extraction, reservation releases)
# finish at shutdown.
_GRAPH_DRAIN_SECONDS = 5.0


@lru_cache(maxsize=1)
def get_chat_model() -> Any:
    """The agent's chat model (tools are bound per use in ``AgentModels``)."""

    settings = _settings()

    return build_chat_model(
        _chat_backends(settings.GEMINI_API_KEY, settings.GEMINI_CHAT_MODEL),
        temperature=0.2,
        max_retries=3,
        # Minimal thinking for the tool-calling agent. It does not fix the
        # Gemini 3 thought-signature error; prompts.model_context does.
        thinking_budget=0,
    )


@lru_cache(maxsize=1)
def get_agent_toolset() -> AgentToolset:
    return build_agent_toolset(
        booking=get_booking_planning_service(),
        place_search=get_activity_search_service(),
        amenities=get_nearby_amenity_service(),
        currency=get_currency_provider(),
    )


@lru_cache(maxsize=1)
def get_profile_loader() -> ProfileLoader:
    return ProfileLoader(get_memory_service().fetch_context)


@lru_cache(maxsize=1)
def get_memory_extraction_queue() -> MemoryExtractionQueue:
    return MemoryExtractionQueue(
        partial(
            run_memory_extraction_task,
            chain=get_memory_extraction_chain(),
            memory_service=get_memory_service(),
        ),
        on_saved=get_profile_loader().invalidate,
    )


@lru_cache(maxsize=1)
def get_graph_services() -> GraphServices:
    """Everything the graph nodes call out to."""

    routes = get_daily_route_service()
    reviewer = get_output_review_service()
    chat_model = get_chat_model()

    return GraphServices(
        planning=PlanningServices(
            planner=get_itinerary_planning_service(),
            booking=get_booking_planning_service(),
            activities=get_activity_planning_service(),
            routes=routes,
            currency=get_currency_provider(),
            budget=get_budget_assessment_service(),
        ),
        replies=ReplyServices(reviewer=reviewer, writer=chat_model),
        models=AgentModels.bind(chat_model, get_agent_toolset()),
        edits=EditServices(
            reviewer=reviewer,
            place_search=get_activity_search_service(),
            routes=routes,
        ),
        profiles=get_profile_loader(),
        factual=get_factual_qa_service(),
        destinations=get_destination_resolution_service(),
        memory=get_memory_extraction_queue(),
    )


class _GraphResources:
    """The running graph and the async resources only it uses."""

    runtime: GraphRuntime | None = None
    checkpoint_pool: Any = None
    lock: asyncio.Lock | None = None


async def init_graph() -> GraphRuntime:
    """Open the checkpointer and compile the graph (idempotent)."""

    if _GraphResources.lock is None:
        _GraphResources.lock = asyncio.Lock()

    async with _GraphResources.lock:
        if _GraphResources.runtime is not None:
            return _GraphResources.runtime

        pool = await open_checkpoint_pool(_settings().DATABASE_URL)

        try:
            saver = ResilientAsyncPostgresSaver(pool)
            await setup_checkpointer(saver)
            workflow = compile_graph(
                build_nodes(get_graph_services()),
                get_agent_toolset(),
                checkpointer=saver,
            )
        except BaseException:
            await pool.close()
            raise

        _GraphResources.checkpoint_pool = pool
        _GraphResources.runtime = GraphRuntime(
            workflow=workflow,
            checkpointer=saver,
            sessions=get_session_access_service(),
        )
        return _GraphResources.runtime


def graph_ready() -> bool:
    """True while the runtime exists and its checkpoint pool is open."""

    pool = _GraphResources.checkpoint_pool
    return _GraphResources.runtime is not None and pool is not None and not pool.closed


def get_graph_runtime() -> GraphRuntime:
    runtime = _GraphResources.runtime

    if runtime is None:
        raise RuntimeError("The graph is not initialised; call init_graph() at startup")

    return runtime


async def shutdown_graph() -> None:
    """Let background work finish briefly, then close the checkpoint pool."""

    runtime, pool = _GraphResources.runtime, _GraphResources.checkpoint_pool
    _GraphResources.runtime = _GraphResources.checkpoint_pool = None

    if get_memory_extraction_queue.cache_info().currsize:
        await get_memory_extraction_queue().drain(_GRAPH_DRAIN_SECONDS)

    if runtime is not None:
        await runtime.drain(_GRAPH_DRAIN_SECONDS)

    if pool is not None:
        await pool.close()


# Every cached getter, dependents before what they use. A new getter must be
# added here, or shutdown won't release it.
_ALL_GETTERS = (
    get_gemini_rate_limiter,
    get_openrouter_rate_limiter,
    get_backend_health,
    get_new_trip_intake,
    get_onboarding_service,
    get_place_pin_locator,
    get_pin_search_client,
    get_graph_services,
    get_memory_extraction_queue,
    get_profile_loader,
    get_agent_toolset,
    get_chat_model,
    get_factual_qa_service,
    get_grounding_reviewer,
    get_factual_answer_writer,
    get_factual_request_classifier,
    get_factual_web_search,
    get_factual_search_client,
    get_output_review_service,
    get_output_reviewer,
    get_itinerary_planning_service,
    get_booking_planning_service,
    get_nearby_amenity_service,
    get_activity_search_service,
    get_activity_planning_service,
    get_activity_idea_generator,
    get_verified_place_search,
    get_place_search_provider,
    get_budget_assessment_service,
    get_destination_resolution_service,
    get_destination_city_picker,
    get_daily_route_service,
    get_flight_search_provider,
    get_hotel_search_provider,
    get_flight_search_client,
    get_hotel_search_client,
    get_place_search_client,
    get_mapbox_directions_client,
    get_mapbox_forward_geocoder,
    get_mapbox_reverse_geocoder,
    get_currency_provider,
    get_memory_extraction_chain,
    get_memory_service,
    get_semantic_preference_store,
    get_travel_option_store,
    get_embedder,
    get_session_access_service,
    get_user_profile_store,
    get_firestore_cache_store,
)


def shutdown_container() -> None:
    """Close the clients that exist (never creating one just to close it),
    then clear every cache and close the Postgres pool."""

    closable_getters = (
        get_flight_search_client,
        get_hotel_search_client,
        get_place_search_client,
        get_pin_search_client,
        get_factual_search_client,
        get_mapbox_directions_client,
        get_mapbox_forward_geocoder,
        get_mapbox_reverse_geocoder,
        get_currency_provider,
    )

    for getter in closable_getters:
        if not getter.cache_info().currsize:
            continue

        try:
            getter().close()

        except Exception:
            logger.exception(
                "Failed to close infrastructure resource: %s",
                getter.__name__,
            )

    # Clearing releases references only; nothing is constructed.
    for getter in _ALL_GETTERS:
        getter.cache_clear()

    close_connection_pool()
