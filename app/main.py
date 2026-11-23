"""
Travel Assistant AI — Application Entry Point

FastAPI backend for the self-adaptive travel-planning agent.

Key design decisions
--------------------
• **Nothing at import** — storage, the checkpointer and the graph start in
  ``lifespan``; importing this module needs no ``.env`` (tests rely on it).
• **Windows** — start with ``python run.py``: psycopg's async driver needs a
  SelectorEventLoop, and only run.py can choose it before the loop exists.
• **Errors** — every graph/provider/storage exception maps to a fixed-detail
  HTTP response in ``app/api/errors.py``; routes have no try/except.
• **orjson** responses, **GZip** over 1 KB, explicit **CORS** origins.
• **Probes** — ``/health`` and ``/healthz`` are liveness (the process
  answers); ``/health/ready`` checks Postgres, vectors and checkpoints.
• **ENVIRONMENT** (read without loading settings, see
  ``infrastructure/config/environment.py``). In "production": no ``/docs``,
  ``/redoc`` or ``/openapi.json``; CORS allows only the https origins in
  ``CORS_ORIGINS`` (required), without credentials and with a fixed header
  list; the container refuses to start with development sign-in or
  without the request-limit store. "development" keeps the open defaults.
• **Threads** — the shared pool (``asyncio.to_thread``: health checks,
  profile reads, memory extraction) gets ``SHARED_THREADS``; planning has
  its own small pool in the container, so long plans can't starve it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator, Final

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from app.api.errors import register_exception_handlers
from app.api.routers import chat, form, geocode, profile
from app.bootstrap.container import (
    init_graph,
    init_resources,
    shutdown_container,
    shutdown_graph,
)
from app.bootstrap.health import readiness, verify_startup_health
from app.infrastructure.config.environment import app_environment, deployment_value

# ──────────────────────────────────────────────────────────────────── Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    # ASCII separators: Windows consoles mangle box-drawing characters.
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("travel_assistant")

# These clients log every request URL at INFO, and provider URLs can carry
# API keys or user data in the query string; keep only their warnings.
for _noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_UNLOGGED_PATHS = frozenset({"/", "/health", "/healthz", "/health/ready", "/favicon.ico"})

# The default pool is min(32, CPUs + 4): 5 threads on a 1-CPU server, too
# few once requests wait on providers in threads.
SHARED_THREADS: Final = 32

_DEV_ORIGINS: Final = "http://localhost:3000,http://localhost:5173"
_API_METHODS: Final = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
# Everything the frontend sends; production allows nothing else.
_API_HEADERS: Final = ["Authorization", "Content-Type", "X-Request-ID"]
# Readable by the frontend's JavaScript across origins.
_EXPOSED_HEADERS: Final = ["X-Process-Time", "Retry-After"]


# ──────────────────────────────────────────────────────────────────── Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open resources at startup; release them even if startup failed."""

    started = time.perf_counter()
    logger.info("Starting Travel Assistant AI...")

    # Before the first to_thread below. asyncio.run shuts it down at exit.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=SHARED_THREADS, thread_name_prefix="shared")
    )

    try:
        try:
            # Blocking connects and schema checks: keep them off the loop.
            await asyncio.to_thread(init_resources)
            await init_graph()
        except Exception:
            logger.critical("Resource initialization failed.", exc_info=True)
            raise

        if not await asyncio.to_thread(verify_startup_health):
            logger.warning("Startup health check FAILED — degraded mode active.")

        logger.info("Startup finished in %.1f ms.", (time.perf_counter() - started) * 1000)
        yield

    finally:
        logger.info("Shutting down Travel Assistant AI...")

        try:
            await shutdown_graph()
        except Exception:
            logger.warning("Error shutting down graph.", exc_info=True)

        try:
            # Closes provider clients, then the Postgres pool.
            await asyncio.to_thread(shutdown_container)
        except Exception:
            logger.warning("Error shutting down container.", exc_info=True)

        logger.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────── App factory
def create_app() -> FastAPI:
    production = app_environment() == "production"

    app = FastAPI(
        title="Travel Assistant AI",
        description="Backend for self-adaptive travel planning agent",
        version=os.getenv("APP_VERSION", "1.0.0"),
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
        # Production publishes no API description.
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )

    register_exception_handlers(app)

    # ── Middleware (the last one added runs outermost) ──

    app.add_middleware(GZipMiddleware, minimum_size=1024)

    _add_cors(app, production=production)

    @app.middleware("http")
    async def _timing_and_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"

        if request.url.path not in _UNLOGGED_PATHS:
            logger.info("%s %s → %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)

        return response

    # ── Routers ──
    # Tags are declared once on each APIRouter; include_router would append
    # a second copy and list every route under two Swagger groups.
    app.include_router(chat.router, prefix="/api/chat")
    app.include_router(form.router, prefix="/api/form")
    app.include_router(profile.router, prefix="/api/profile")
    app.include_router(geocode.router, prefix="/api/geocode")

    # ── Root + probes ──
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Travel Assistant AI is running"}

    @app.get("/health", tags=["Health"])
    async def health():
        """Liveness: the process is up and answering."""
        return {"status": "ok"}

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        """Liveness alias used by some load balancers (e.g. AWS ALB)."""
        return {"status": "ok"}

    @app.get("/health/ready", tags=["Health"])
    async def ready():
        """Readiness: 503 until Postgres, vectors and checkpoints all work."""
        checks = await asyncio.to_thread(readiness)
        ok = all(checks.values())
        return ORJSONResponse(
            {"status": "ok" if ok else "unavailable", "checks": checks},
            status_code=200 if ok else 503,
        )

    return app


def _add_cors(app: FastAPI, *, production: bool) -> None:
    origins = [
        origin.strip()
        for origin in deployment_value("CORS_ORIGINS", "" if production else _DEV_ORIGINS).split(",")
        if origin.strip()
    ]

    if production:
        if not origins or any(not origin.startswith("https://") or "*" in origin for origin in origins):
            raise RuntimeError(
                "In production, CORS_ORIGINS must list the frontend's https:// origin(s), comma-separated."
            )

        # Sign-in is a bearer token, never a cookie: no credentials needed.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=_API_METHODS,
            allow_headers=_API_HEADERS,
            expose_headers=_EXPOSED_HEADERS,
            max_age=600,
        )
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Any localhost port too: Vite moves to 5174, 5175, … when 5173 is
        # busy, and a mismatched port must not silently break the app.
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=_API_METHODS,
        allow_headers=["*"],
        expose_headers=_EXPOSED_HEADERS,
    )


# Module-level app instance (uvicorn: app.main:app)
app = create_app()
