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
• **orjson** responses, **GZip** over 1 KB, explicit **CORS** origins
  (browsers reject ``*`` together with credentials).
• **Probes** — ``/health`` and ``/healthz`` are liveness (the process
  answers); ``/health/ready`` checks Postgres, vectors and checkpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

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

# ──────────────────────────────────────────────────────────────────── Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    # ASCII separators: Windows consoles mangle box-drawing characters.
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("travel_assistant")

# These clients log every request URL at INFO, and Supabase URLs carry the
# user id (``?id=eq.<user>``); keep only their warnings.
for _noisy in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_UNLOGGED_PATHS = frozenset({"/", "/health", "/healthz", "/health/ready", "/favicon.ico"})


# ──────────────────────────────────────────────────────────────────── Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open resources at startup; release them even if startup failed."""

    started = time.perf_counter()
    logger.info("Starting Travel Assistant AI...")

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
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Travel Assistant AI",
        description="Backend for self-adaptive travel planning agent",
        version=os.getenv("APP_VERSION", "1.0.0"),
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    register_exception_handlers(app)

    # ── Middleware (the last one added runs outermost) ──

    app.add_middleware(GZipMiddleware, minimum_size=1024)

    allowed_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Any localhost port too: Vite moves to 5174, 5175, … when 5173 is
        # busy, and a mismatched port must not silently break the app.
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"],
    )

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


# Module-level app instance (uvicorn: app.main:app)
app = create_app()
