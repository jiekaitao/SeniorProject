from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from services.appwrite_db import AppwriteDB
from services.training_manager import TrainingManager
from services.training_runner import TrainingRunner

from api.routes import health, chat, sessions, jobs, data, admin

logger = logging.getLogger(__name__)

redis_client: aioredis.Redis | None = None


async def _queue_worker(app: FastAPI) -> None:
    """Drain the Redis job queue and invoke the TrainingRunner for each job.

    Runs as a background task for the lifetime of the FastAPI app.
    """
    redis = app.state.redis
    db: AppwriteDB = app.state.db
    manager = TrainingManager(redis=redis, db=db)
    runner = TrainingRunner(manager=manager, redis=redis)

    while True:
        try:
            job_id = await manager.process_queue()
            if job_id is None:
                await asyncio.sleep(0.1)
                continue

            # Load full config from Redis hash that queue_job populated.
            raw = await redis.hget(f"trm:jobs:{job_id}", "config")
            if not raw:
                await manager.fail_job(job_id, "missing config in Redis")
                continue

            config_str = raw.decode() if isinstance(raw, bytes) else raw
            config = json.loads(config_str)

            logger.info("Queue worker starting training for job %s", job_id)
            await runner.run_training(job_id, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Queue worker iteration failed")
            await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: connect to Redis, Appwrite, and start queue worker."""
    global redis_client

    # Connect to Redis
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
    app.state.redis = redis_client

    # Create Appwrite DB client (falls back to Redis-backed local DB when
    # APPWRITE_PROJECT_ID is empty).
    app.state.db = AppwriteDB()

    # Start background queue worker
    worker_task = asyncio.create_task(_queue_worker(app))
    app.state.queue_worker = worker_task

    yield

    # Shutdown — cancel worker first so it doesn't hold connections.
    worker_task.cancel()
    try:
        await worker_task
    except (asyncio.CancelledError, Exception):
        pass

    if redis_client:
        await redis_client.close()
    await app.state.db.close()


app = FastAPI(
    title="TRM Spinner Worker",
    description="FastAPI backend for training Tiny Recursive Reasoning Models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(jobs.router)
app.include_router(data.router)
app.include_router(admin.router)
