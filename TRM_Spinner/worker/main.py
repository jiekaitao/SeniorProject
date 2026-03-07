from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from services.appwrite_db import AppwriteDB

from api.routes import health, chat, sessions, jobs, data, admin

redis_client: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: connect to Redis and Appwrite."""
    global redis_client

    # Connect to Redis
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
    app.state.redis = redis_client

    # Create Appwrite DB client
    app.state.db = AppwriteDB()

    yield

    # Shutdown
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
