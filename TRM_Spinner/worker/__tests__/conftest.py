from __future__ import annotations

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure worker package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_redis():
    """Mock redis.asyncio.Redis client."""
    r = AsyncMock()
    r.ping = AsyncMock(return_value=True)
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock(return_value=True)
    r.hset = AsyncMock(return_value=True)
    r.hget = AsyncMock(return_value=None)
    r.hgetall = AsyncMock(return_value={})
    r.lpush = AsyncMock(return_value=1)
    r.brpop = AsyncMock(return_value=None)
    r.publish = AsyncMock(return_value=1)
    r.subscribe = AsyncMock()
    r.delete = AsyncMock(return_value=1)
    r.close = AsyncMock()
    return r


@pytest.fixture
def mock_appwrite():
    """Mock Appwrite database client."""
    client = AsyncMock()
    client.create_document = AsyncMock(return_value={"$id": "doc_123"})
    client.get_document = AsyncMock(return_value={"$id": "doc_123"})
    client.list_documents = AsyncMock(return_value={"documents": [], "total": 0})
    client.update_document = AsyncMock(return_value={"$id": "doc_123"})
    client.delete_document = AsyncMock(return_value=True)

    # Session helpers
    client.create_session = AsyncMock(return_value={"$id": "doc_123", "user_id": "test-user-001", "status": "greeting", "created_at": "2025-01-01T00:00:00+00:00", "updated_at": "2025-01-01T00:00:00+00:00"})
    client.get_session = AsyncMock(return_value={"$id": "doc_123", "user_id": "test-user-001", "status": "greeting", "created_at": "2025-01-01T00:00:00+00:00", "updated_at": "2025-01-01T00:00:00+00:00"})
    client.update_session = AsyncMock(return_value={"$id": "doc_123", "user_id": "test-user-001", "status": "greeting", "created_at": "2025-01-01T00:00:00+00:00", "updated_at": "2025-01-01T00:00:00+00:00"})
    client.list_sessions = AsyncMock(return_value={"documents": [], "total": 0})

    # Message helpers
    client.create_message = AsyncMock(return_value={"$id": "msg_123"})
    client.list_messages = AsyncMock(return_value={"documents": [], "total": 0})

    # Job helpers
    client.create_job = AsyncMock(return_value={"$id": "job_123"})
    client.get_job = AsyncMock(return_value={"$id": "job_123"})
    client.update_job = AsyncMock(return_value={"$id": "job_123"})
    client.list_jobs = AsyncMock(return_value={"documents": [], "total": 0})

    client.close = AsyncMock()
    return client


@pytest_asyncio.fixture
async def app(mock_redis, mock_appwrite):
    """Create a FastAPI test app with mocked dependencies."""
    with patch("main.redis_client", mock_redis):
        from main import app as fastapi_app
        fastapi_app.state.redis = mock_redis
        fastapi_app.state.db = mock_appwrite
        yield fastapi_app


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP client for testing the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_training_data():
    """Sample training data in JSON format (ARC-like grid pairs)."""
    return [
        {
            "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "output": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        },
        {
            "input": [[1, 0], [0, 1]],
            "output": [[0, 1], [1, 0]],
        },
        {
            "input": [[2, 1], [1, 2]],
            "output": [[1, 2], [2, 1]],
        },
    ]


@pytest.fixture
def auth_headers():
    """Auth headers using X-API-Key bypass for testing protected endpoints."""
    return {"X-API-Key": "Jack123123@!"}
