from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_health_returns_200(client):
    """Health endpoint should return 200 with status, gpu, and redis info."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "gpu" in data
    assert "redis" in data


@pytest.mark.asyncio
async def test_health_redis_connected(client, mock_redis):
    """Health should report redis as connected when ping succeeds."""
    mock_redis.ping = AsyncMock(return_value=True)
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["redis"] == "connected"


@pytest.mark.asyncio
async def test_health_redis_disconnected(client, mock_redis):
    """Health should report redis as disconnected when ping fails."""
    mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["redis"] == "disconnected"
