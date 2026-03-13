from __future__ import annotations

import shutil
from typing import Any, Dict

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/health")
async def health(request: Request) -> Dict[str, Any]:
    """Health check endpoint returning status, gpu, and redis info."""
    # Check Redis
    redis_status = "disconnected"
    try:
        redis = request.app.state.redis
        if redis and await redis.ping():
            redis_status = "connected"
    except Exception:
        redis_status = "disconnected"

    # Check GPU
    gpu_status = "unavailable"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_status = f"available ({gpu_name})"
    except Exception:
        gpu_status = "unavailable"

    return {
        "status": "ok",
        "gpu": gpu_status,
        "redis": redis_status,
    }
