from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings
from services.appwrite_db import AppwriteDB

router = APIRouter()
security = HTTPBearer(auto_error=False)


async def verify_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    """Verify admin password from X-API-Key header or Bearer token."""
    # Check X-API-Key header first
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key == settings.admin_password:
        return True

    # Fall back to Bearer token
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization")
    if credentials.credentials != settings.admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    return True


@router.get("/api/admin/stats")
async def admin_stats(
    request: Request,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """Get system stats (admin only)."""
    redis = request.app.state.redis

    # Count queued jobs
    queue_len = await redis.llen("trm:jobs:queue")

    # Get all job keys
    job_keys = []
    async for key in redis.scan_iter("trm:jobs:*"):
        k = key.decode() if isinstance(key, bytes) else key
        if ":metrics" not in k and ":cancel" not in k and ":queue" not in k and ":latest" not in k:
            job_keys.append(k)

    # Count by status
    status_counts: Dict[str, int] = {}
    for key in job_keys:
        status = await redis.hget(key, "status")
        if status:
            s = status.decode() if isinstance(status, bytes) else status
            status_counts[s] = status_counts.get(s, 0) + 1

    # GPU info
    gpu_info = "unavailable"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_mem,
                "memory_allocated": torch.cuda.memory_allocated(0),
            }
    except Exception:
        pass

    return {
        "queue_length": queue_len,
        "total_jobs": len(job_keys),
        "status_counts": status_counts,
        "gpu": gpu_info,
    }


@router.get("/api/admin/jobs")
async def admin_list_jobs(
    request: Request,
    _: bool = Depends(verify_admin),
) -> Dict[str, Any]:
    """List all jobs (admin only)."""
    db: AppwriteDB = request.app.state.db
    result = await db.list_jobs()
    return result
