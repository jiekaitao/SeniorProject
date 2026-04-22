from __future__ import annotations

import asyncio
import glob
import json
import os
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from api.middleware.auth import verify_token
from schemas.jobs import JobCreate, JobStatus, TrainingJob
from services.appwrite_db import AppwriteDB
from services.training_manager import TrainingManager

router = APIRouter()


@router.get("/api/jobs")
async def list_jobs(
    request: Request,
    user_id: str = Depends(verify_token),
) -> Dict[str, Any]:
    """List all jobs for the current user."""
    db: AppwriteDB = request.app.state.db
    query = json.dumps({"method": "equal", "attribute": "user_id", "values": [user_id]})
    result = await db.list_jobs(queries=[query])
    return result


@router.post("/api/jobs", response_model=TrainingJob)
async def create_job(
    body: JobCreate,
    request: Request,
    user_id: str = Depends(verify_token),
) -> TrainingJob:
    """Create and queue a new training job."""
    if body.user_id != user_id:
        raise HTTPException(status_code=403, detail="Cannot create job for another user")

    db: AppwriteDB = request.app.state.db
    manager = TrainingManager(redis=request.app.state.redis, db=db)

    job_id = await manager.queue_job(body)

    return TrainingJob(
        id=job_id,
        session_id=body.session_id,
        user_id=user_id,
        status=JobStatus.PENDING,
        data_path=body.data_path,
        config=body.config,
    )


@router.get("/api/jobs/{job_id}")
async def get_job(
    job_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Dict[str, Any]:
    """Get job status."""
    manager = TrainingManager(redis=request.app.state.redis, db=request.app.state.db)
    status = await manager.get_job_status(job_id)

    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")

    if status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your job")

    return status


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Dict[str, str]:
    """Cancel a running job."""
    manager = TrainingManager(redis=request.app.state.redis, db=request.app.state.db)
    status = await manager.get_job_status(job_id)

    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your job")

    await manager.cancel_job(job_id)
    return {"status": "cancelled"}


@router.get("/api/jobs/{job_id}/download")
async def download_weights(
    job_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> FileResponse:
    """Download trained model weights."""
    manager = TrainingManager(redis=request.app.state.redis, db=request.app.state.db)
    status = await manager.get_job_status(job_id)

    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your job")
    if status.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Training not completed yet")

    # Get checkpoint path from Redis
    redis = request.app.state.redis
    checkpoint_path = await redis.hget(f"trm:jobs:{job_id}", "checkpoint_path")
    if checkpoint_path:
        checkpoint_path = checkpoint_path.decode() if isinstance(checkpoint_path, bytes) else checkpoint_path

    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        raise HTTPException(status_code=404, detail="Model weights not found")

    # Find the latest checkpoint file
    step_files = sorted(glob.glob(os.path.join(checkpoint_path, "step_*")))
    if not step_files:
        # Try any .pt or .pth files
        step_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.pt")))
        step_files += sorted(glob.glob(os.path.join(checkpoint_path, "*.pth")))

    if not step_files:
        raise HTTPException(status_code=404, detail="No checkpoint files found")

    latest = step_files[-1]
    filename = f"trm_weights_{job_id[:8]}.pt"

    return FileResponse(
        path=latest,
        media_type="application/octet-stream",
        filename=filename,
    )


@router.get("/api/jobs/{job_id}/predictions")
async def get_predictions(
    job_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Dict[str, Any]:
    """Return the stored `predictions.json` for a completed job."""
    manager = TrainingManager(redis=request.app.state.redis, db=request.app.state.db)
    status = await manager.get_job_status(job_id)

    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your job")

    redis = request.app.state.redis
    pred_path = await redis.hget(f"trm:jobs:{job_id}", "predictions_path")
    if pred_path:
        pred_path = pred_path.decode() if isinstance(pred_path, bytes) else pred_path

    # Fall back: look at data_path/predictions.json
    if not pred_path:
        data_path = status.get("data_path")
        if data_path:
            candidate = os.path.join(data_path, "predictions.json")
            if os.path.exists(candidate):
                pred_path = candidate

    if not pred_path or not os.path.exists(pred_path):
        raise HTTPException(status_code=404, detail="Predictions not ready yet")

    with open(pred_path, "r") as f:
        return json.load(f)


@router.get("/api/jobs/{job_id}/latest")
async def get_latest_metrics(
    job_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Dict[str, Any]:
    """Return the most recent training metrics (from Redis cache)."""
    manager = TrainingManager(redis=request.app.state.redis, db=request.app.state.db)
    status = await manager.get_job_status(job_id)

    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your job")

    redis = request.app.state.redis
    raw = await redis.hgetall(f"trm:jobs:{job_id}:latest")
    result: Dict[str, Any] = {}
    for k, v in raw.items():
        key = k.decode() if isinstance(k, bytes) else k
        val = v.decode() if isinstance(v, bytes) else v
        # Try to coerce numerics.
        try:
            if "." in val:
                result[key] = float(val)
            else:
                result[key] = int(val)
        except (ValueError, TypeError):
            result[key] = val
    return {"job_id": job_id, "status": status.get("status"), "metrics": result}


@router.get("/api/jobs/{job_id}/stream")
async def stream_job(
    job_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> StreamingResponse:
    """SSE stream for real-time job updates using Redis SUBSCRIBE."""
    manager = TrainingManager(redis=request.app.state.redis, db=request.app.state.db)
    status = await manager.get_job_status(job_id)

    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    if status.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your job")

    async def event_generator():
        redis = request.app.state.redis
        pubsub = redis.pubsub()
        channel = f"trm:jobs:{job_id}:metrics"

        await pubsub.subscribe(channel)
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    yield f"data: {data}\n\n"

                    # Check for terminal events
                    try:
                        parsed = json.loads(data)
                        if parsed.get("status") in ("completed", "failed", "cancelled"):
                            break
                    except json.JSONDecodeError:
                        pass
                else:
                    # Send keepalive
                    yield ": keepalive\n\n"

                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
