from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from schemas.jobs import JobCreate, JobStatus
from services.appwrite_db import AppwriteDB


class TrainingManager:
    """Manages the training job queue using Redis and persists state to Appwrite."""

    QUEUE_KEY = "trm:jobs:queue"
    JOB_PREFIX = "trm:jobs:"
    CANCEL_PREFIX = "trm:jobs:"
    CANCEL_SUFFIX = ":cancel"

    def __init__(self, redis: aioredis.Redis, db: AppwriteDB) -> None:
        self.redis = redis
        self.db = db

    async def queue_job(self, job_create: JobCreate) -> str:
        """Queue a new training job. Returns the job ID."""
        job_id = str(uuid.uuid4())

        # Store job metadata in Redis hash
        job_data = {
            "id": job_id,
            "session_id": job_create.session_id,
            "user_id": job_create.user_id,
            "data_path": job_create.data_path,
            "config": json.dumps(job_create.config),
            "status": JobStatus.PENDING,
        }
        await self.redis.hset(f"{self.JOB_PREFIX}{job_id}", mapping=job_data)

        # Push job ID to the queue
        await self.redis.lpush(self.QUEUE_KEY, job_id)

        # Persist to Appwrite
        await self.db.create_job(job_id, {
            "session_id": job_create.session_id,
            "user_id": job_create.user_id,
            "config_json": json.dumps(job_create.config),
            "status": JobStatus.PENDING,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        return job_id

    async def process_queue(self) -> Optional[str]:
        """Pop the next job from the queue. Returns job_id or None."""
        result = await self.redis.brpop(self.QUEUE_KEY, timeout=1)
        if result is None:
            return None

        _, job_id = result
        if isinstance(job_id, bytes):
            job_id = job_id.decode()

        # Update status to running
        await self.redis.hset(f"{self.JOB_PREFIX}{job_id}", "status", JobStatus.RUNNING)
        await self.db.update_job(job_id, {"status": JobStatus.RUNNING})

        return job_id

    async def cancel_job(self, job_id: str) -> bool:
        """Set a cancellation flag for a running job."""
        await self.redis.set(f"{self.JOB_PREFIX}{job_id}{self.CANCEL_SUFFIX}", "1")

        # Update status
        await self.redis.hset(f"{self.JOB_PREFIX}{job_id}", "status", JobStatus.CANCELLED)
        await self.db.update_job(job_id, {"status": JobStatus.CANCELLED})
        return True

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Read job status from Redis hash."""
        data = await self.redis.hgetall(f"{self.JOB_PREFIX}{job_id}")
        if not data:
            return {"status": "not_found"}

        # Decode bytes if needed
        decoded = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            decoded[key] = val

        return decoded

    async def complete_job(self, job_id: str) -> None:
        """Mark a job as completed, and bubble the status up to the session."""
        await self.redis.hset(f"{self.JOB_PREFIX}{job_id}", "status", JobStatus.COMPLETED)
        await self.db.update_job(job_id, {"status": JobStatus.COMPLETED})
        # Mirror completion to the session so the chat/results pages don't
        # have to keep polling /jobs/{id}/latest to infer the final state.
        try:
            raw = await self.redis.hget(f"{self.JOB_PREFIX}{job_id}", "session_id")
            session_id = raw.decode() if isinstance(raw, bytes) else raw
            if session_id:
                await self.db.update_session(session_id, {"status": "completed"})
        except Exception:
            pass

    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed with an error message."""
        await self.redis.hset(
            f"{self.JOB_PREFIX}{job_id}",
            mapping={"status": JobStatus.FAILED, "error": error},
        )
        await self.db.update_job(job_id, {"status": JobStatus.FAILED, "error_message": error})

    async def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        val = await self.redis.get(f"{self.JOB_PREFIX}{job_id}{self.CANCEL_SUFFIX}")
        return val is not None
