from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from api.middleware.auth import verify_token
from schemas.chat import ChatState
from schemas.sessions import Session, SessionCreate, SessionUpdate
from services.appwrite_db import AppwriteDB

router = APIRouter()

# Redis key for transient session state (data_path, job_id, classification)
def _redis_state_key(session_id: str) -> str:
    return f"session:{session_id}:state"


@router.post("/api/sessions", response_model=Session)
async def create_session(
    body: SessionCreate,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Session:
    """Create a new chat session."""
    if body.user_id != user_id:
        raise HTTPException(status_code=403, detail="Cannot create session for another user")

    db: AppwriteDB = request.app.state.db
    redis = request.app.state.redis
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Appwrite schema fields only: user_id, title, status, problem_type, is_suitable, created_at, updated_at
    doc = await db.create_session(session_id, {
        "user_id": user_id,
        "title": "New Session",
        "status": ChatState.GREETING.value,
        "problem_type": "",
        "is_suitable": False,
        "created_at": now,
        "updated_at": now,
    })

    return Session(
        id=doc["$id"],
        user_id=user_id,
        state=ChatState.GREETING,
        created_at=now,
        updated_at=now,
    )


@router.get("/api/sessions/{session_id}", response_model=Session)
async def get_session(
    session_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Session:
    """Get a session by ID."""
    db: AppwriteDB = request.app.state.db
    redis = request.app.state.redis

    try:
        doc = await db.get_session(session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    if doc.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    # Get transient state from Redis
    redis_state = await redis.hgetall(_redis_state_key(session_id))
    classification = _decode(redis_state.get(b"classification") or redis_state.get("classification"))
    data_path = _decode(redis_state.get(b"data_path") or redis_state.get("data_path"))
    job_id = _decode(redis_state.get(b"job_id") or redis_state.get("job_id"))

    # status field in Appwrite maps to our ChatState
    status_val = doc.get("status", doc.get("state", "greeting"))

    return Session(
        id=doc["$id"],
        user_id=doc["user_id"],
        state=ChatState(status_val),
        classification=classification,
        data_path=data_path,
        job_id=job_id,
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


@router.patch("/api/sessions/{session_id}", response_model=Session)
async def update_session(
    session_id: str,
    body: SessionUpdate,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Session:
    """Update a session."""
    db: AppwriteDB = request.app.state.db
    redis = request.app.state.redis

    try:
        doc = await db.get_session(session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    if doc.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    update_data = body.model_dump(exclude_none=True)
    now = datetime.now(timezone.utc).isoformat()

    # Separate Appwrite fields from Redis fields
    appwrite_update: Dict[str, Any] = {"updated_at": now}
    redis_update: Dict[str, str] = {}

    if "state" in update_data:
        appwrite_update["status"] = update_data["state"].value
    if "classification" in update_data:
        redis_update["classification"] = update_data["classification"]
    if "data_path" in update_data:
        redis_update["data_path"] = update_data["data_path"]
    if "job_id" in update_data:
        redis_update["job_id"] = update_data["job_id"]

    updated = await db.update_session(session_id, appwrite_update)

    if redis_update:
        await redis.hset(_redis_state_key(session_id), mapping=redis_update)

    # Get full Redis state
    redis_state = await redis.hgetall(_redis_state_key(session_id))
    classification = _decode(redis_state.get(b"classification") or redis_state.get("classification"))
    data_path = _decode(redis_state.get(b"data_path") or redis_state.get("data_path"))
    job_id = _decode(redis_state.get(b"job_id") or redis_state.get("job_id"))

    status_val = updated.get("status", doc.get("status", doc.get("state", "greeting")))

    return Session(
        id=updated["$id"],
        user_id=updated.get("user_id", doc["user_id"]),
        state=ChatState(status_val),
        classification=classification,
        data_path=data_path,
        job_id=job_id,
        created_at=updated.get("created_at", doc.get("created_at")),
        updated_at=updated.get("updated_at", now),
    )


@router.get("/api/sessions/{session_id}/messages")
async def list_session_messages(
    session_id: str,
    request: Request,
    user_id: str = Depends(verify_token),
) -> Dict[str, Any]:
    """List all messages for a session, ordered by creation time."""
    db: AppwriteDB = request.app.state.db

    # Verify session ownership
    try:
        doc = await db.get_session(session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    if doc.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    query = json.dumps({"method": "equal", "attribute": "session_id", "values": [session_id]})
    result = await db.list_messages(queries=[query])
    return result


@router.get("/api/sessions")
async def list_sessions(
    request: Request,
    user_id: str = Depends(verify_token),
) -> Dict[str, Any]:
    """List all sessions for the current user."""
    db: AppwriteDB = request.app.state.db
    query = json.dumps({"method": "equal", "attribute": "user_id", "values": [user_id]})
    result = await db.list_sessions(queries=[query])
    return result


def _decode(val: Any) -> str | None:
    """Decode a bytes or str value, returning None if empty."""
    if val is None:
        return None
    if isinstance(val, bytes):
        val = val.decode()
    return val if val else None
