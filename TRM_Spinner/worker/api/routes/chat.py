from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.middleware.auth import verify_token
from schemas.chat import ChatRequest, ChatResponse, ChatState
from services.chat_engine import ChatEngine
from services.appwrite_db import AppwriteDB

router = APIRouter()

# Redis key for transient session state (data_path, job_id, classification)
def _redis_state_key(session_id: str) -> str:
    return f"session:{session_id}:state"


def _get_engine(request: Request) -> ChatEngine:
    return ChatEngine(redis=request.app.state.redis, db=request.app.state.db)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    user_id: str = Depends(verify_token),
) -> ChatResponse:
    """Process a chat message through the state machine."""
    engine = _get_engine(request)
    db: AppwriteDB = request.app.state.db
    redis = request.app.state.redis

    # Get session from Appwrite
    try:
        doc = await db.get_session(body.session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify ownership
    if doc.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    # Get transient state from Redis
    redis_state = await redis.hgetall(_redis_state_key(body.session_id))

    # Build merged session dict for the chat engine
    status_val = doc.get("status", doc.get("state", "greeting"))
    session = {
        "id": doc["$id"],
        "user_id": doc["user_id"],
        "state": status_val,
    }
    # Merge Redis transient fields
    for key in (b"classification", b"data_path", b"job_id"):
        str_key = key.decode() if isinstance(key, bytes) else key
        val = redis_state.get(key) or redis_state.get(str_key)
        if val:
            session[str_key] = val.decode() if isinstance(val, bytes) else val

    now = datetime.now(timezone.utc).isoformat()

    # Store user message
    await db.create_message(str(uuid.uuid4()), {
        "session_id": body.session_id,
        "user_id": user_id,
        "role": "user",
        "content": body.message,
        "metadata": "{}",
        "created_at": now,
    })

    # Process through state machine
    response = await engine.process_message(session, body.message)

    # Update Appwrite with persistent fields only (status, updated_at)
    appwrite_update = {
        "status": response.state.value,
        "updated_at": now,
    }
    await db.update_session(body.session_id, appwrite_update)

    # Update Redis with transient fields
    redis_update = {}
    if response.classification:
        redis_update["classification"] = response.classification
    if redis_update:
        await redis.hset(_redis_state_key(body.session_id), mapping=redis_update)

    # Store assistant message
    await db.create_message(str(uuid.uuid4()), {
        "session_id": body.session_id,
        "user_id": user_id,
        "role": "assistant",
        "content": response.message,
        "metadata": "{}",
        "created_at": now,
    })

    return response


@router.post("/api/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    user_id: str = Depends(verify_token),
) -> StreamingResponse:
    """Process a chat message and stream the response as SSE events."""
    engine = _get_engine(request)
    db: AppwriteDB = request.app.state.db
    redis = request.app.state.redis

    # Get session from Appwrite
    try:
        doc = await db.get_session(body.session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify ownership
    if doc.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your session")

    # Get transient state from Redis
    redis_state = await redis.hgetall(_redis_state_key(body.session_id))

    # Build merged session dict
    status_val = doc.get("status", doc.get("state", "greeting"))
    session = {
        "id": doc["$id"],
        "user_id": doc["user_id"],
        "state": status_val,
    }
    for key in (b"classification", b"data_path", b"job_id"):
        str_key = key.decode() if isinstance(key, bytes) else key
        val = redis_state.get(key) or redis_state.get(str_key)
        if val:
            session[str_key] = val.decode() if isinstance(val, bytes) else val

    now = datetime.now(timezone.utc).isoformat()

    # Store user message
    await db.create_message(str(uuid.uuid4()), {
        "session_id": body.session_id,
        "user_id": user_id,
        "role": "user",
        "content": body.message,
        "metadata": "{}",
        "created_at": now,
    })

    async def event_generator():
        full_text = ""
        final_state = None
        classification = None
        job_id = None

        async for event in engine.process_message_stream(session, body.message):
            if event["type"] == "text_delta":
                full_text += event["content"]
            elif event["type"] == "done":
                final_state = event.get("state")
                classification = event.get("classification")
                job_id = event.get("job_id")
            yield f"data: {json.dumps(event)}\n\n"

        # Persist assistant message after stream completes
        await db.create_message(str(uuid.uuid4()), {
            "session_id": body.session_id,
            "user_id": user_id,
            "role": "assistant",
            "content": full_text,
            "metadata": "{}",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        # Update session state
        if final_state:
            appwrite_update = {
                "status": final_state,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            await db.update_session(body.session_id, appwrite_update)

        # Update Redis with transient fields
        redis_update = {}
        if classification:
            redis_update["classification"] = classification
        if redis_update:
            await redis.hset(_redis_state_key(body.session_id), mapping=redis_update)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
