from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from schemas.chat import ChatState


class SessionCreate(BaseModel):
    user_id: str


class Session(BaseModel):
    id: str
    user_id: str
    state: ChatState
    classification: Optional[str] = None
    data_path: Optional[str] = None
    job_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SessionUpdate(BaseModel):
    state: Optional[ChatState] = None
    classification: Optional[str] = None
    data_path: Optional[str] = None
    job_id: Optional[str] = None
