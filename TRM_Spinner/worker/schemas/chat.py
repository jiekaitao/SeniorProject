from __future__ import annotations

from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, field_validator


class ChatState(str, Enum):
    GREETING = "greeting"
    CLASSIFICATION = "classification"
    DATA_COLLECTION = "data_collection"
    TRAINING = "training"
    COMPLETED = "completed"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v


class ChatResponse(BaseModel):
    message: str
    state: ChatState
    classification: Optional[str] = None
