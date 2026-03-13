from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreate(BaseModel):
    session_id: str
    user_id: str
    data_path: str
    config: Dict[str, Any]


class TrainingJob(BaseModel):
    id: str
    session_id: str
    user_id: str
    status: JobStatus
    data_path: str
    config: Dict[str, Any]
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TrainingMetrics(BaseModel):
    step: int
    total_steps: int
    loss: float
    lr: float
    accuracy: Optional[float] = None
    extra: Optional[Dict[str, float]] = None
