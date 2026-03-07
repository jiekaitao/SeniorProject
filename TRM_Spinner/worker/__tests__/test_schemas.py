from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.chat import ChatMessage, ChatRequest, ChatResponse, ChatState
from schemas.jobs import TrainingJob, JobCreate, JobStatus, TrainingMetrics
from schemas.sessions import Session, SessionCreate, SessionUpdate
from schemas.data import DataUpload, DataValidation, DataPoint


class TestChatSchemas:
    """Test chat-related Pydantic models."""

    def test_chat_state_values(self):
        assert ChatState.GREETING == "greeting"
        assert ChatState.CLASSIFICATION == "classification"
        assert ChatState.DATA_COLLECTION == "data_collection"
        assert ChatState.TRAINING == "training"
        assert ChatState.COMPLETED == "completed"

    def test_chat_message_valid(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_rejects_invalid_role(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid_role", content="Hello")

    def test_chat_request_valid(self):
        req = ChatRequest(
            session_id="session_123",
            message="I want to solve sudoku",
        )
        assert req.session_id == "session_123"
        assert req.message == "I want to solve sudoku"

    def test_chat_request_rejects_empty_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="s1", message="")

    def test_chat_response_valid(self):
        resp = ChatResponse(
            message="I can help with that!",
            state=ChatState.CLASSIFICATION,
            classification=None,
        )
        assert resp.state == ChatState.CLASSIFICATION

    def test_chat_response_with_classification(self):
        resp = ChatResponse(
            message="Classified as puzzle solving",
            state=ChatState.DATA_COLLECTION,
            classification="puzzle_solving",
        )
        assert resp.classification == "puzzle_solving"


class TestJobSchemas:
    """Test job-related Pydantic models."""

    def test_job_status_values(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_job_create_valid(self):
        job = JobCreate(
            session_id="session_123",
            user_id="user_456",
            data_path="/data/training",
            config={"epochs": 100, "lr": 0.001},
        )
        assert job.session_id == "session_123"
        assert job.config["epochs"] == 100

    def test_job_create_rejects_missing_fields(self):
        with pytest.raises(ValidationError):
            JobCreate(session_id="s1")  # type: ignore

    def test_training_job_valid(self):
        job = TrainingJob(
            id="job_789",
            session_id="session_123",
            user_id="user_456",
            status=JobStatus.RUNNING,
            data_path="/data/training",
            config={"epochs": 100},
        )
        assert job.status == JobStatus.RUNNING

    def test_training_metrics_valid(self):
        metrics = TrainingMetrics(
            step=100,
            total_steps=1000,
            loss=0.5,
            lr=0.001,
        )
        assert metrics.step == 100
        assert metrics.loss == 0.5

    def test_training_metrics_optional_extras(self):
        metrics = TrainingMetrics(
            step=100,
            total_steps=1000,
            loss=0.5,
            lr=0.001,
            accuracy=0.85,
        )
        assert metrics.accuracy == 0.85


class TestSessionSchemas:
    """Test session-related Pydantic models."""

    def test_session_create_valid(self):
        s = SessionCreate(user_id="user_123")
        assert s.user_id == "user_123"

    def test_session_valid(self):
        s = Session(
            id="session_123",
            user_id="user_456",
            state=ChatState.GREETING,
        )
        assert s.state == ChatState.GREETING

    def test_session_update_valid(self):
        u = SessionUpdate(
            state=ChatState.CLASSIFICATION,
            classification="puzzle_solving",
        )
        assert u.state == ChatState.CLASSIFICATION


class TestDataSchemas:
    """Test data-related Pydantic models."""

    def test_data_point_valid(self):
        dp = DataPoint(
            input=[[1, 0], [0, 1]],
            output=[[0, 1], [1, 0]],
        )
        assert dp.input == [[1, 0], [0, 1]]
        assert dp.output == [[0, 1], [1, 0]]

    def test_data_point_rejects_non_2d(self):
        with pytest.raises(ValidationError):
            DataPoint(input=[1, 2, 3], output=[3, 2, 1])  # type: ignore

    def test_data_upload_valid(self):
        upload = DataUpload(
            session_id="session_123",
            data=[
                DataPoint(input=[[1, 0]], output=[[0, 1]]),
                DataPoint(input=[[2, 1]], output=[[1, 2]]),
            ],
        )
        assert len(upload.data) == 2

    def test_data_upload_rejects_empty(self):
        with pytest.raises(ValidationError):
            DataUpload(session_id="s1", data=[])

    def test_data_validation_valid(self):
        v = DataValidation(
            valid=True,
            num_examples=10,
            max_grid_size=30,
            vocab_size=12,
        )
        assert v.valid is True
        assert v.num_examples == 10

    def test_data_validation_with_errors(self):
        v = DataValidation(
            valid=False,
            num_examples=0,
            max_grid_size=0,
            vocab_size=0,
            errors=["No valid data points found"],
        )
        assert not v.valid
        assert len(v.errors) == 1
