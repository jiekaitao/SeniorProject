from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from schemas.chat import ChatState
from services.chat_engine import ChatEngine


# Patch the LLM helper so it always returns the fallback (no real OpenAI calls)
@pytest.fixture(autouse=True)
def _no_llm():
    """Disable LLM calls in all chat flow tests — always use fallback."""
    with patch("services.chat_engine.settings") as mock_settings:
        mock_settings.openai_api_key = ""
        yield


class TestChatFlowStateMachine:
    """Test the chat state machine: greeting -> classification -> data_collection -> training -> completed."""

    @pytest.fixture
    def engine(self, mock_redis, mock_appwrite):
        return ChatEngine(redis=mock_redis, db=mock_appwrite)

    @pytest.mark.asyncio
    async def test_initial_state_is_greeting(self, engine):
        """New sessions should start in greeting state."""
        session = await engine.create_session("user_123")
        assert session["state"] == ChatState.GREETING

    @pytest.mark.asyncio
    async def test_greeting_to_classification(self, engine):
        """After user describes their problem, state should move to classification."""
        session = {"state": ChatState.GREETING, "user_id": "user_123", "id": "s1"}
        response = await engine.process_message(session, "I want to train a model for sudoku solving")
        assert response.state == ChatState.CLASSIFICATION

    @pytest.mark.asyncio
    async def test_classification_identifies_problem(self, engine):
        """Classification state should identify the problem type."""
        session = {"state": ChatState.CLASSIFICATION, "user_id": "user_123", "id": "s1"}
        response = await engine.process_message(session, "I want to solve sudoku puzzles with constraints")
        assert response.classification is not None
        assert response.classification != "unsuitable"
        assert response.state == ChatState.DATA_COLLECTION

    @pytest.mark.asyncio
    async def test_classification_unsuitable_stays(self, engine):
        """If problem is unsuitable, stay in classification and ask to rephrase."""
        session = {"state": ChatState.CLASSIFICATION, "user_id": "user_123", "id": "s1"}
        response = await engine.process_message(session, "hello how are you")
        assert response.state == ChatState.CLASSIFICATION

    @pytest.mark.asyncio
    async def test_data_collection_waits_for_data(self, engine):
        """In data_collection state, engine should prompt for data upload."""
        session = {
            "state": ChatState.DATA_COLLECTION,
            "user_id": "user_123",
            "id": "s1",
            "classification": "puzzle_solving",
        }
        response = await engine.process_message(session, "What format should my data be?")
        assert response.state == ChatState.DATA_COLLECTION

    @pytest.mark.asyncio
    async def test_data_collection_to_training(self, engine):
        """When data is uploaded and confirmed, move to training state."""
        session = {
            "state": ChatState.DATA_COLLECTION,
            "user_id": "user_123",
            "id": "s1",
            "classification": "puzzle_solving",
            "data_path": "/app/uploads/s1",
        }
        response = await engine.process_message(session, "start training")
        assert response.state == ChatState.TRAINING

    @pytest.mark.asyncio
    async def test_training_to_completed(self, engine):
        """When training finishes, state should become completed."""
        session = {
            "state": ChatState.TRAINING,
            "user_id": "user_123",
            "id": "s1",
            "classification": "puzzle_solving",
            "data_path": "/app/uploads/s1",
            "job_id": "job_123",
        }
        # Simulate job completion
        engine.redis.hget = AsyncMock(return_value=b"completed")
        response = await engine.process_message(session, "how is training going?")
        assert response.state == ChatState.COMPLETED

    @pytest.mark.asyncio
    async def test_training_still_running(self, engine):
        """If training is still running, stay in training state."""
        session = {
            "state": ChatState.TRAINING,
            "user_id": "user_123",
            "id": "s1",
            "classification": "puzzle_solving",
            "data_path": "/app/uploads/s1",
            "job_id": "job_123",
        }
        engine.redis.hget = AsyncMock(return_value=b"running")
        response = await engine.process_message(session, "how is training going?")
        assert response.state == ChatState.TRAINING

    @pytest.mark.asyncio
    async def test_completed_state_responds(self, engine):
        """Completed state should still respond to messages."""
        session = {
            "state": ChatState.COMPLETED,
            "user_id": "user_123",
            "id": "s1",
        }
        response = await engine.process_message(session, "thanks")
        assert response.state == ChatState.COMPLETED
        assert response.message  # Should have some response

    @pytest.mark.asyncio
    async def test_full_flow_greeting_through_classification(self, engine):
        """Test the full happy path from greeting through classification."""
        # Create session
        session = await engine.create_session("user_123")
        assert session["state"] == ChatState.GREETING

        # User describes problem -> classification
        response = await engine.process_message(session, "I want to solve grid puzzles like ARC")
        assert response.state == ChatState.CLASSIFICATION

        # Classification identifies problem
        session["state"] = ChatState.CLASSIFICATION
        response = await engine.process_message(session, "ARC puzzle grid transform reasoning")
        assert response.classification is not None
        assert response.state == ChatState.DATA_COLLECTION
