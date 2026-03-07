from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, AsyncIterator, Dict

from config.settings import settings
from schemas.chat import ChatResponse, ChatState
from services.classifier import classify_problem
from services.data_generator import generate_training_data
from services.data_converter import convert_data

logger = logging.getLogger(__name__)

# Action tags the LLM can emit to trigger system behaviour
_ACTION_RE = re.compile(r"\[ACTION:([\w_]+)\]")


async def _llm_or_fallback(
    messages: list[dict],
    fallback: str,
    state: str = "",
    extra_context: str = "",
) -> str:
    """Try LLM completion; return fallback on any failure or missing key."""
    if not settings.openai_api_key:
        return fallback
    try:
        from services.llm import chat_completion
        return await chat_completion(messages, state=state, extra_context=extra_context)
    except Exception:
        logger.warning("LLM call failed, using fallback response", exc_info=True)
        return fallback


def _strip_actions(text: str) -> tuple[str, list[str]]:
    """Extract [ACTION:XXX] tags from text and return (clean_text, actions)."""
    actions = _ACTION_RE.findall(text)
    clean = _ACTION_RE.sub("", text).strip()
    return clean, actions


class ChatEngine:
    """State machine that drives the chat conversation flow.

    States: greeting -> classification -> data_collection -> training -> completed
    """

    def __init__(self, redis: Any, db: Any) -> None:
        self.redis = redis
        self.db = db

    async def create_session(self, user_id: str) -> Dict[str, Any]:
        """Create a new chat session in greeting state."""
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "user_id": user_id,
            "state": ChatState.GREETING,
        }
        return session

    async def process_message(self, session: Dict[str, Any], message: str) -> ChatResponse:
        """Process a user message based on the current session state."""
        state = session.get("state", ChatState.GREETING)
        if isinstance(state, str):
            state = ChatState(state)

        if state == ChatState.GREETING:
            return await self._handle_greeting(message)
        elif state == ChatState.CLASSIFICATION:
            return await self._handle_classification(message)
        elif state == ChatState.DATA_COLLECTION:
            return await self._handle_data_collection(session, message)
        elif state == ChatState.TRAINING:
            return await self._handle_training(session, message)
        elif state == ChatState.COMPLETED:
            return await self._handle_completed(message)
        else:
            return ChatResponse(
                message="Something went wrong. Please start a new session.",
                state=ChatState.GREETING,
            )

    async def _handle_greeting(self, message: str) -> ChatResponse:
        """In greeting state, any message transitions to classification."""
        fallback = (
            "Welcome! I'll help you train a reasoning model. "
            "Could you describe the type of problem you'd like the model to solve? "
            "For example: grid puzzles, routing problems, scheduling, etc."
        )
        reply = await _llm_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="greeting",
            extra_context="The user just started a new session. Welcome them and ask what kind of problem they want to train a model for.",
        )
        return ChatResponse(message=reply, state=ChatState.CLASSIFICATION)

    async def _handle_classification(self, message: str) -> ChatResponse:
        """Classify the problem type from the user's description."""
        result = classify_problem(message)

        if result.category == "unsuitable":
            fallback = (
                "I couldn't identify a clear problem type from your description. "
                "Could you provide more details? For example: "
                "'I want to solve grid transformation puzzles like ARC' or "
                "'I need to find shortest paths in graphs'."
            )
            reply = await _llm_or_fallback(
                [{"role": "user", "content": message}],
                fallback,
                state="classification",
                extra_context="The classifier could not identify a suitable problem type. Ask the user to rephrase with more specific details about the kind of reasoning problem they want to train.",
            )
            return ChatResponse(message=reply, state=ChatState.CLASSIFICATION)

        category_names = {
            "constraint_satisfaction": "Constraint Satisfaction",
            "routing": "Routing / Pathfinding",
            "scheduling": "Scheduling",
            "symbolic_math": "Symbolic Math",
            "puzzle_solving": "Puzzle Solving",
            "pattern_recognition": "Pattern Recognition",
        }

        name = category_names.get(result.category, result.category)
        fallback = (
            f"I've classified your problem as **{name}** "
            f"(confidence: {result.confidence:.0%}). "
            f"Now I need your training data. You can upload a file, "
            f"or ask me to generate examples for you."
        )
        reply = await _llm_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="classification",
            extra_context=(
                f"The classifier identified this as '{name}' with {result.confidence:.0%} confidence. "
                f"Tell the user the classification result. Then ask: do they want to upload training data, "
                f"or should you generate some examples? Also ask if they have preferences about grid size or number of examples."
            ),
        )
        return ChatResponse(
            message=reply,
            state=ChatState.DATA_COLLECTION,
            classification=result.category,
        )

    async def _handle_data_collection(self, session: Dict[str, Any], message: str) -> ChatResponse:
        """Handle data collection phase — LLM-driven with action tags."""
        data_path = session.get("data_path")
        classification = session.get("classification", "")

        # Build context about current state
        if data_path:
            state_summary = "Data has been uploaded and validated. Training can be started."
        else:
            state_summary = "No training data uploaded yet."

        extra = (
            f"Current state: {state_summary}\n"
            f"Problem classification: {classification or 'unknown'}\n\n"
            "You can include ACTION TAGS in your response to trigger system actions:\n"
            "  [ACTION:GENERATE_DATA] — generate training examples based on what the user described\n"
            "  [ACTION:START_TRAINING] — begin training (only if data is available)\n\n"
            "Decision guidelines:\n"
            "- If the user asks you to generate/create/make data or examples, include [ACTION:GENERATE_DATA]\n"
            "- If the user says to start/begin training and data is available, include [ACTION:START_TRAINING]\n"
            "- If data was just uploaded or generated and the user seems ready (e.g. 'looks good', 'let's go', 'train it'), include [ACTION:START_TRAINING]\n"
            "- If you need more information (grid size, problem details, number of examples), just ask — don't include any action tag\n"
            "- If no data exists and the user hasn't asked to generate any, ask what they'd like to do: upload a file or have you generate examples\n"
            "- Action tags are stripped from the displayed message, so write your response naturally around them"
        )

        reply = await _llm_or_fallback(
            [{"role": "user", "content": message}],
            self._data_collection_fallback(data_path, message),
            state="data_collection",
            extra_context=extra,
        )

        clean_reply, actions = _strip_actions(reply)

        # Handle GENERATE_DATA action
        if "GENERATE_DATA" in actions and not data_path:
            gen_response = await self._handle_data_generation(session, message)
            # Combine the LLM's conversational reply with generation result
            if gen_response.state == ChatState.DATA_COLLECTION:
                return gen_response
            return gen_response

        # Handle START_TRAINING action
        if "START_TRAINING" in actions and data_path:
            return ChatResponse(message=clean_reply or "Starting training now.", state=ChatState.TRAINING)

        return ChatResponse(message=clean_reply or reply, state=ChatState.DATA_COLLECTION)

    def _data_collection_fallback(self, data_path: str | None, message: str) -> str:
        """Generate a fallback response when LLM is unavailable."""
        lower = message.lower()

        if data_path and any(kw in lower for kw in ("start", "train", "begin", "go", "let's", "ready")):
            return "Starting training with your uploaded data. [ACTION:START_TRAINING]"

        if not data_path and any(kw in lower for kw in ("generate", "create", "make", "give", "produce", "sample", "example")):
            return "Generating training examples for you now. [ACTION:GENERATE_DATA]"

        if data_path:
            return (
                "Your data is uploaded and ready. "
                "Say 'start training' when you're ready, or upload more data."
            )

        return (
            "You can upload training data (any text format — JSON, CSV, TXT), "
            "or ask me to generate examples for you. "
            'Try saying "generate 10 examples" or upload a file.'
        )

    async def _handle_data_generation(self, session: Dict[str, Any], message: str) -> ChatResponse:
        """Generate training data via LLM and save it."""
        session_id = session.get("id", "unknown")
        classification = session.get("classification")

        grid_pairs, error = await generate_training_data(
            problem_description=message,
            classification=classification,
            num_examples=10,
        )

        if error or len(grid_pairs) == 0:
            return ChatResponse(
                message=f"Couldn't generate data: {error or 'no examples produced'}. Try describing your problem differently.",
                state=ChatState.DATA_COLLECTION,
            )

        # Convert and save
        upload_dir = os.path.abspath(
            os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "uploads"))
        )
        output_dir = os.path.join(upload_dir, session_id)
        os.makedirs(output_dir, exist_ok=True)

        try:
            result = convert_data(grid_pairs, output_dir=output_dir)
        except Exception as e:
            return ChatResponse(
                message=f"Generated data but conversion failed: {e}",
                state=ChatState.DATA_COLLECTION,
            )

        # Store data_path in Redis
        await self.redis.hset(f"session:{session_id}:state", "data_path", output_dir)

        fallback = (
            f"Generated {result.num_examples} training examples. "
            "The data looks good — want me to start training, or would you like to tweak anything first?"
        )
        reply = await _llm_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="data_collection",
            extra_context=(
                f"Successfully generated {result.num_examples} grid-pair examples and saved them. "
                f"Data is now ready for training. Tell the user the data was generated and ask if they want to "
                f"start training or adjust anything. If they seem eager, you can include [ACTION:START_TRAINING]."
            ),
        )
        clean_reply, actions = _strip_actions(reply)

        if "START_TRAINING" in actions:
            return ChatResponse(message=clean_reply or fallback, state=ChatState.TRAINING)

        return ChatResponse(message=clean_reply or fallback, state=ChatState.DATA_COLLECTION)

    async def _handle_training(self, session: Dict[str, Any], message: str) -> ChatResponse:
        """Handle training state - check job status."""
        job_id = session.get("job_id")

        if job_id:
            status = await self.redis.hget(f"trm:job:{job_id}", "status")
            if status:
                status_str = status.decode() if isinstance(status, bytes) else status

                if status_str == "completed":
                    return ChatResponse(
                        message="Training has completed successfully! Your model is ready.",
                        state=ChatState.COMPLETED,
                    )
                elif status_str == "failed":
                    error = await self.redis.hget(f"trm:job:{job_id}", "error")
                    error_str = error.decode() if isinstance(error, bytes) else (error or "Unknown error")
                    return ChatResponse(
                        message=f"Training failed: {error_str}",
                        state=ChatState.COMPLETED,
                    )
                elif status_str == "cancelled":
                    return ChatResponse(
                        message="Training was cancelled.",
                        state=ChatState.COMPLETED,
                    )
                else:
                    return ChatResponse(
                        message=f"Training is in progress (status: {status_str}). Please wait...",
                        state=ChatState.TRAINING,
                    )

        return ChatResponse(
            message="Training is being set up. Please wait...",
            state=ChatState.TRAINING,
        )

    async def _handle_completed(self, message: str) -> ChatResponse:
        """Handle completed state."""
        fallback = (
            "This session's training is complete. "
            "You can start a new session to train another model."
        )
        reply = await _llm_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="completed",
            extra_context="Training is finished. The user can start a new session if they want to train another model.",
        )
        return ChatResponse(message=reply, state=ChatState.COMPLETED)

    # ------------------------------------------------------------------ #
    #  Streaming variants                                                 #
    # ------------------------------------------------------------------ #

    async def _llm_stream_or_fallback(
        self,
        messages: list[dict],
        fallback: str,
        state: str = "",
        extra_context: str = "",
    ) -> AsyncIterator[str]:
        """Yield tokens from streaming LLM, or yield fallback as one chunk."""
        if not settings.openai_api_key:
            yield fallback
            return
        try:
            from services.llm import chat_completion_stream
            async for token in chat_completion_stream(messages, state=state, extra_context=extra_context):
                yield token
        except Exception:
            logger.warning("LLM stream failed, using fallback", exc_info=True)
            yield fallback

    @staticmethod
    def _parse_stream_for_actions(text: str) -> tuple[str, list[str]]:
        """Extract action tags from accumulated streamed text."""
        actions = _ACTION_RE.findall(text)
        clean = _ACTION_RE.sub("", text).strip()
        return clean, actions

    async def process_message_stream(
        self, session: Dict[str, Any], message: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process a user message and yield SSE event dicts."""
        state = session.get("state", ChatState.GREETING)
        if isinstance(state, str):
            state = ChatState(state)

        handler = {
            ChatState.GREETING: self._handle_greeting_stream,
            ChatState.CLASSIFICATION: self._handle_classification_stream,
            ChatState.DATA_COLLECTION: self._handle_data_collection_stream,
            ChatState.TRAINING: self._handle_training_stream,
            ChatState.COMPLETED: self._handle_completed_stream,
        }.get(state)

        if handler is None:
            yield {"type": "text_delta", "content": "Something went wrong. Please start a new session."}
            yield {"type": "done", "state": ChatState.GREETING.value}
            return

        if state in (ChatState.DATA_COLLECTION, ChatState.TRAINING):
            async for event in handler(session, message):
                yield event
        else:
            async for event in handler(message):
                yield event

    async def _handle_greeting_stream(self, message: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream greeting response."""
        fallback = (
            "Welcome! I'll help you train a reasoning model. "
            "Could you describe the type of problem you'd like the model to solve? "
            "For example: grid puzzles, routing problems, scheduling, etc."
        )
        async for token in self._llm_stream_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="greeting",
            extra_context="The user just started a new session. Welcome them and ask what kind of problem they want to train a model for.",
        ):
            yield {"type": "text_delta", "content": token}
        yield {"type": "done", "state": ChatState.CLASSIFICATION.value}

    async def _handle_classification_stream(self, message: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream classification response with tool call box."""
        # Tool call: classify
        yield {"type": "tool_start", "name": "classify_problem", "description": "Analyzing Problem Type"}

        result = classify_problem(message)

        if result.category == "unsuitable":
            yield {"type": "tool_done", "name": "classify_problem", "result": "No clear match found"}

            fallback = (
                "I couldn't identify a clear problem type from your description. "
                "Could you provide more details?"
            )
            async for token in self._llm_stream_or_fallback(
                [{"role": "user", "content": message}],
                fallback,
                state="classification",
                extra_context="The classifier could not identify a suitable problem type. Ask the user to rephrase.",
            ):
                yield {"type": "text_delta", "content": token}
            yield {"type": "done", "state": ChatState.CLASSIFICATION.value}
            return

        category_names = {
            "constraint_satisfaction": "Constraint Satisfaction",
            "routing": "Routing / Pathfinding",
            "scheduling": "Scheduling",
            "symbolic_math": "Symbolic Math",
            "puzzle_solving": "Puzzle Solving",
            "pattern_recognition": "Pattern Recognition",
        }
        name = category_names.get(result.category, result.category)

        yield {
            "type": "tool_done",
            "name": "classify_problem",
            "result": f"{name} ({result.confidence:.0%} confidence)",
        }

        fallback = (
            f"I've classified your problem as **{name}** "
            f"(confidence: {result.confidence:.0%}). "
            f"Now I need your training data. You can upload a file, "
            f"or ask me to generate examples for you."
        )
        async for token in self._llm_stream_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="classification",
            extra_context=(
                f"The classifier identified this as '{name}' with {result.confidence:.0%} confidence. "
                f"Tell the user the classification result. Then ask: do they want to upload training data, "
                f"or should you generate some examples?"
            ),
        ):
            yield {"type": "text_delta", "content": token}

        yield {
            "type": "done",
            "state": ChatState.DATA_COLLECTION.value,
            "classification": result.category,
        }

    async def _handle_data_collection_stream(
        self, session: Dict[str, Any], message: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream data collection phase with action tag detection."""
        data_path = session.get("data_path")
        classification = session.get("classification", "")

        if data_path:
            state_summary = "Data has been uploaded and validated. Training can be started."
        else:
            state_summary = "No training data uploaded yet."

        extra = (
            f"Current state: {state_summary}\n"
            f"Problem classification: {classification or 'unknown'}\n\n"
            "You can include ACTION TAGS in your response to trigger system actions:\n"
            "  [ACTION:GENERATE_DATA] — generate training examples based on what the user described\n"
            "  [ACTION:START_TRAINING] — begin training (only if data is available)\n\n"
            "Decision guidelines:\n"
            "- If the user asks you to generate/create/make data or examples, include [ACTION:GENERATE_DATA]\n"
            "- If the user says to start/begin training and data is available, include [ACTION:START_TRAINING]\n"
            "- If data was just uploaded or generated and the user seems ready, include [ACTION:START_TRAINING]\n"
            "- If you need more information, just ask — don't include any action tag\n"
            "- Action tags are stripped from the displayed message, so write naturally around them"
        )

        # Collect full LLM response to detect actions
        full_text = ""
        text_chunks: list[str] = []

        async for token in self._llm_stream_or_fallback(
            [{"role": "user", "content": message}],
            self._data_collection_fallback(data_path, message),
            state="data_collection",
            extra_context=extra,
        ):
            full_text += token
            text_chunks.append(token)

        clean_text, actions = self._parse_stream_for_actions(full_text)

        # Stream the clean text (without action tags) as deltas
        # Re-emit the text, stripping any action tags from chunks
        clean_remaining = clean_text
        for chunk in text_chunks:
            # Find how much of clean_remaining starts with this chunk (minus tags)
            chunk_clean = _ACTION_RE.sub("", chunk)
            if chunk_clean:
                yield {"type": "text_delta", "content": chunk_clean}

        # Handle GENERATE_DATA action
        if "GENERATE_DATA" in actions and not data_path:
            async for event in self._generate_data_with_progress(session, message):
                yield event
            return

        # Handle START_TRAINING action
        if "START_TRAINING" in actions and data_path:
            yield {"type": "done", "state": ChatState.TRAINING.value}
            return

        yield {"type": "done", "state": ChatState.DATA_COLLECTION.value}

    async def _generate_data_with_progress(
        self, session: Dict[str, Any], message: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate training data with progress events."""
        session_id = session.get("id", "unknown")
        classification = session.get("classification")

        yield {"type": "tool_start", "name": "generate_data", "description": "Generating Training Data"}
        yield {"type": "tool_progress", "name": "generate_data", "message": "Calling LLM to generate grid pairs..."}

        grid_pairs, error = await generate_training_data(
            problem_description=message,
            classification=classification,
            num_examples=10,
        )

        if error or len(grid_pairs) == 0:
            yield {
                "type": "tool_error",
                "name": "generate_data",
                "message": f"Failed: {error or 'no examples produced'}",
            }
            yield {"type": "text_delta", "content": f"Couldn't generate data: {error or 'no examples produced'}. Try describing your problem differently."}
            yield {"type": "done", "state": ChatState.DATA_COLLECTION.value}
            return

        yield {"type": "tool_progress", "name": "generate_data", "message": f"Generated {len(grid_pairs)} examples, converting to numpy format..."}

        upload_dir = os.path.abspath(
            os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "uploads"))
        )
        output_dir = os.path.join(upload_dir, session_id)
        os.makedirs(output_dir, exist_ok=True)

        try:
            result = convert_data(grid_pairs, output_dir=output_dir)
        except Exception as e:
            yield {"type": "tool_error", "name": "generate_data", "message": f"Conversion failed: {e}"}
            yield {"type": "text_delta", "content": f"Generated data but conversion failed: {e}"}
            yield {"type": "done", "state": ChatState.DATA_COLLECTION.value}
            return

        yield {"type": "tool_progress", "name": "generate_data", "message": f"Saved {result.num_examples} examples to disk"}
        yield {"type": "tool_done", "name": "generate_data", "result": f"{result.num_examples} examples ready"}

        # Store data_path in Redis
        await self.redis.hset(f"session:{session_id}:state", "data_path", output_dir)

        # Stream follow-up LLM response
        fallback = (
            f"Generated {result.num_examples} training examples. "
            "The data looks good — want me to start training, or would you like to tweak anything first?"
        )
        full_followup = ""
        async for token in self._llm_stream_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="data_collection",
            extra_context=(
                f"Successfully generated {result.num_examples} grid-pair examples and saved them. "
                f"Data is now ready for training. Tell the user and ask if they want to start training."
            ),
        ):
            full_followup += token

        clean_followup, followup_actions = self._parse_stream_for_actions(full_followup)
        if clean_followup:
            yield {"type": "text_delta", "content": clean_followup}

        if "START_TRAINING" in followup_actions:
            yield {"type": "done", "state": ChatState.TRAINING.value}
        else:
            yield {"type": "done", "state": ChatState.DATA_COLLECTION.value}

    async def _handle_training_stream(
        self, session: Dict[str, Any], message: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream training status check."""
        job_id = session.get("job_id")

        if job_id:
            status = await self.redis.hget(f"trm:job:{job_id}", "status")
            if status:
                status_str = status.decode() if isinstance(status, bytes) else status

                if status_str == "completed":
                    yield {"type": "text_delta", "content": "Training has completed successfully! Your model is ready."}
                    yield {"type": "done", "state": ChatState.COMPLETED.value}
                    return
                elif status_str == "failed":
                    error = await self.redis.hget(f"trm:job:{job_id}", "error")
                    error_str = error.decode() if isinstance(error, bytes) else (error or "Unknown error")
                    yield {"type": "text_delta", "content": f"Training failed: {error_str}"}
                    yield {"type": "done", "state": ChatState.COMPLETED.value}
                    return
                elif status_str == "cancelled":
                    yield {"type": "text_delta", "content": "Training was cancelled."}
                    yield {"type": "done", "state": ChatState.COMPLETED.value}
                    return
                else:
                    yield {"type": "text_delta", "content": f"Training is in progress (status: {status_str}). Please wait..."}
                    yield {"type": "done", "state": ChatState.TRAINING.value}
                    return

        yield {"type": "text_delta", "content": "Training is being set up. Please wait..."}
        yield {"type": "done", "state": ChatState.TRAINING.value}

    async def _handle_completed_stream(self, message: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream completed state response."""
        fallback = (
            "This session's training is complete. "
            "You can start a new session to train another model."
        )
        async for token in self._llm_stream_or_fallback(
            [{"role": "user", "content": message}],
            fallback,
            state="completed",
            extra_context="Training is finished. The user can start a new session if they want to train another model.",
        ):
            yield {"type": "text_delta", "content": token}
        yield {"type": "done", "state": ChatState.COMPLETED.value}
