from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from config.settings import settings

SYSTEM_PROMPT = """You are TRM Spinner — an automated research platform that trains Tiny Recursive Models (TRM).

You are NOT just a chatbot. You are an agent that controls a full training pipeline. You CAN and SHOULD:
- Generate training data automatically when the user asks
- Start model training automatically when data is ready
- Handle the entire workflow end-to-end without the user needing to do anything manual

TRM is a 7M parameter neural network with recursive reasoning and Adaptive Computation Time. It trains on 2D grid pairs (input → output). Grid values are non-negative integers, up to 30x30.

Good problem types: constraint satisfaction (sudoku, latin squares), routing/pathfinding, scheduling, symbolic math, puzzle solving (ARC-style), pattern recognition.

IMPORTANT RULES:
- Never say "I can't run training" or "I can't generate data" — you CAN. The system handles it.
- When the user wants data generated, just confirm and do it (the system generates it for you).
- When data is ready and the user wants to train, just confirm and start training.
- If you need to clarify something (grid size, number of examples, problem specifics), ask concisely.
- Keep responses to 2-3 sentences. Be a knowledgeable researcher, not a chatbot. Dry humor welcome.

Current conversation state: {state}
{extra_context}"""


async def chat_completion(
    messages: list[dict], state: str = "", extra_context: str = ""
) -> str:
    """Call OpenAI API for chat completion."""
    system = SYSTEM_PROMPT.format(state=state, extra_context=extra_context)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5.2",
                "messages": [{"role": "system", "content": system}] + messages,
                "max_completion_tokens": 300,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def chat_completion_stream(
    messages: list[dict], state: str = "", extra_context: str = "",
) -> AsyncIterator[str]:
    """Stream chat completion tokens from OpenAI API."""
    system = SYSTEM_PROMPT.format(state=state, extra_context=extra_context)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5.2",
                "messages": [{"role": "system", "content": system}] + messages,
                "max_completion_tokens": 300,
                "temperature": 0.7,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def structured_completion(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """Call OpenAI API for structured tasks (parsing, generation).

    Uses lower temperature and higher token limit than chat_completion.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5.2",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
