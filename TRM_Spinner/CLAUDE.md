# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRM Spinner is a full-stack platform for training Tiny Recursive Models (TRM) — 7M parameter neural networks with recursive reasoning and Adaptive Computation Time. Users interact through a chat UI that guides them from problem description through data generation to model training and download.

**Three services (Docker Compose):**
- **Frontend** — Next.js 15 + React 19, port 3100 (maps to 3000 internal)
- **Worker** — FastAPI + PyTorch on GPU, port 8000
- **Redis** — Job queue and transient state, port 6380 (maps to 6379 internal)

## Commands

```bash
# Full stack (requires NVIDIA GPU + Docker)
make dev              # prebuild TRM sources + docker compose up --build
make down             # stop all services
make logs             # stream logs from all services

# Backend tests (run from host, not in container)
cd worker && python -m pytest __tests__/ -v --tb=short

# Single test file
cd worker && python -m pytest __tests__/test_chat_flow.py -v

# Single test
cd worker && python -m pytest __tests__/test_chat_flow.py::TestChatFlowStateMachine::test_greeting_to_classification -v

# Frontend
cd frontend && npx tsc --noEmit    # type check (always run before committing)
cd frontend && npm test            # Jest tests
cd frontend && npm run dev         # dev server on localhost:3000

# Both test suites
make test
```

## Architecture

### Chat State Machine

The core flow is a state machine in `worker/services/chat_engine.py`:

```
GREETING → CLASSIFICATION → DATA_COLLECTION → TRAINING → COMPLETED
```

Each state has both a synchronous handler (`_handle_*` → `ChatResponse`) and a streaming handler (`_handle_*_stream` → yields SSE event dicts). The streaming path is used by `POST /api/chat/stream`; the sync path by `POST /api/chat`.

The LLM can emit `[ACTION:GENERATE_DATA]` or `[ACTION:START_TRAINING]` tags in its response during `DATA_COLLECTION`. These are stripped from displayed text and trigger system actions (data generation, job queuing).

### Data Storage Split

**Appwrite (persistent):** sessions, messages, training jobs — all user-facing data with ownership checks.

**Redis (transient/fast):**
- `session:{id}:state` hash — classification, data_path, job_id (per-session working state)
- `trm:jobs:queue` — FIFO job queue (LPUSH/BRPOP)
- `trm:jobs:{id}` hash — job metadata, status, checkpoint_path
- `trm:jobs:{id}:metrics` — Pub/Sub channel for real-time training metrics
- `trm:jobs:{id}:cancel` — cancellation flag

### Authentication

`worker/api/middleware/auth.py` — `verify_token()` FastAPI dependency:
1. `X-API-Key` header matching `settings.admin_password` → returns `"test-user-001"` (test/admin bypass)
2. Bearer JWT → verified against Appwrite `/account` endpoint → returns user `$id`

Tests use the API key bypass via the `auth_headers` fixture.

### Streaming (SSE)

Two SSE patterns exist:
- **Chat streaming** (`POST /api/chat/stream`): Backend yields `text_delta`, `tool_start`, `tool_progress`, `tool_done`, `done` events. Frontend uses `fetchSSE()` (POST-based, `ReadableStream.getReader()`).
- **Training metrics** (`GET /api/jobs/{id}/stream`): Backend subscribes to Redis Pub/Sub. Frontend uses `EventSource` via `createSSEConnection()`.

### Frontend API Layer

`frontend/lib/api.ts` provides:
- `fetchAPI<T>()` — JSON requests with JWT auth
- `fetchSSE()` — POST-based SSE for chat streaming
- `fetchFormData<T>()` — multipart uploads (no Content-Type header — browser sets boundary)
- `fetchBlob()` — file downloads
- `createSSEConnection()` — `EventSource` wrapper for GET-based SSE

### TRM Source Files

The `worker/trm/` directory is **not checked in** — it's populated by `make prebuild` which copies model code from sibling repos (`../RR_TRM/`, `../CGAR_TRM/`). Never edit files in `worker/trm/` directly; edit the source repos instead.

## Testing Patterns

- All tests in `worker/__tests__/`, using `pytest-asyncio`
- External services (Redis, Appwrite, OpenAI) are fully mocked — see `conftest.py` fixtures
- LLM calls are disabled in chat tests via `patch("services.chat_engine.settings")` with empty API key, forcing fallback responses
- HTTP endpoint tests use `httpx.AsyncClient` with `ASGITransport` (no real server)
- Test deps (`pytest`, `pytest-asyncio`, `httpx`) are dev-only — install manually if running in Docker

## Theme

Gator color palette used across all frontend components:
- `gator-50` through `gator-700` (green tones, #f0f7f0 to #0f2518)
- `cream` (#f5f0e8), `amber-accent` (#d4a574)
- Fonts: Caveat (headings), Patrick Hand (body)
- Base font size: 24px

## GPU Requirements

RTX 5090 / Blackwell (sm_120) requires CUDA 12.8+ and PyTorch with cu128. The worker Dockerfile uses `nvidia/cuda:12.8.1-devel-ubuntu24.04` base image.
