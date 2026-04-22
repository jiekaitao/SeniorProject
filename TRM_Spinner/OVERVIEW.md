# TRM_Spinner

The product-y part of the project. A full-stack web app that takes the TRM training code from `../RR_TRM/` and `../CGAR_TRM/` and wraps it in a chat interface so a user without a PhD in machine learning can train a reasoning model on their own problem.

The user flow is five states:

1. **Greeting.** User shows up, says hi, describes what they want to do.
2. **Classification.** The backend figures out what kind of reasoning problem they're describing. Sudoku-like, maze-like, pattern completion, something else.
3. **Data collection.** User uploads training data (CSV or JSON) or asks the system to generate synthetic data based on their description.
4. **Training.** A job goes into the Redis queue. The worker picks it up, runs real TRM training on a GPU, streams metrics back to the frontend via SSE.
5. **Completed.** User downloads the trained model.

All the training code is real. No simulations, no mocks. The `make prebuild` step copies the actual TRM model files and training loop from the sibling research folders into `worker/trm/`, and the web worker runs them as subprocess jobs.

## Top-level files

```
docker-compose.yml   # three services: frontend:3100, worker:8000, redis:6380
Makefile             # make prebuild, make dev, make test
.env.example         # Appwrite, Redis, API key, OpenAI template
.env                 # real secrets, gitignored
CLAUDE.md            # architecture notes for anyone extending this
.gitignore
```

`make dev` is the one command you'll use most. It runs `make prebuild` first (copies TRM files across), then `docker compose up --build` to spin up the three services with GPU allocation for the worker.

Ports:

- **Frontend**: 3100 externally, 3000 internally.
- **Worker**: 8000 with a /health endpoint that Docker uses for its healthcheck every 15s.
- **Redis**: 6380 externally, 6379 internally.

## Subfolders

### `frontend/`
Next.js 15 with React 19 and TypeScript. App Router, not pages directory.

- `app/` — route handlers. Chat page, auth pages (login, signup), dev tools, a global layout and CSS.
- `components/` — React components grouped by area. `chat/` has the `ChatContainer`, `MessageBubble`, `ChatInput`, `DataUpload`, and `ToolCallBox` components that make up the main UI. `training/` has the job monitor views. `auth/` has login and signup forms. `ui/` has the shared design system primitives (buttons, inputs, modals).
- `lib/` — utilities. `api.ts` is the frontend HTTP layer. It handles JWT auth against Appwrite, streams SSE from the chat endpoint, and does file uploads. `appwrite.ts` wraps the Appwrite SDK.
- `hooks/` — custom React hooks. `useAuth` handles Appwrite session state. `useTrainingProgress` subscribes to the training metrics stream.

### `worker/`
FastAPI + PyTorch backend. This is where the work happens.

- `api/routes/` — the HTTP endpoints. `health.py` for Docker healthchecks. `chat.py` is the SSE streaming endpoint (`POST /api/chat/stream`). `sessions.py` for CRUD on user sessions. `jobs.py` for queuing training jobs and streaming metrics (`GET /api/jobs/{id}/stream`). `data.py` for upload and format conversion. `admin.py` for test and debug utilities.
- `api/middleware/` — `auth.py` does JWT validation. It has a dual mode: if the request has an `X-API-Key` header matching the admin password, it returns a fake test user ID. Otherwise it validates the Bearer token against Appwrite's `/account` endpoint and returns the real user ID. This lets the test suite run without needing Appwrite credentials.
- `services/` — business logic.
  - `chat_engine.py` implements the 5-state machine described above.
  - `classifier.py` detects what kind of problem a user is describing.
  - `data_generator.py` creates synthetic training data when the user doesn't upload their own.
  - `training_runner.py` orchestrates the training subprocess, captures logs, pipes metrics to Redis.
  - `training_manager.py` manages the Redis job queue (LPUSH to enqueue, BRPOP in the worker to dequeue).
  - `appwrite_db.py` is the persistence layer. Has a LocalDB fallback implementation backed by Redis so you can run the system offline without an Appwrite project set up.
- `schemas/` — Pydantic models. Chat request schemas, job definitions, Appwrite document shapes.
- `config/` — `settings.py` reads environment variables (Appwrite URL, project ID, API key, Redis URL, admin password, OpenAI API key for the classifier).
- `trm/` — this directory is auto-populated by `make prebuild`. It copies from `../RR_TRM/` and `../CGAR_TRM/`: models, evaluators, dataset builders, configs, utilities. Gitignored so you don't accidentally commit stale copies.
- `__tests__/` — pytest-asyncio test suite. Mocks Appwrite and Redis where needed.

### `checkpoints/`
Where trained models land. Mounted from the host so they persist across container restarts.

### `redis-data/`
Redis RDB snapshots. Also mounted. Keeps job queue state and session data across restarts.

### `uploads/`
User-uploaded training data. Parsed and converted into the format the training code expects.

### `scripts/`
Just one file for now: `init_appwrite.py`. Run this once after setting up Appwrite to create the required collections with the right schemas.

## Getting it running locally

You need:

- Docker with the nvidia-container-toolkit (for GPU passthrough).
- An NVIDIA GPU. The training code expects an RTX 5090 with sm_120 and CUDA 12.8. It'll technically run on older GPUs if you rebuild the CUDA kernels, but we haven't tested that.
- An Appwrite project. Cloud is fine. Set up the collections with `scripts/init_appwrite.py`.
- Optional: an OpenAI API key for the classifier service. If you don't provide one, the classifier falls back to a simpler keyword-based approach.

Copy `.env.example` to `.env` and fill in:

```
APPWRITE_URL=...
APPWRITE_PROJECT_ID=...
APPWRITE_API_KEY=...
ADMIN_PASSWORD=...
REDIS_URL=redis://redis:6379
OPENAI_API_KEY=...   # optional
```

Then:

```bash
make prebuild   # copies TRM source files from sibling repos
make dev        # docker compose up --build
```

Frontend at `http://localhost:3100`. Worker at `http://localhost:8000/health` to sanity-check.

## Running tests

```bash
make test
```

The tests use the X-API-Key bypass so they don't need real Appwrite credentials. They also use the LocalDB fallback in `appwrite_db.py` so they run without network access.

## Architecture notes

Two SSE patterns coexist in this codebase, which is a little unusual:

- **Chat uses POST-based streaming.** The client calls `fetchSSE()` which POSTs the request and reads from the response body stream. This is because the chat request carries a meaningful payload (user messages, session state) that doesn't fit cleanly in a GET URL.
- **Training metrics use GET-based EventSource.** This is the standard browser SSE API. Used because it supports auto-reconnect and the metrics subscription is long-lived.

Redis does triple duty: job queue (LPUSH from API routes, BRPOP in the worker's queue consumer), active session cache, and pub/sub for metric streaming. Appwrite stores the things that matter long-term: user accounts, completed training jobs, message history.

The dual-mode auth is worth flagging. In tests and local dev, any request with an `X-API-Key: <admin_password>` header is treated as user `test-user-001`. In production, the worker expects a Bearer JWT that it validates against Appwrite. If you're adding a new endpoint, make sure it respects whichever mode is active.

## Relationship to the TRM research folders

The worker doesn't have its own copy of the TRM training code. It borrows from the research folders via `make prebuild`:

- From `../RR_TRM/`: models (`trm.py`, `trm_hier6.py`, `trm_singlez.py`), losses, dataset loaders, configs, utilities, evaluators.
- From `../CGAR_TRM/`: `trm_cgar.py` and `losses_cgar.py`.

The web-specific entry point is `pretrain_web.py` (not in the research folders, written specifically for the worker). It wraps the original Hydra-based training loop and redirects metrics to Redis instead of just WandB. This is how we got "real training" running through the web UI without forking the research code.

## Deployment posture

This isn't deployed anywhere public. It's a local-first research demo. The architecture (Appwrite, Redis, containerized services) could be pushed to a cloud provider, but there's no Kubernetes manifest, no Terraform, no CI, no cloud IAM config. If you want to deploy it, you're starting from scratch on the infra side.

That said, the app is more production-shaped than a research prototype usually is. Real auth, real persistence, real streaming, tests that actually pass. So it's not far from being shippable if someone wanted to.

## Things to know before extending

- Don't commit `.env`. It's gitignored, but double-check.
- The Appwrite schema is created by `scripts/init_appwrite.py`. If you add new fields, update that script and re-run it.
- The LocalDB fallback in `appwrite_db.py` is handy but it doesn't fully implement every Appwrite feature. If you're adding features that depend on Appwrite-specific capabilities (indexes, permissions, full-text search), the fallback will silently stub them.
- The chat state machine in `chat_engine.py` is strict about transitions. Don't add a new state without updating all three layers (the enum, the transition logic, the frontend view).
- GPU memory is tight on a 5090 when the worker is running full training. If you're also running RR_TRM or DeepPass experiments on the same box, you'll OOM. Schedule.
