"""Full end-to-end test: user flow through actual TRM training on GPU.

Simulates a real user:
1. Create session
2. Chat: greeting → classification (puzzle_solving)
3. Upload grid training data
4. Create training job with small config
5. Monitor job until completion/failure
6. Verify results
"""
import asyncio
import json
import sys
import time

import httpx

API = "http://localhost:8099"
HEADERS = {"X-API-Key": "Jack123123@!", "Content-Type": "application/json"}
TIMEOUT = 30.0


async def main():
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:

        print("=" * 60)
        print("TRM SPINNER — FULL TRAINING PIPELINE TEST")
        print("=" * 60)

        # 1. Health check
        print("\n[1/8] Health check...")
        r = await c.get(f"{API}/api/health")
        health = r.json()
        print(f"  Status: {health['status']}, GPU: {health['gpu']}, Redis: {health['redis']}")
        if health["redis"] != "connected":
            print("  ERROR: Redis not connected!")
            return

        # 2. Create session
        print("\n[2/8] Creating session...")
        r = await c.post(f"{API}/api/sessions", json={"user_id": "test-user-001"}, headers=HEADERS)
        assert r.status_code == 200, f"Failed to create session: {r.text}"
        session = r.json()
        session_id = session["id"]
        print(f"  Session ID: {session_id}")
        print(f"  State: {session['state']}")

        # 3. Chat: greeting
        print("\n[3/8] Chatting (greeting)...")
        r = await c.post(f"{API}/api/chat", json={
            "session_id": session_id,
            "message": "Hi, I want to train a model to solve grid transformation puzzles like ARC"
        }, headers=HEADERS)
        assert r.status_code == 200, f"Chat failed: {r.text}"
        chat = r.json()
        print(f"  Bot: {chat['message'][:100]}...")
        print(f"  State: {chat['state']}")

        # 4. Chat: classification
        print("\n[4/8] Chatting (classification)...")
        r = await c.post(f"{API}/api/chat", json={
            "session_id": session_id,
            "message": "Specifically, 2D grid puzzles where patterns in the input grid get transformed into output patterns. Like rotating, flipping, or applying rules to cells."
        }, headers=HEADERS)
        assert r.status_code == 200, f"Chat failed: {r.text}"
        chat = r.json()
        print(f"  Bot: {chat['message'][:100]}...")
        print(f"  State: {chat['state']}, Classification: {chat.get('classification', 'none')}")

        # 5. Upload training data — 10 simple grid transformation pairs
        print("\n[5/8] Uploading training data (10 grid pairs)...")
        training_data = []
        for i in range(10):
            # Simple pattern: invert the grid (0→1, 1→0)
            size = 3
            inp = [[(i + r + c) % 2 for c in range(size)] for r in range(size)]
            out = [[1 - v for v in row] for row in inp]
            training_data.append({"input": inp, "output": out})

        r = await c.post(f"{API}/api/data/upload", json={
            "session_id": session_id,
            "data": training_data,
        }, headers=HEADERS)
        assert r.status_code == 200, f"Upload failed: {r.text}"
        upload = r.json()
        print(f"  Valid: {upload['valid']}, Examples: {upload['num_examples']}, Vocab: {upload['vocab_size']}")

        # Get the data_path from Redis (stored by upload endpoint)
        r = await c.get(f"{API}/api/sessions/{session_id}", headers=HEADERS)
        sess = r.json()
        data_path = sess.get("data_path")
        print(f"  Data path: {data_path}")

        if not data_path:
            print("  ERROR: data_path not set in session!")
            return

        # 6. Create training job with minimal config
        print("\n[6/8] Creating training job (minimal config for quick test)...")
        job_config = {
            "arch": "trm",
            "data_paths": [f"{data_path}/train"],
            "global_batch_size": 10,
            "epochs": 100,
            "lr": 1e-3,
            "lr_min_ratio": 0.1,
            "lr_warmup_steps": 5,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "puzzle_emb_lr": 1e-2,
            "puzzle_emb_weight_decay": 0.1,
        }
        r = await c.post(f"{API}/api/jobs", json={
            "session_id": session_id,
            "user_id": "test-user-001",
            "data_path": data_path,
            "config": job_config,
        }, headers=HEADERS)
        assert r.status_code == 200, f"Job creation failed: {r.text}"
        job = r.json()
        job_id = job["id"]
        print(f"  Job ID: {job_id}")
        print(f"  Status: {job['status']}")

        # 7. Now we need to actually trigger training
        # The training runner needs to pick up the job from the queue
        # In production this would be a background worker loop
        # For testing, let's manually trigger it
        print("\n[7/8] Starting training (manual trigger)...")
        print("  Spawning training runner...")

        # Import and run the training directly
        import subprocess
        import os

        # The training runner spawns pretrain_web.py as a subprocess
        # Let's do that directly here
        trm_dir = os.path.join(os.path.dirname(__file__), "trm")
        script = os.path.join(trm_dir, "pretrain_web.py")

        # Build Hydra CLI args
        cli_args = [
            f"+job_id={job_id}",
            f"+redis_url=redis://localhost:6379",
            f"data_paths=[{data_path}]",
            "arch=trm",
            "global_batch_size=10",
            "epochs=100",
            "lr=1e-3",
            "lr_min_ratio=0.1",
            "lr_warmup_steps=5",
            "weight_decay=0.01",
            "beta1=0.9",
            "beta2=0.95",
            "puzzle_emb_lr=1e-2",
            "puzzle_emb_weight_decay=0.1",
            "evaluators=[]",
            "eval_interval=100",
            f"+checkpoint_path={data_path}/checkpoints",
        ]

        venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
        cmd = [venv_python, script] + cli_args

        print(f"  Command: {os.path.basename(venv_python)} pretrain_web.py ...")
        print(f"  Working dir: {trm_dir}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=trm_dir,
            env={**os.environ, "DISABLE_COMPILE": "1"},
        )

        # Monitor output and Redis for progress
        print("\n  --- Training Output ---")
        start_time = time.time()
        last_status_check = 0

        while proc.poll() is None:
            # Read output line by line (non-blocking)
            import select
            if select.select([proc.stdout], [], [], 0.5)[0]:
                line = proc.stdout.readline()
                if line:
                    text = line.decode(errors="replace").rstrip()
                    if text:
                        print(f"  | {text[:120]}")

            # Check Redis for metrics periodically
            elapsed = time.time() - start_time
            if elapsed - last_status_check > 5:
                last_status_check = elapsed
                r2 = await c.get(f"{API}/api/health")  # Just to keep connection alive

            if elapsed > 300:  # 5 minute timeout
                print("  TIMEOUT: Training exceeded 5 minutes")
                proc.terminate()
                break

        # Read remaining output
        remaining = proc.stdout.read().decode(errors="replace")
        for line in remaining.strip().split("\n"):
            if line.strip():
                print(f"  | {line[:120]}")

        return_code = proc.returncode
        elapsed = time.time() - start_time
        print(f"  --- End Training Output ---")
        print(f"  Return code: {return_code}")
        print(f"  Elapsed: {elapsed:.1f}s")

        # 8. Verify results
        print("\n[8/8] Verifying results...")

        # Check Redis for final job status
        import redis as redis_sync
        r_client = redis_sync.Redis.from_url("redis://localhost:6379", decode_responses=True)
        latest = r_client.hgetall(f"trm:jobs:{job_id}:latest")
        print(f"  Redis latest: {json.dumps(latest, indent=2)[:500]}")

        if return_code == 0:
            print("\n  TRAINING COMPLETED SUCCESSFULLY!")

            # Check for checkpoint
            import glob
            checkpoints = glob.glob(f"{data_path}/checkpoints/**/*", recursive=True)
            if checkpoints:
                print(f"  Checkpoints saved: {len(checkpoints)} files")
                for cp in checkpoints[:5]:
                    print(f"    - {os.path.basename(cp)}")
        else:
            print(f"\n  TRAINING FAILED (exit code {return_code})")

        # Check job status in Appwrite
        r = await c.get(f"{API}/api/jobs/{job_id}", headers=HEADERS)
        if r.status_code == 200:
            job_status = r.json()
            print(f"  Appwrite job status: {job_status.get('status', 'unknown')}")

        print("\n" + "=" * 60)
        if return_code == 0:
            print("FULL PIPELINE TEST: PASSED")
        else:
            print("FULL PIPELINE TEST: TRAINING FAILED")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
