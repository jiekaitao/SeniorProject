"""Full pipeline test: Towers of Hanoi.

Simulates a real user who wants to train TRM on the Towers of Hanoi problem.
Towers of Hanoi represented as grids:
- 3 columns (pegs), N rows (max disk slots)
- Cell values = disk sizes (0 = empty)
- Input = initial state, Output = next optimal move

For 3 disks on 3 pegs with height 3:
  Input:  [[3,0,0],    Output: [[0,0,0],
           [2,0,0],             [3,0,0],
           [1,0,0]]             [2,0,1]]
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import httpx

API = "http://localhost:8099"
HEADERS = {"X-API-Key": "Jack123123@!", "Content-Type": "application/json"}
TIMEOUT = 30.0


def solve_hanoi(n, source=0, target=2, auxiliary=1):
    """Generate all states for n-disk Towers of Hanoi."""
    # Initial state: all disks on peg 0
    pegs = [list(range(n, 0, -1)), [], []]
    states = [_pegs_to_grid(pegs, n)]

    def _move(num_disks, src, tgt, aux):
        if num_disks == 0:
            return
        _move(num_disks - 1, src, aux, tgt)
        disk = pegs[src].pop()
        pegs[tgt].append(disk)
        states.append(_pegs_to_grid(pegs, n))
        _move(num_disks - 1, aux, tgt, src)

    _move(n, source, target, auxiliary)
    return states


def _pegs_to_grid(pegs, height):
    """Convert peg lists to a 2D grid (height x 3). Bottom row = index 0."""
    grid = []
    for row in range(height):
        r = []
        for peg in pegs:
            r.append(peg[row] if row < len(peg) else 0)
        grid.append(r)
    # Reverse so top of tower is row 0 (visual convention)
    return list(reversed(grid))


def generate_hanoi_training_data():
    """Generate input/output pairs for Towers of Hanoi.

    We generate data for 2, 3, and 4 disk problems.
    Each pair: input = state_i, output = state_{i+1} (optimal move).
    """
    pairs = []

    max_disks = 8
    for n_disks in range(2, max_disks + 1):
        states = solve_hanoi(n_disks)

        for i in range(len(states) - 1):
            # Pad to consistent height (max_disks rows for the largest case)
            inp = pad_grid(states[i], max_disks, 3)
            out = pad_grid(states[i + 1], max_disks, 3)
            pairs.append({"input": inp, "output": out})

    return pairs


def pad_grid(grid, target_rows, target_cols):
    """Pad grid to target dimensions with zeros."""
    result = []
    for r in range(target_rows):
        if r < len(grid):
            row = list(grid[r]) + [0] * (target_cols - len(grid[r]))
        else:
            row = [0] * target_cols
        result.append(row[:target_cols])
    return result


async def main():
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        print("=" * 60)
        print("TRM SPINNER — TOWERS OF HANOI PIPELINE TEST")
        print("=" * 60)

        # 1. Health check
        print("\n[1/8] Health check...")
        r = await c.get(f"{API}/api/health")
        health = r.json()
        print(f"  GPU: {health['gpu']}, Redis: {health['redis']}")

        # 2. Create session
        print("\n[2/8] Creating session...")
        r = await c.post(
            f"{API}/api/sessions",
            json={"user_id": "test-user-001"},
            headers=HEADERS,
        )
        assert r.status_code == 200, f"Failed: {r.text}"
        session = r.json()
        session_id = session["id"]
        print(f"  Session: {session_id}")

        # 3. Chat: describe the problem
        print("\n[3/8] Chatting — describing Towers of Hanoi...")
        r = await c.post(
            f"{API}/api/chat",
            json={
                "session_id": session_id,
                "message": "I want to train a model to solve the Towers of Hanoi puzzle. "
                "Given a grid showing the current state of disks on 3 pegs, "
                "predict the next optimal move. Each state is a 2D grid where "
                "columns are pegs and values are disk sizes.",
            },
            headers=HEADERS,
        )
        assert r.status_code == 200, f"Chat failed: {r.text}"
        chat = r.json()
        print(f"  GPT-5.2: {chat['message']}")
        print(f"  State: {chat['state']}")

        # 4. Chat: more detail for classification
        print("\n[4/8] Chatting — classification...")
        r = await c.post(
            f"{API}/api/chat",
            json={
                "session_id": session_id,
                "message": "It's essentially a constraint satisfaction and planning puzzle. "
                "The model needs to learn the rules: only move one disk at a time, "
                "never place a larger disk on a smaller one. The grid pattern "
                "transforms follow strict logical constraints. I think it maps "
                "well to grid reasoning and pattern recognition.",
            },
            headers=HEADERS,
        )
        assert r.status_code == 200, f"Chat failed: {r.text}"
        chat = r.json()
        print(f"  GPT-5.2: {chat['message']}")
        print(f"  State: {chat['state']}, Classification: {chat.get('classification', 'none')}")

        # 5. Generate and upload training data
        print("\n[5/8] Generating Towers of Hanoi training data...")
        training_data = generate_hanoi_training_data()
        print(f"  Generated {len(training_data)} input/output pairs")
        print(f"  Disk sizes: n=2..8 (3 to 255 moves each)")
        print(f"  Grid size: 8x3 (8 rows x 3 pegs)")
        print(f"  Example pair:")
        print(f"    Input:  {training_data[3]['input']}")
        print(f"    Output: {training_data[3]['output']}")

        r = await c.post(
            f"{API}/api/data/upload",
            json={"session_id": session_id, "data": training_data},
            headers=HEADERS,
        )
        assert r.status_code == 200, f"Upload failed: {r.text}"
        upload = r.json()
        print(f"  Valid: {upload['valid']}, Examples: {upload['num_examples']}, Vocab: {upload['vocab_size']}")

        if not upload["valid"]:
            print(f"  ERRORS: {upload.get('errors')}")
            return

        # Get data_path
        r = await c.get(f"{API}/api/sessions/{session_id}", headers=HEADERS)
        sess = r.json()
        data_path = sess.get("data_path")
        print(f"  Data path: {data_path}")

        if not data_path:
            print("  ERROR: data_path not set!")
            return

        # 6. Create training job
        print("\n[6/8] Creating training job...")
        job_config = {
            "arch": "trm",
            "data_paths": [data_path],
            "global_batch_size": 64,
            "epochs": 50000,
            "lr": 3e-4,
            "lr_min_ratio": 0.1,
            "lr_warmup_steps": 50,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "puzzle_emb_lr": 1e-2,
            "puzzle_emb_weight_decay": 0.1,
            "eval_interval": 50000,
            "evaluators": [],
        }
        r = await c.post(
            f"{API}/api/jobs",
            json={
                "session_id": session_id,
                "user_id": "test-user-001",
                "data_path": data_path,
                "config": job_config,
            },
            headers=HEADERS,
        )
        assert r.status_code == 200, f"Job creation failed: {r.text}"
        job = r.json()
        job_id = job["id"]
        print(f"  Job ID: {job_id}")

        # 7. Run training
        print("\n[7/8] Training TRM on Towers of Hanoi...")
        trm_dir = os.path.join(os.path.dirname(__file__), "trm")
        script = os.path.join(trm_dir, "pretrain_web.py")
        venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")

        cli_args = [
            f"+job_id={job_id}",
            "+redis_url=redis://localhost:6379",
            f"data_paths=[{data_path}]",
            "arch=trm",
            "global_batch_size=64",
            "epochs=50000",
            "eval_interval=50000",
            "lr=3e-4",
            "lr_min_ratio=0.1",
            "lr_warmup_steps=50",
            "weight_decay=0.01",
            "beta1=0.9",
            "beta2=0.95",
            "puzzle_emb_lr=1e-2",
            "puzzle_emb_weight_decay=0.1",
            "evaluators=[]",
            f"+checkpoint_path={data_path}/checkpoints",
        ]

        cmd = [venv_python, script] + cli_args
        print(f"  Config: {len(training_data)} examples, 50k epochs, batch_size=64, lr=3e-4")

        start_time = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=trm_dir,
            env={**os.environ, "DISABLE_COMPILE": "1"},
        )

        print("\n  --- Training Output ---")
        import select

        while proc.poll() is None:
            if select.select([proc.stdout], [], [], 0.5)[0]:
                line = proc.stdout.readline()
                if line:
                    text = line.decode(errors="replace").rstrip()
                    if text:
                        print(f"  | {text[:140]}")

        remaining = proc.stdout.read().decode(errors="replace")
        for line in remaining.strip().split("\n"):
            if line.strip():
                print(f"  | {line[:140]}")

        elapsed = time.time() - start_time
        print(f"  --- End Training ---")
        print(f"  Return code: {proc.returncode}, Time: {elapsed:.1f}s")

        # 8. Check results
        print("\n[8/8] Results...")
        import redis as redis_sync

        r_client = redis_sync.Redis.from_url(
            "redis://localhost:6379", decode_responses=True
        )
        latest = r_client.hgetall(f"trm:jobs:{job_id}:latest")
        print(f"  Status: {latest.get('status', '?')}")
        print(f"  Steps: {latest.get('step', '?')}/{latest.get('total_steps', '?')}")
        print(f"  Loss: {latest.get('train/lm_loss', '?')}")
        print(f"  Accuracy: {latest.get('train/accuracy', '?')}")
        print(f"  Exact accuracy: {latest.get('train/exact_accuracy', '?')}")
        print(f"  Halt steps: {latest.get('train/steps', '?')}")

        print("\n" + "=" * 60)
        if proc.returncode == 0 and latest.get("status") == "completed":
            acc = float(latest.get("train/accuracy", 0))
            print(f"TOWERS OF HANOI TEST: {'PASSED' if acc > 0.8 else 'PARTIAL'} (accuracy={acc:.1%})")
        else:
            print("TOWERS OF HANOI TEST: FAILED")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
