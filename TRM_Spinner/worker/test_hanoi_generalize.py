"""Generalization test: Train on n=2..6, evaluate on held-out n=7.

Uses the data_converter directly to create train and test splits
with a shared vocab (covering all disk values up to 7).
"""
import json
import os
import subprocess
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from services.data_converter import convert_data, PAD_ID, EOS_ID, CELL_OFFSET


def solve_hanoi(n, source=0, target=2, auxiliary=1):
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
    grid = []
    for row in range(height):
        r = []
        for peg in pegs:
            r.append(peg[row] if row < len(peg) else 0)
        grid.append(r)
    return list(reversed(grid))


def pad_grid(grid, target_rows, target_cols):
    result = []
    for r in range(target_rows):
        if r < len(grid):
            row = list(grid[r]) + [0] * (target_cols - len(grid[r]))
        else:
            row = [0] * target_cols
        result.append(row[:target_cols])
    return result


def generate_pairs(n_range, max_height):
    pairs = []
    for n in n_range:
        states = solve_hanoi(n)
        for i in range(len(states) - 1):
            inp = pad_grid(states[i], max_height, 3)
            out = pad_grid(states[i + 1], max_height, 3)
            pairs.append({"input": inp, "output": out})
    return pairs


def main():
    import numpy as np

    MAX_HEIGHT = 7
    WORK_DIR = "/tmp/trm_hanoi_generalize"

    # Clean previous run
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    train_pairs = generate_pairs(range(2, 7), MAX_HEIGHT)  # n=2..6
    test_pairs = generate_pairs([7], MAX_HEIGHT)            # n=7
    all_pairs = train_pairs + test_pairs                    # for vocab calculation

    print("=" * 60)
    print("TRM — TOWERS OF HANOI GENERALIZATION TEST")
    print("=" * 60)
    print(f"\n  Train: n=2..6 → {len(train_pairs)} pairs")
    print(f"  Test:  n=7    → {len(test_pairs)} pairs (UNSEEN)")

    # Step 1: Convert ALL data to get correct vocab/seq_len
    all_result = convert_data(all_pairs, WORK_DIR, split="all_temp")

    print(f"  Vocab: {all_result.vocab_size}, Seq len: {all_result.seq_len}")

    # Step 2: Now create proper train split with same dimensions
    # We need to manually create numpy arrays with the SAME vocab_size/seq_len
    # but only containing train data
    from services.data_converter import _encode_pair

    seq_len = all_result.seq_len
    vocab_size = all_result.vocab_size

    # Create train split
    train_dir = os.path.join(WORK_DIR, "train")
    os.makedirs(train_dir, exist_ok=True)

    n_train = len(train_pairs)
    train_inputs = np.zeros((n_train, seq_len), dtype=np.int32)
    train_labels = np.zeros((n_train, seq_len), dtype=np.int32)

    for i, pair in enumerate(train_pairs):
        tokens, labels = _encode_pair(pair["input"], pair["output"], seq_len)
        train_inputs[i] = tokens
        train_labels[i] = labels

    np.save(os.path.join(train_dir, "train__inputs.npy"), train_inputs)
    np.save(os.path.join(train_dir, "train__labels.npy"), train_labels)
    np.save(os.path.join(train_dir, "train__puzzle_identifiers.npy"),
            np.arange(n_train, dtype=np.int32))
    np.save(os.path.join(train_dir, "train__puzzle_indices.npy"),
            np.arange(n_train + 1, dtype=np.int32))
    np.save(os.path.join(train_dir, "train__group_indices.npy"),
            np.array([0, n_train], dtype=np.int32))

    train_meta = {
        "pad_id": PAD_ID,
        "ignore_label_id": -1,
        "blank_identifier_id": 0,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_puzzle_identifiers": n_train,
        "total_groups": 1,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": n_train,
        "sets": ["train"],
    }
    with open(os.path.join(train_dir, "dataset.json"), "w") as f:
        json.dump(train_meta, f, indent=2)

    # Save test pairs as JSON for eval script
    test_json_path = os.path.join(WORK_DIR, "test_pairs.json")
    with open(test_json_path, "w") as f:
        json.dump(test_pairs, f)

    # Clean temp
    shutil.rmtree(os.path.join(WORK_DIR, "all_temp"), ignore_errors=True)

    # Step 3: Train
    print(f"\n  Training: 100k epochs, batch=64, lr=3e-4")
    trm_dir = os.path.join(os.path.dirname(__file__), "trm")
    script = os.path.join(trm_dir, "pretrain_web.py")
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
    job_id = "hanoi-gen-test"

    cli_args = [
        f"+job_id={job_id}",
        "+redis_url=redis://localhost:6379",
        f"data_paths=[{WORK_DIR}]",
        "arch=trm",
        "global_batch_size=64",
        "epochs=100000",
        "eval_interval=100000",
        "lr=3e-4",
        "lr_min_ratio=0.1",
        "lr_warmup_steps=50",
        "weight_decay=0.01",
        "beta1=0.9",
        "beta2=0.95",
        "puzzle_emb_lr=1e-2",
        "puzzle_emb_weight_decay=0.1",
        "evaluators=[]",
        f"+checkpoint_path={WORK_DIR}/checkpoints",
    ]

    start = time.time()
    proc = subprocess.Popen(
        [venv_python, script] + cli_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=trm_dir,
        env={**os.environ, "DISABLE_COMPILE": "1"},
    )

    import select
    while proc.poll() is None:
        if select.select([proc.stdout], [], [], 0.5)[0]:
            line = proc.stdout.readline()
            if line:
                text = line.decode(errors="replace").rstrip()
                if text and ("it/s" in text or "SAVE" in text or "Epoch" in text
                             or "TRAIN" in text or "num_params" in text.lower()):
                    print(f"  | {text[:140]}")

    remaining = proc.stdout.read().decode(errors="replace")
    for line in remaining.strip().split("\n"):
        if line.strip() and ("it/s" in line or "SAVE" in line):
            print(f"  | {line[:140]}")

    elapsed = time.time() - start
    print(f"\n  Training done: {elapsed:.1f}s, exit={proc.returncode}")

    # Training metrics
    import redis as redis_sync
    rc = redis_sync.Redis.from_url("redis://localhost:6379", decode_responses=True)
    latest = rc.hgetall(f"trm:jobs:{job_id}:latest")
    print(f"  Train loss:     {latest.get('train/lm_loss', '?')}")
    print(f"  Train accuracy: {latest.get('train/accuracy', '?')}")
    print(f"  Train exact:    {latest.get('train/exact_accuracy', '?')}")
    print(f"  ACT steps:      {latest.get('train/steps', '?')}")

    if proc.returncode != 0:
        print("  TRAINING FAILED")
        return

    # Step 4: Evaluate on held-out n=7
    print(f"\n  --- Evaluating on unseen n=7 ({len(test_pairs)} pairs) ---")
    eval_result = subprocess.run(
        [
            venv_python, "eval_hanoi.py",
            "--ckpt-dir", os.path.join(WORK_DIR, "checkpoints"),
            "--test-json", test_json_path,
            "--data-dir", WORK_DIR,
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
        env={**os.environ, "DISABLE_COMPILE": "1"},
    )

    print(eval_result.stdout)
    if eval_result.returncode != 0:
        stderr_lines = eval_result.stderr.strip().split("\n")
        # Show last meaningful errors
        for line in stderr_lines[-15:]:
            if line.strip():
                print(f"  ERR: {line[:140]}")

    print("=" * 60)


if __name__ == "__main__":
    main()
