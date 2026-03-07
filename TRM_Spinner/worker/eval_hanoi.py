"""Evaluate a trained TRM checkpoint on held-out Towers of Hanoi data."""
import argparse
import glob
import json
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trm"))

from omegaconf import OmegaConf
from utils.functions import load_model_class

PAD_ID = 0
EOS_ID = 1
CELL_OFFSET = 2
IGNORE_LABEL_ID = -100


def grid_to_tokens(grid):
    tokens = []
    for row in grid:
        for v in row:
            tokens.append(v + CELL_OFFSET)
    return tokens


def tokens_to_grid(tokens, rows, cols):
    grid = []
    for r in range(rows):
        grid.append([tokens[r * cols + c] - CELL_OFFSET for c in range(cols)])
    return grid


def make_batch(pairs, seq_len):
    """Convert test pairs into a batch dict matching PuzzleDataset output."""
    batch_inputs = []
    batch_labels = []

    for pair in pairs:
        inp_tokens = grid_to_tokens(pair["input"])
        out_tokens = grid_to_tokens(pair["output"])
        full_seq = inp_tokens + [EOS_ID] + out_tokens + [EOS_ID]

        # Labels: IGNORE for input portion + EOS, predict output + final EOS
        labels = [IGNORE_LABEL_ID] * (len(inp_tokens) + 1) + out_tokens + [EOS_ID]

        # Pad to seq_len
        inputs_padded = full_seq + [PAD_ID] * (seq_len - len(full_seq))
        labels_padded = labels + [IGNORE_LABEL_ID] * (seq_len - len(labels))

        batch_inputs.append(inputs_padded[:seq_len])
        batch_labels.append(labels_padded[:seq_len])

    return {
        "inputs": torch.tensor(batch_inputs, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "puzzle_identifiers": torch.zeros(len(pairs), dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--test-json", required=True, help="JSON file with held-out test pairs")
    parser.add_argument("--data-dir", help="Data dir with train/dataset.json")
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(args.ckpt_dir, "all_config.yaml"))

    data_dir = args.data_dir or config.data_paths[0]
    with open(os.path.join(data_dir, "train", "dataset.json")) as f:
        meta = json.load(f)

    # Build model
    arch_extra = {k: v for k, v in OmegaConf.to_container(config.arch).items()
                  if k not in ("name", "loss")}
    model_cfg = {
        **arch_extra,
        "batch_size": 1,  # Will be overridden by actual batch
        "vocab_size": meta["vocab_size"],
        "seq_len": meta["seq_len"],
        "num_puzzle_identifiers": meta["num_puzzle_identifiers"],
        "causal": False,
    }

    model_cls = load_model_class(config.arch.name)
    loss_cls = load_model_class(config.arch.loss.name)
    loss_extra = {k: v for k, v in OmegaConf.to_container(config.arch.loss).items()
                  if k not in ("name",)}

    with torch.device("cuda"):
        inner_model = model_cls(model_cfg)
        model = loss_cls(inner_model, **loss_extra)

    # Load weights
    ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, "step_*")))
    if not ckpt_files:
        print("ERROR: No checkpoint found!")
        return

    state = torch.load(ckpt_files[-1], map_location="cuda", weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"])
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    print(f"Loaded: {ckpt_files[-1]}")
    model.eval()

    with open(args.test_json) as f:
        test_pairs = json.load(f)

    total = len(test_pairs)
    seq_len = meta["seq_len"]

    # Process in small batches to fit GPU memory
    BATCH_SIZE = 32
    all_preds = []
    all_expected = []

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            chunk = test_pairs[start:start + BATCH_SIZE]
            batch = make_batch(chunk, seq_len)
            batch = {k: v.cuda() for k, v in batch.items()}

            # ACT inference: initial_carry + forward loop
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=[]
                )
                steps += 1
                if all_finish or steps > 20:
                    break

            # Extract predictions from the carry's outputs
            # preds dict has "preds" key with argmax predictions
            if preds is not None and "preds" in preds:
                pred_ids = preds["preds"].cpu()
            else:
                # Fallback: get logits from last forward
                pred_ids = metrics.get("preds", carry.current_data.get("inputs", batch["inputs"])).cpu()

            # For each example, extract the output portion predictions
            for j, pair in enumerate(chunk):
                inp_tokens = grid_to_tokens(pair["input"])
                out_tokens = grid_to_tokens(pair["output"])

                out_start = len(inp_tokens) + 1  # after input + EOS
                out_end = out_start + len(out_tokens)

                pred_out = pred_ids[j, out_start:out_end].tolist()
                all_preds.append(pred_out)
                all_expected.append(out_tokens)

            if start == 0:
                print(f"  ACT inference steps: {steps}")

    # Compute metrics
    exact_correct = 0
    correct_tokens = 0
    total_tokens = 0
    wrong_examples = []

    for i in range(total):
        pred = all_preds[i]
        exp = all_expected[i]

        n_correct = sum(1 for p, e in zip(pred, exp) if p == e)
        correct_tokens += n_correct
        total_tokens += len(exp)

        if pred == exp:
            exact_correct += 1
        elif len(wrong_examples) < 10:
            pair = test_pairs[i]
            rows = len(pair["input"])
            cols = len(pair["input"][0])
            pred_grid = tokens_to_grid(pred, rows, cols)
            exp_grid = tokens_to_grid(exp, rows, cols)
            wrong_examples.append((i, pair["input"], exp_grid, pred_grid))

    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
    exact_pct = exact_correct / total if total > 0 else 0

    print(f"\nGENERALIZATION RESULTS (n=8, {total} unseen pairs):")
    print(f"  Token accuracy: {token_acc:.1%}")
    print(f"  Exact match:    {exact_correct}/{total} ({exact_pct:.1%})")

    if wrong_examples:
        print(f"\n  Sample mismatches (first {len(wrong_examples)}):")
        for idx, inp, exp, pred in wrong_examples:
            diffs = []
            for r in range(len(exp)):
                if exp[r] != pred[r]:
                    diffs.append(f"row{r}: exp={exp[r]} got={pred[r]}")
            print(f"    Pair {idx}: {'; '.join(diffs[:3])}")
    else:
        print("\n  PERFECT GENERALIZATION!")


if __name__ == "__main__":
    main()
