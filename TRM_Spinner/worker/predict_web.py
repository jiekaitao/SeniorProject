"""Run inference with a trained TRM checkpoint on a saved dataset split.

Reads the same `dataset.json`/`*.npy` layout produced by `data_converter.py`,
loads the matching checkpoint, then writes a `predictions.json` file with
the input grid, predicted output grid, and ground-truth grid for each
example.

Intended to be invoked as a subprocess after training finishes; the worker
API then serves the resulting JSON to the frontend results dashboard.

Usage:

    python predict_web.py \
        --data-dir /app/uploads/<session>/train \
        --checkpoint /path/to/step_N \
        --output /app/uploads/<session>/predictions.json \
        --arch trm --hidden-size 512 --H-cycles 3 --L-cycles 6 \
        --L-layers 2 --num-heads 8 --halt-max-steps 16 \
        --limit 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


PAD_ID = 0
EOS_ID = 1
CELL_OFFSET = 2


def load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cuda", weights_only=True)
    # torch.compile wraps params under `_orig_mod.`; the bare inner model
    # does not, so strip the prefix if present.
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("_orig_mod.", "")] = v
    return cleaned


def build_model(
    vocab_size: int,
    seq_len: int,
    num_puzzle_identifiers: int,
    arch: str,
    hidden_size: int,
    H_cycles: int,
    L_cycles: int,
    H_layers: int,
    L_layers: int,
    num_heads: int,
    expansion: int,
    halt_max_steps: int,
    puzzle_emb_ndim: int,
    puzzle_emb_len: int,
    batch_size: int,
    forward_dtype: str = "bfloat16",
) -> torch.nn.Module:
    # We reuse the same model class the trainer does.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.functions import load_model_class  # noqa: E402

    model_cls = load_model_class(arch)
    config = dict(
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False,
        hidden_size=hidden_size,
        num_heads=num_heads,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        H_layers=H_layers,
        L_layers=L_layers,
        expansion=expansion,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=0.0,
        puzzle_emb_ndim=puzzle_emb_ndim,
        puzzle_emb_len=puzzle_emb_len,
        pos_encodings="rope",
        forward_dtype=forward_dtype,
        mlp_t=False,
        reduced_mlp=False,
        no_ACT_continue=True,
    )
    with torch.device("cuda"):
        return model_cls(config)


def _decode_grid(tokens: np.ndarray, mask: np.ndarray, rows: int, cols: int) -> List[List[int]]:
    """Pull `rows*cols` valid cells out of a token sequence and reshape."""
    vals = []
    for t, m in zip(tokens.tolist(), mask.tolist()):
        if not m:
            continue
        if t <= 1:  # PAD or EOS — stop
            break
        vals.append(int(t) - CELL_OFFSET)
        if len(vals) == rows * cols:
            break
    while len(vals) < rows * cols:
        vals.append(0)
    grid = []
    for r in range(rows):
        grid.append(vals[r * cols : (r + 1) * cols])
    return grid


def _split_input_output(
    tokens: np.ndarray, labels: np.ndarray
) -> Tuple[List[List[int]], List[List[int]]]:
    """Recover the original input/output grids from the packed token seq.

    Layout (see data_converter._encode_pair):
      [input_flat] EOS [output_flat] EOS [PAD...]

    `labels == -1` on everything that isn't part of the output region;
    the first EOS in the tokens marks end of input.
    """
    # Find EOS after input region by walking tokens until we hit EOS.
    in_end = None
    for i, t in enumerate(tokens):
        if int(t) == EOS_ID:
            in_end = i
            break
    if in_end is None:
        in_end = len(tokens)

    in_tokens = tokens[:in_end]
    in_vals = [int(t) - CELL_OFFSET for t in in_tokens if int(t) > 1]

    # Output region = tokens where labels != -1 and != EOS
    out_mask = (labels != -1) & (tokens != EOS_ID)
    out_tokens = tokens[out_mask]
    out_vals = [int(t) - CELL_OFFSET for t in out_tokens]

    # We don't know the exact grid dims from the sequence alone, so infer
    # them heuristically: most workloads in TRM Spinner use square-ish
    # aspect ratios and we pad to max, so guessing rows == cols == sqrt(n)
    # is a decent fallback. For Hanoi (rows x 3) we pick a factor of 3.
    def _reshape(vals: List[int]) -> List[List[int]]:
        n = len(vals)
        if n == 0:
            return [[]]
        if n % 3 == 0:
            rows = n // 3
            cols = 3
        else:
            rows = int(np.sqrt(n))
            cols = max(1, n // max(rows, 1))
            while rows * cols < n:
                cols += 1
        grid = [vals[r * cols : (r + 1) * cols] for r in range(rows) if r * cols < n]
        # pad last row if short
        if grid and len(grid[-1]) < cols:
            grid[-1] = grid[-1] + [0] * (cols - len(grid[-1]))
        return grid

    return _reshape(in_vals), _reshape(out_vals)


def run_inference(
    data_dir: str,
    checkpoint: str,
    output: str,
    arch: str = "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
    hidden_size: int = 512,
    H_cycles: int = 3,
    L_cycles: int = 6,
    H_layers: int = 0,
    L_layers: int = 2,
    num_heads: int = 8,
    expansion: int = 4,
    halt_max_steps: int = 16,
    puzzle_emb_ndim: int = 512,
    puzzle_emb_len: int = 16,
    limit: int = 16,
) -> Dict[str, Any]:
    with open(os.path.join(data_dir, "dataset.json"), "r") as f:
        metadata = json.load(f)

    set_name = metadata["sets"][0]
    inputs = np.load(os.path.join(data_dir, f"{set_name}__inputs.npy"))
    labels = np.load(os.path.join(data_dir, f"{set_name}__labels.npy"))
    puzzle_ids = np.load(os.path.join(data_dir, f"{set_name}__puzzle_identifiers.npy"))

    n = inputs.shape[0]
    k = min(limit, n)
    idx = np.arange(k)
    batch_inputs = torch.from_numpy(inputs[idx]).to("cuda", dtype=torch.int32)
    batch_labels = torch.from_numpy(labels[idx]).to("cuda", dtype=torch.int32)
    batch_pids = torch.from_numpy(puzzle_ids[idx]).to("cuda", dtype=torch.int32)

    model = build_model(
        vocab_size=metadata["vocab_size"],
        seq_len=metadata["seq_len"],
        num_puzzle_identifiers=metadata["num_puzzle_identifiers"],
        arch=arch,
        hidden_size=hidden_size,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        H_layers=H_layers,
        L_layers=L_layers,
        num_heads=num_heads,
        expansion=expansion,
        halt_max_steps=halt_max_steps,
        puzzle_emb_ndim=puzzle_emb_ndim,
        puzzle_emb_len=puzzle_emb_len,
        batch_size=k,
    )

    state_dict = load_checkpoint_state_dict(checkpoint)
    # Strip the ACTLossHead wrapper prefix ("model.") if present.
    clean = {}
    for key, v in state_dict.items():
        clean[key[6:] if key.startswith("model.") else key] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"checkpoint loaded, missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)

    model.eval()

    batch = {
        "inputs": batch_inputs,
        "labels": batch_labels,
        "puzzle_identifiers": batch_pids,
    }

    with torch.inference_mode():
        carry = model.initial_carry(batch)
        # initial_carry constructs bookkeeping tensors on CPU by default;
        # the model runs on CUDA, so move every tensor across.
        carry.steps = carry.steps.cuda()
        carry.halted = carry.halted.cuda()
        carry.prev_q = carry.prev_q.cuda()
        carry.current_data = {
            k_: v.cuda() if isinstance(v, torch.Tensor) else v
            for k_, v in carry.current_data.items()
        }
        carry.inner_carry.z_H = carry.inner_carry.z_H.cuda()
        carry.inner_carry.z_L = carry.inner_carry.z_L.cuda()

        # Run ACT loop until all halted or max steps.
        for _ in range(halt_max_steps):
            carry, outputs = model(carry=carry, batch=batch)
            if carry.halted.all().item():
                break

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        label_np = batch_labels.cpu().numpy()
        input_np = batch_inputs.cpu().numpy()

    records: List[Dict[str, Any]] = []
    total_correct_seq = 0
    total_correct_tokens = 0
    total_tokens = 0

    for i in range(k):
        in_grid, gt_grid = _split_input_output(input_np[i], label_np[i])
        pred_tokens = preds[i]
        # Predicted output: positions where labels != -1
        out_mask = label_np[i] != -1
        ignore_eos = label_np[i] != EOS_ID
        take = out_mask & ignore_eos
        pred_vals = [max(0, int(t) - CELL_OFFSET) for t in pred_tokens[take]]
        gt_vals = [max(0, int(t) - CELL_OFFSET) for t in label_np[i][take]]

        # Reshape pred to same shape as ground truth.
        if gt_grid and gt_grid[0]:
            rows = len(gt_grid)
            cols = len(gt_grid[0])
            while len(pred_vals) < rows * cols:
                pred_vals.append(0)
            pred_vals = pred_vals[: rows * cols]
            pred_grid = [pred_vals[r * cols : (r + 1) * cols] for r in range(rows)]
        else:
            pred_grid = [pred_vals]

        correct_tokens = sum(1 for p, g in zip(pred_vals, gt_vals) if p == g)
        total_correct_tokens += correct_tokens
        total_tokens += len(gt_vals)

        exact_match = pred_vals == gt_vals and len(pred_vals) == len(gt_vals)
        if exact_match:
            total_correct_seq += 1

        records.append(
            {
                "index": int(i),
                "input": in_grid,
                "expected_output": gt_grid,
                "predicted_output": pred_grid,
                "exact_match": exact_match,
                "token_accuracy": correct_tokens / max(1, len(gt_vals)),
            }
        )

    summary = {
        "total_examples": int(k),
        "exact_match_rate": total_correct_seq / max(1, k),
        "token_accuracy": total_correct_tokens / max(1, total_tokens),
    }

    result = {"summary": summary, "predictions": records}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {output} ({k} predictions, exact={summary['exact_match_rate']:.2%})",
          file=sys.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with inputs.npy etc.")
    parser.add_argument("--checkpoint", required=True, help="Path to step_N checkpoint file.")
    parser.add_argument("--output", required=True, help="Path to predictions.json output.")
    parser.add_argument("--arch", default="recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--H-cycles", type=int, default=3)
    parser.add_argument("--L-cycles", type=int, default=6)
    parser.add_argument("--H-layers", type=int, default=0)
    parser.add_argument("--L-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--halt-max-steps", type=int, default=16)
    parser.add_argument("--puzzle-emb-ndim", type=int, default=512)
    parser.add_argument("--puzzle-emb-len", type=int, default=16)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    run_inference(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        arch=args.arch,
        hidden_size=args.hidden_size,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        halt_max_steps=args.halt_max_steps,
        puzzle_emb_ndim=args.puzzle_emb_ndim,
        puzzle_emb_len=args.puzzle_emb_len,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
