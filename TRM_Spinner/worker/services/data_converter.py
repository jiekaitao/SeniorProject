from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class DataConversionResult:
    vocab_size: int
    seq_len: int
    num_examples: int
    output_dir: str


# Vocab constants
PAD_ID = 0
EOS_ID = 1
CELL_OFFSET = 2  # Cell values are encoded as value + CELL_OFFSET


def _flatten_grid(grid: List[List[int]]) -> List[int]:
    """Flatten a 2D grid row-by-row."""
    flat = []
    for row in grid:
        flat.extend(row)
    return flat


def _encode_pair(
    inp: List[List[int]], out: List[List[int]], seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Encode an input/output grid pair into padded sequences.

    Format: [input_flat] + [EOS] + [output_flat] + [EOS] + [PAD...]
    Labels: [-1 for input region] + [EOS] + [output values] + [EOS] + [-1 for pad]
    Using -1 as ignore label for input/pad positions.
    """
    flat_in = [v + CELL_OFFSET for v in _flatten_grid(inp)]
    flat_out = [v + CELL_OFFSET for v in _flatten_grid(out)]

    # Build input sequence: input_flat + EOS + output_flat + EOS
    tokens = flat_in + [EOS_ID] + flat_out + [EOS_ID]

    # Build label sequence: ignore input region, predict output region
    # The label for the input part and first EOS is "ignore" (-1)
    # The label for the output part and final EOS is the actual value
    ignore_label = -1
    labels = [ignore_label] * (len(flat_in) + 1) + flat_out + [EOS_ID]

    # Pad to seq_len
    pad_len = seq_len - len(tokens)
    if pad_len > 0:
        tokens = tokens + [PAD_ID] * pad_len
        labels = labels + [ignore_label] * pad_len
    else:
        tokens = tokens[:seq_len]
        labels = labels[:seq_len]

    return np.array(tokens, dtype=np.int32), np.array(labels, dtype=np.int32)


def convert_data(
    data: List[Dict[str, Any]], output_dir: str, split: str = "train"
) -> DataConversionResult:
    """Convert JSON array of {input, output} grid pairs to numpy format.

    Produces:
      - inputs.npy: (N, seq_len) int32
      - labels.npy: (N, seq_len) int32
      - puzzle_identifiers.npy: (N,) int32
      - puzzle_indices.npy: (N+1,) int32
      - group_indices.npy: (2,) int32
      - dataset.json: PuzzleDatasetMetadata
    """
    if not data:
        raise ValueError("Data cannot be empty")

    # Determine max cell value and max sequence length
    max_cell_value = 0
    max_seq_needed = 0

    for pair in data:
        inp = pair["input"]
        out = pair["output"]

        for row in inp:
            for v in row:
                max_cell_value = max(max_cell_value, v)
        for row in out:
            for v in row:
                max_cell_value = max(max_cell_value, v)

        # Seq length needed: flatten(input) + EOS + flatten(output) + EOS
        in_size = sum(len(row) for row in inp)
        out_size = sum(len(row) for row in out)
        seq_needed = in_size + 1 + out_size + 1
        max_seq_needed = max(max_seq_needed, seq_needed)

    vocab_size = max_cell_value + CELL_OFFSET + 1  # PAD + EOS + cell values
    seq_len = max_seq_needed  # Exactly the max needed (no extra padding)

    # Encode all pairs
    num_examples = len(data)
    all_inputs = np.zeros((num_examples, seq_len), dtype=np.int32)
    all_labels = np.zeros((num_examples, seq_len), dtype=np.int32)

    for i, pair in enumerate(data):
        tokens, labels = _encode_pair(pair["input"], pair["output"], seq_len)
        all_inputs[i] = tokens
        all_labels[i] = labels

    # Each example is its own puzzle (simple 1:1 mapping)
    puzzle_identifiers = np.arange(num_examples, dtype=np.int32)
    puzzle_indices = np.arange(num_examples + 1, dtype=np.int32)  # [0, 1, 2, ..., N]
    group_indices = np.array([0, num_examples], dtype=np.int32)  # Single group

    # Create output directory
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Save numpy arrays
    set_name = split
    np.save(os.path.join(split_dir, f"{set_name}__inputs.npy"), all_inputs)
    np.save(os.path.join(split_dir, f"{set_name}__labels.npy"), all_labels)
    np.save(os.path.join(split_dir, f"{set_name}__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(split_dir, f"{set_name}__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(split_dir, f"{set_name}__group_indices.npy"), group_indices)

    # Save metadata
    metadata = {
        "pad_id": PAD_ID,
        "ignore_label_id": -1,
        "blank_identifier_id": 0,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "num_puzzle_identifiers": num_examples,
        "total_groups": 1,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": num_examples,
        "sets": [set_name],
    }
    with open(os.path.join(split_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return DataConversionResult(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_examples=num_examples,
        output_dir=output_dir,
    )
