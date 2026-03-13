"""Build bprna_data.plk from HuggingFace bpRNA dataset + test splits.

This script recreates the RNAformer training pickle from publicly available
bpRNA data, since the original Uni Freiburg server is down.

Usage:
    python3 build_bprna_dataset.py [--output data/bprna_data.plk]
"""
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path


def dot_bracket_to_pairs(structure: str):
    """Convert dot-bracket notation to pos1id/pos2id lists.

    Handles standard brackets (), [], {}, <> for pseudoknots.
    Returns two lists: pos1id (opening indices), pos2id (closing indices).
    """
    bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
    close_to_open = {v: k for k, v in bracket_pairs.items()}

    stacks = {k: [] for k in bracket_pairs}
    pos1id = []
    pos2id = []

    for i, ch in enumerate(structure):
        if ch in bracket_pairs:
            stacks[ch].append(i)
        elif ch in close_to_open:
            opener = close_to_open[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                pos1id.append(j)
                pos2id.append(i)
        # '.' and other characters are unpaired

    # Sort by pos1id for consistency
    if pos1id:
        pairs = sorted(zip(pos1id, pos2id))
        pos1id = [p[0] for p in pairs]
        pos2id = [p[1] for p in pairs]

    return pos1id, pos2id


def normalize_sequence(seq: str) -> str:
    """Normalize RNA sequence: uppercase, T->U."""
    seq = seq.upper().replace('T', 'U')
    return seq


def main():
    parser = argparse.ArgumentParser(description='Build bpRNA pickle for RNAformer')
    parser.add_argument('--output', type=str, default='data/bprna_data.plk',
                        help='Output pickle path')
    parser.add_argument('--test-data', type=str, default='data/all_test_data.npy',
                        help='Path to all_test_data.npy for test split identification')
    parser.add_argument('--max-len', type=int, default=500,
                        help='Maximum sequence length to include')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading bpRNA from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset('multimolecule/bprna', 'default', split='train')
    print(f"  Loaded {len(ds)} samples")

    # Build sequence -> HuggingFace index mapping
    hf_seq_to_idx = {}
    for i, sample in enumerate(ds):
        seq = normalize_sequence(sample['sequence'])
        hf_seq_to_idx[seq] = i

    # Load test data to identify test sequences
    test_data_path = Path(args.test_data)
    test_sequences = set()
    if test_data_path.exists():
        print(f"Loading test data from {test_data_path}...")
        test_data = np.load(str(test_data_path), allow_pickle=True)
        for d in test_data:
            if d['dataset'] == 'bprna_ts0':
                test_sequences.add(normalize_sequence(d['sequence']))
        print(f"  Found {len(test_sequences)} bpRNA test sequences to exclude from training")
    else:
        print("  WARNING: No test data found, cannot exclude test sequences from training")

    # Convert HuggingFace data to RNAformer pickle format
    print("Converting to RNAformer format...")
    records = []
    skipped_long = 0
    skipped_empty = 0
    skipped_mismatch = 0

    for i, sample in enumerate(ds):
        seq = normalize_sequence(sample['sequence'])
        structure = sample['secondary_structure']

        # Skip sequences that are too long
        if len(seq) > args.max_len:
            skipped_long += 1
            continue

        # Parse secondary structure
        pos1id, pos2id = dot_bracket_to_pairs(structure)

        if len(pos1id) < 1:
            skipped_empty += 1
            continue

        if len(pos1id) != len(pos2id):
            skipped_mismatch += 1
            continue

        # Determine split
        if seq in test_sequences:
            split_name = 'bprna_ts0'
        else:
            split_name = 'train'

        records.append({
            'sequence': seq,
            'pos1id': pos1id,
            'pos2id': pos2id,
            'is_pdb': 0,
            'set': split_name,
        })

    print(f"  Converted {len(records)} samples")
    print(f"  Skipped: {skipped_long} too long, {skipped_empty} no pairs, {skipped_mismatch} mismatched")

    # Build DataFrame
    df = pd.DataFrame(records)

    # Report stats
    for split_name in df['set'].unique():
        subset = df[df['set'] == split_name]
        seq_lens = subset['sequence'].str.len()
        print(f"  {split_name}: {len(subset)} samples, "
              f"seq_len range={seq_lens.min()}-{seq_lens.max()}, "
              f"mean={seq_lens.mean():.0f}")

    # Save pickle
    df.to_pickle(str(output_path))
    print(f"\nSaved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Also build test_sets.plk from the npy file if available
    if test_data_path.exists():
        test_records = []
        test_data = np.load(str(test_data_path), allow_pickle=True)
        for d in test_data:
            test_records.append({
                'sequence': normalize_sequence(d['sequence']),
                'pos1id': d['gt_pos1id'],
                'pos2id': d['gt_pos2id'],
                'is_pdb': 1 if 'pdb' in d['dataset'] else 0,
                'set': d['dataset'],
            })
        test_df = pd.DataFrame(test_records)
        test_output = output_path.parent / 'test_sets.plk'
        test_df.to_pickle(str(test_output))
        print(f"Saved test sets to {test_output} ({test_output.stat().st_size / 1024 / 1024:.1f} MB)")

        for split_name in test_df['set'].unique():
            subset = test_df[test_df['set'] == split_name]
            print(f"  {split_name}: {len(subset)} samples")


if __name__ == '__main__':
    main()
