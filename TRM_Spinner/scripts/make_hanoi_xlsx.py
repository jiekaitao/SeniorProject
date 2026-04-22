#!/usr/bin/env python3
"""Generate a Towers-of-Hanoi training workbook that can be uploaded to the
TRM Spinner chat ("upload file" button) end-to-end.

Each row contains two columns:
  A — JSON-encoded input grid   (state of the 3 pegs now)
  B — JSON-encoded output grid  (state after the next optimal move)

The first row is a plain header ("input", "output") so a user opening the
file in Excel can see what's going on without code. The parser skips the
header automatically.
"""
from __future__ import annotations

import argparse
import json
import os

import openpyxl


def pegs_to_grid(pegs, height):
    grid = []
    for row in range(height):
        r = []
        for peg in pegs:
            r.append(peg[row] if row < len(peg) else 0)
        grid.append(r)
    return list(reversed(grid))


def solve_hanoi(n, source=0, target=2, auxiliary=1):
    pegs = [list(range(n, 0, -1)), [], []]
    states = [pegs_to_grid(pegs, n)]

    def move(k, src, tgt, aux):
        if k == 0:
            return
        move(k - 1, src, aux, tgt)
        disk = pegs[src].pop()
        pegs[tgt].append(disk)
        states.append(pegs_to_grid(pegs, n))
        move(k - 1, aux, tgt, src)

    move(n, source, target, auxiliary)
    return states


def pad_grid(grid, rows, cols):
    out = []
    for r in range(rows):
        if r < len(grid):
            row = list(grid[r]) + [0] * (cols - len(grid[r]))
        else:
            row = [0] * cols
        out.append(row[:cols])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-disks", type=int, default=5)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "hanoi_training.xlsx"),
    )
    args = parser.parse_args()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "hanoi"
    ws.append(["input", "output"])

    count = 0
    for n_disks in range(2, args.max_disks + 1):
        states = solve_hanoi(n_disks)
        for i in range(len(states) - 1):
            inp = pad_grid(states[i], args.max_disks, 3)
            out = pad_grid(states[i + 1], args.max_disks, 3)
            ws.append([json.dumps(inp), json.dumps(out)])
            count += 1

    # Column widths so it's pleasant to read in Excel.
    ws.column_dimensions["A"].width = 60
    ws.column_dimensions["B"].width = 60

    out_path = os.path.abspath(args.output)
    wb.save(out_path)
    print(f"wrote {count} pairs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
