# Multi-Block and Multi-Pass Experiments

Tests of advanced duplication strategies beyond single-block, single-pass.

## Files

- **`multi_block_test.py`** — Simultaneous duplication of two non-overlapping blocks. Result: blocks interfere rather than stack. Best dual +24.9% < best single +25.7%.

- **`multi_pass_test.py`** — Running the same block 1-6 times. Result: 2 passes optimal, then diminishing returns.

- **`even_odd_test.py`** — Tests if odd/even extra layers have systematic differences. Result: parity doesn't matter; the dominant pattern is diminishing returns.

- **`adaptive_depth.py`** — Per-input adaptive pass count based on convergence threshold `||F(h) - h|| < τ`. Result on 7B: threshold never triggers (blocks aren't contractive on small models).
