"""
DICE — Directed Interaction Compatibility Estimator

Predicts N-tuple block stacking quality from:
  - Measured singleton gains
  - Predicted pairwise epistasis (from cheap spectral/BLOOD/CKA features)

Core equation:
  F_hat(S) = F(∅) + Σ Δ_i + Σ ε_hat(i→j)

where ε_hat is predicted from cheap pair features without running the dual probe.
"""
from __future__ import annotations
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import numpy as np
import json

Block = Tuple[int, int]  # (start, end)


def rank01(x: np.ndarray) -> np.ndarray:
    """Rank-normalize to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n <= 1:
        return np.array([0.5] * n)
    order = x.argsort().argsort()
    return order / (n - 1)


def compute_observed_epistasis(
    pair_score: float,
    score_a: float,
    score_b: float,
    baseline: float,
) -> float:
    """ε_ij = F({i,j}) - F({i}) - F({j}) + F(∅)"""
    return pair_score - score_a - score_b + baseline


def score_edge(
    features: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Predict epistasis from cheap pair features.
    All features are sign-normalized so that HIGHER = BETTER (less interference).

    Default weights are theory-signed, no fitting required.
    """
    if weights is None:
        weights = {
            "rho_lift": 1.00,        # conditional rho improvement
            "blood_lift": 1.00,      # conditional BLOOD improvement
            "effect_orth": 1.00,     # output-effect orthogonality (1 - CKA)
            "territory_orth": 0.75,  # BLOOD-territory disjointness
            "region_dist": 0.50,     # depth distance / N
            "ood_safe": 0.75,        # negative seam Mahalanobis change
        }

    score = 0.0
    for key, w in weights.items():
        score += w * features.get(key, 0.0)
    return score


def predict_tuple_score(
    blocks: List[Block],
    baseline: float,
    single_gains: Dict[Block, float],
    edge_scores: Dict[Tuple[Block, Block], float],
) -> float:
    """
    F_hat(S) = F(∅) + Σ Δ_i + Σ ε_hat(i→j)

    blocks must be sorted by start position.
    """
    S = sorted(blocks, key=lambda b: b[0])
    score = baseline
    for b in S:
        score += single_gains.get(b, 0.0)
    for a, b in combinations(S, 2):
        score += edge_scores.get((a, b), 0.0)
    return score


def rank_all_pairs(
    candidates: List[Block],
    baseline: float,
    single_gains: Dict[Block, float],
    edge_scores: Dict[Tuple[Block, Block], float],
) -> List[Tuple[Tuple[Block, Block], float]]:
    """Rank all non-overlapping pairs by predicted score."""
    results = []
    for a, b in combinations(sorted(candidates), 2):
        # Check non-overlapping
        if a[1] > b[0]:
            continue
        score = predict_tuple_score([a, b], baseline, single_gains, edge_scores)
        results.append(((a, b), score))
    results.sort(key=lambda x: -x[1])
    return results


def beam_search_tuples(
    candidates: List[Block],
    baseline: float,
    single_gains: Dict[Block, float],
    edge_scores: Dict[Tuple[Block, Block], float],
    max_k: int = 3,
    beam_size: int = 10,
) -> List[Tuple[Tuple[Block, ...], float]]:
    """
    Beam search over N-tuples using the quadratic surrogate.
    Returns sorted list of (tuple_of_blocks, predicted_score).
    """
    candidates = sorted(candidates, key=lambda b: b[0])

    # Start with empty set and all singletons
    beam = [(tuple(), baseline)]
    all_results = {}

    for step in range(max_k):
        new_beam = {}
        for S, current_score in beam:
            used_ranges = set()
            for b in S:
                for l in range(b[0], b[1]):
                    used_ranges.add(l)

            for c in candidates:
                # Check non-overlapping
                if any(l in used_ranges for l in range(c[0], c[1])):
                    continue

                # Compute marginal gain
                marginal = single_gains.get(c, 0.0)
                for a in S:
                    key = (a, c) if a[0] < c[0] else (c, a)
                    marginal += edge_scores.get(key, 0.0)

                if marginal <= 0:
                    continue

                T = tuple(sorted(S + (c,), key=lambda b: b[0]))
                new_score = current_score + marginal

                if T not in new_beam or new_score > new_beam[T]:
                    new_beam[T] = new_score

        if not new_beam:
            break

        beam = sorted(new_beam.items(), key=lambda x: -x[1])[:beam_size]
        for T, s in beam:
            if T not in all_results or s > all_results[T]:
                all_results[T] = s

    return sorted(all_results.items(), key=lambda x: -x[1])
