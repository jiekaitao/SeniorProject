from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ClassificationResult:
    category: str
    confidence: float


# Keyword lists and weights for each category
CATEGORY_KEYWORDS: Dict[str, List[Tuple[str, float]]] = {
    "constraint_satisfaction": [
        ("sudoku", 0.4),
        ("csp", 0.35),
        ("constraint", 0.3),
        ("satisfy", 0.25),
        ("valid assignment", 0.3),
        ("n-queens", 0.4),
        ("nqueens", 0.4),
        ("latin square", 0.3),
        ("backtracking", 0.2),
    ],
    "routing": [
        ("path", 0.2),
        ("route", 0.25),
        ("shortest", 0.3),
        ("graph", 0.2),
        ("navigate", 0.25),
        ("maze", 0.3),
        ("tsp", 0.4),
        ("traveling salesman", 0.4),
        ("dijkstra", 0.3),
        ("network flow", 0.3),
    ],
    "scheduling": [
        ("schedule", 0.35),
        ("timetable", 0.35),
        ("assign", 0.15),
        ("resource", 0.2),
        ("job shop", 0.4),
        ("allocation", 0.2),
        ("calendar", 0.2),
    ],
    "symbolic_math": [
        ("equation", 0.3),
        ("polynomial", 0.35),
        ("arithmetic", 0.3),
        ("algebra", 0.3),
        ("compute", 0.15),
        ("calculate", 0.2),
        ("math", 0.2),
        ("formula", 0.2),
        ("symbolic", 0.3),
    ],
    "puzzle_solving": [
        ("puzzle", 0.3),
        ("grid", 0.15),
        ("arc", 0.4),
        ("transform", 0.2),
        ("pattern", 0.15),
        ("reasoning", 0.15),
        ("abstraction", 0.25),
    ],
    "pattern_recognition": [
        ("classify", 0.25),
        ("recognize", 0.25),
        ("detect", 0.25),
        ("image", 0.2),
        ("sequence", 0.15),
        ("predict", 0.2),
        ("recognition", 0.3),
        ("feature", 0.15),
    ],
}


def classify_problem(description: str) -> ClassificationResult:
    """Classify a problem description into a category with confidence score.

    Scores 0-1 for each category. Returns top match with >0.3 threshold.
    If no match >0.3, returns "unsuitable".
    """
    if not description.strip():
        return ClassificationResult(category="unsuitable", confidence=0.0)

    text = description.lower()
    scores: Dict[str, float] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        matched = 0
        for keyword, weight in keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text):
                score += weight
                matched += 1

        # Normalize: cap at 1.0
        if matched > 0:
            scores[category] = min(score, 1.0)
        else:
            scores[category] = 0.0

    if not scores:
        return ClassificationResult(category="unsuitable", confidence=0.0)

    # Find top category
    best_category = max(scores, key=lambda k: scores[k])
    best_score = scores[best_category]

    if best_score > 0.3:
        return ClassificationResult(category=best_category, confidence=best_score)

    return ClassificationResult(category="unsuitable", confidence=best_score)
