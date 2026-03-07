from __future__ import annotations

import pytest
from services.classifier import classify_problem


class TestClassifier:
    """Test the rule-based problem classifier."""

    def test_sudoku_classified_as_constraint_satisfaction(self):
        result = classify_problem("I want to solve sudoku puzzles")
        assert result.category == "constraint_satisfaction"
        assert result.confidence > 0.3

    def test_csp_classified_as_constraint_satisfaction(self):
        result = classify_problem("constraint satisfaction problem with valid assignment")
        assert result.category == "constraint_satisfaction"

    def test_nqueens_classified_as_constraint_satisfaction(self):
        result = classify_problem("n-queens problem")
        assert result.category == "constraint_satisfaction"

    def test_shortest_path_classified_as_routing(self):
        result = classify_problem("find the shortest path in a graph")
        assert result.category == "routing"
        assert result.confidence > 0.3

    def test_maze_classified_as_routing(self):
        result = classify_problem("navigate through a maze")
        assert result.category == "routing"

    def test_tsp_classified_as_routing(self):
        result = classify_problem("traveling salesman problem TSP")
        assert result.category == "routing"

    def test_schedule_classified_as_scheduling(self):
        result = classify_problem("create a schedule for classes")
        assert result.category == "scheduling"
        assert result.confidence > 0.3

    def test_timetable_classified_as_scheduling(self):
        result = classify_problem("job shop timetable resource assignment")
        assert result.category == "scheduling"

    def test_polynomial_classified_as_symbolic_math(self):
        result = classify_problem("solve polynomial equations")
        assert result.category == "symbolic_math"
        assert result.confidence > 0.3

    def test_arithmetic_classified_as_symbolic_math(self):
        result = classify_problem("compute arithmetic algebra calculations")
        assert result.category == "symbolic_math"

    def test_arc_classified_as_puzzle_solving(self):
        result = classify_problem("ARC challenge grid transform")
        assert result.category == "puzzle_solving"
        assert result.confidence > 0.3

    def test_puzzle_grid_classified_as_puzzle_solving(self):
        result = classify_problem("puzzle with grid transformation and reasoning")
        assert result.category == "puzzle_solving"

    def test_image_pattern_classified_as_pattern_recognition(self):
        result = classify_problem("classify image patterns and predict sequences")
        assert result.category == "pattern_recognition"
        assert result.confidence > 0.3

    def test_detect_classify_as_pattern_recognition(self):
        result = classify_problem("detect and recognize patterns in data")
        assert result.category == "pattern_recognition"

    def test_unsuitable_returns_unsuitable(self):
        result = classify_problem("hello how are you today nice weather")
        assert result.category == "unsuitable"
        assert result.confidence < 0.3

    def test_empty_input_returns_unsuitable(self):
        result = classify_problem("")
        assert result.category == "unsuitable"

    def test_confidence_between_0_and_1(self):
        result = classify_problem("solve sudoku constraint satisfaction")
        assert 0.0 <= result.confidence <= 1.0

    def test_top_match_returned(self):
        """When multiple categories match, the highest scoring one wins."""
        result = classify_problem("sudoku puzzle grid constraint")
        assert result.confidence > 0.3
        assert result.category in ("constraint_satisfaction", "puzzle_solving")
