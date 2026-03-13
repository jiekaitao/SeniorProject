from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from services.data_converter import convert_data, DataConversionResult


class TestDataConverter:
    """Test converting JSON grid pairs to numpy arrays for TRM training."""

    def test_basic_conversion_produces_files(self, sample_training_data):
        """Converting sample data should produce all required numpy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            assert os.path.exists(os.path.join(tmpdir, "train", "train__inputs.npy"))
            assert os.path.exists(os.path.join(tmpdir, "train", "train__labels.npy"))
            assert os.path.exists(os.path.join(tmpdir, "train", "train__puzzle_identifiers.npy"))
            assert os.path.exists(os.path.join(tmpdir, "train", "train__puzzle_indices.npy"))
            assert os.path.exists(os.path.join(tmpdir, "train", "train__group_indices.npy"))
            assert os.path.exists(os.path.join(tmpdir, "train", "dataset.json"))

    def test_inputs_shape(self, sample_training_data):
        """Inputs should be 2D: (num_examples, seq_len)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            inputs = np.load(os.path.join(tmpdir, "train", "train__inputs.npy"))
            assert inputs.ndim == 2
            assert inputs.shape[0] == len(sample_training_data)

    def test_labels_shape_matches_inputs(self, sample_training_data):
        """Labels should have the same shape as inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            inputs = np.load(os.path.join(tmpdir, "train", "train__inputs.npy"))
            labels = np.load(os.path.join(tmpdir, "train", "train__labels.npy"))
            assert inputs.shape == labels.shape

    def test_vocab_encoding(self, sample_training_data):
        """Vocab: 0=PAD, 1=EOS, 2+N=cell values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            inputs = np.load(os.path.join(tmpdir, "train", "train__inputs.npy"))
            # PAD=0 should appear (padding)
            assert 0 in inputs
            # EOS=1 should appear (separator)
            assert 1 in inputs
            # Cell values should be offset by 2
            # Original data has values 0,1,2 -> encoded as 2,3,4
            assert result.vocab_size >= 5  # PAD + EOS + at least 3 cell values

    def test_eos_separator_present(self, sample_training_data):
        """Each sequence should have EOS tokens separating input and output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            inputs = np.load(os.path.join(tmpdir, "train", "train__inputs.npy"))
            # Each row should have at least 2 EOS tokens (after input, after output)
            for row in inputs:
                eos_count = np.sum(row == 1)
                assert eos_count >= 2, f"Expected at least 2 EOS tokens, got {eos_count}"

    def test_puzzle_identifiers(self, sample_training_data):
        """Each example should get a puzzle identifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            pids = np.load(os.path.join(tmpdir, "train", "train__puzzle_identifiers.npy"))
            assert pids.shape[0] == len(sample_training_data)

    def test_puzzle_indices_correct(self, sample_training_data):
        """Puzzle indices should mark boundaries between puzzles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            indices = np.load(os.path.join(tmpdir, "train", "train__puzzle_indices.npy"))
            # First index should be 0, last should be num_examples
            assert indices[0] == 0
            assert indices[-1] == len(sample_training_data)

    def test_group_indices_correct(self, sample_training_data):
        """Group indices should define at least one group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            groups = np.load(os.path.join(tmpdir, "train", "train__group_indices.npy"))
            assert groups[0] == 0
            assert len(groups) >= 2  # At least start and end

    def test_dataset_json_metadata(self, sample_training_data):
        """dataset.json should contain valid PuzzleDatasetMetadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            with open(os.path.join(tmpdir, "train", "dataset.json")) as f:
                metadata = json.load(f)

            assert "vocab_size" in metadata
            assert "seq_len" in metadata
            assert "pad_id" in metadata
            assert metadata["pad_id"] == 0
            assert "num_puzzle_identifiers" in metadata
            assert "total_groups" in metadata
            assert "total_puzzles" in metadata
            assert "sets" in metadata
            assert "mean_puzzle_examples" in metadata

    def test_auto_detect_seq_len(self):
        """Seq len should be auto-detected from max grid size."""
        # Large grids need longer sequences
        data = [
            {
                "input": [[i for i in range(10)] for _ in range(10)],
                "output": [[i for i in range(10)] for _ in range(10)],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(data, output_dir=tmpdir)
            # 10x10 input + EOS + 10x10 output + EOS = 202
            assert result.seq_len >= 202

    def test_conversion_result_fields(self, sample_training_data):
        """DataConversionResult should have all expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(sample_training_data, output_dir=tmpdir)

            assert isinstance(result, DataConversionResult)
            assert result.vocab_size > 0
            assert result.seq_len > 0
            assert result.num_examples == len(sample_training_data)
            assert result.output_dir == tmpdir

    def test_empty_data_raises(self):
        """Converting empty data should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                convert_data([], output_dir=tmpdir)

    def test_mismatched_grid_sizes_handled(self):
        """Different grid sizes in the same dataset should work via padding."""
        data = [
            {"input": [[1]], "output": [[2]]},
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = convert_data(data, output_dir=tmpdir)
            inputs = np.load(os.path.join(tmpdir, "train", "train__inputs.npy"))
            # Both examples should have the same seq_len
            assert inputs.shape[0] == 2
            assert inputs.shape[1] == result.seq_len
