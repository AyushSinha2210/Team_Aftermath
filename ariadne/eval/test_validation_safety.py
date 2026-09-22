"""Unit tests for validation harness and safety rail enforcement."""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from ariadne.finetuning.validate import verify_safety_rails
from ariadne.eval.metrics import (
    compute_ndcg_at_k,
    compute_mrr_at_k,
    compute_recall_at_k,
    evaluate_ranking,
)


def test_safety_rails_valid_path_passes() -> None:
    """Verifies that legitimate 'valid' paths pass safety rails without error."""
    valid_path = Path("data/raw/valid")
    # Should not raise
    verify_safety_rails(valid_path, "valid")


def test_safety_rails_rejects_test_split_name() -> None:
    """Verifies that any split_name == 'test' raises AssertionError."""
    valid_path = Path("data/raw/valid")
    with pytest.raises(AssertionError, match="CRITICAL SAFETY RAIL VIOLATION"):
        verify_safety_rails(valid_path, "test")


def test_safety_rails_rejects_test_path() -> None:
    """Verifies that any path referencing 'test' raises AssertionError."""
    test_path = Path("data/raw/test")
    with pytest.raises(AssertionError, match="CRITICAL SAFETY RAIL VIOLATION"):
        verify_safety_rails(test_path, "valid")


def test_safety_rails_rejects_non_valid_split() -> None:
    """Verifies that non-valid split names raise AssertionError."""
    train_path = Path("data/raw/train")
    with pytest.raises(AssertionError, match="CRITICAL SAFETY RAIL VIOLATION"):
        verify_safety_rails(train_path, "train")


def test_metrics_perfect_ranking() -> None:
    """Verifies NDCG, MRR, Recall for ideal rank 1."""
    ranks = [1, 1, 1, 1]
    assert compute_ndcg_at_k(ranks, k=10) == pytest.approx(1.0)
    assert compute_mrr_at_k(ranks, k=10) == pytest.approx(1.0)
    assert compute_recall_at_k(ranks, k=1) == pytest.approx(1.0)


def test_metrics_cutoff_exclusion() -> None:
    """Verifies that ranks beyond k receive zero credit."""
    ranks = [11, 12, 15]
    assert compute_ndcg_at_k(ranks, k=10) == 0.0
    assert compute_mrr_at_k(ranks, k=10) == 0.0
    assert compute_recall_at_k(ranks, k=10) == 0.0


def test_metrics_rank_2() -> None:
    """Verifies metrics for rank 2."""
    ranks = [2]
    expected_ndcg = 1.0 / np.log2(3)
    assert compute_ndcg_at_k(ranks, k=10) == pytest.approx(expected_ndcg)
    assert compute_mrr_at_k(ranks, k=10) == pytest.approx(0.5)
    assert compute_recall_at_k(ranks, k=1) == 0.0
    assert compute_recall_at_k(ranks, k=5) == 1.0
