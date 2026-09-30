"""Tests for convex score fusion and normalization."""

from __future__ import annotations

import pytest

from ariadne.retrieval.fusion import convex_score_fusion, min_max_normalize


def test_min_max_normalize_empty():
    assert min_max_normalize([]) == {}


def test_min_max_normalize_constant_scores():
    norm = min_max_normalize([("doc1", 5.0), ("doc2", 5.0)])
    assert norm == {"doc1": 1.0, "doc2": 1.0}


def test_min_max_normalize_spread():
    norm = min_max_normalize([("doc1", 0.0), ("doc2", 10.0), ("doc3", 5.0)])
    assert norm["doc1"] == 0.0
    assert norm["doc2"] == 1.0
    assert norm["doc3"] == 0.5


def test_convex_score_fusion_dense_heavy():
    dense = [("d1", 0.9), ("d2", 0.1)]
    sparse = [("d2", 10.0), ("d1", 1.0)]

    # With alpha=0.85, dense heavily dominates
    fused = convex_score_fusion(dense, sparse, alpha=0.85)
    assert fused[0][0] == "d1"


def test_convex_score_fusion_sparse_heavy():
    dense = [("d1", 0.9), ("d2", 0.1)]
    sparse = [("d2", 10.0), ("d1", 1.0)]

    # With alpha=0.10, sparse dominates
    fused = convex_score_fusion(dense, sparse, alpha=0.10)
    assert fused[0][0] == "d2"


def test_convex_score_fusion_invalid_alpha():
    with pytest.raises(ValueError):
        convex_score_fusion([], [], alpha=1.5)
