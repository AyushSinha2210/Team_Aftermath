"""Unit tests for MaxSim late-interaction retrieval engine."""

from __future__ import annotations

import numpy as np

from ariadne.retrieval.late_interaction import compute_maxsim, rank_by_maxsim


def test_compute_maxsim_empty():
    q = np.empty((0, 32))
    d = np.empty((0, 32))
    assert compute_maxsim(q, d) == 0.0


def test_compute_maxsim_orthogonal_vs_identical():
    # 2 query tokens in 2-D space: (1, 0) and (0, 1)
    q = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Identical document tokens
    d_identical = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Orthogonal document tokens
    d_orthogonal = np.array([[-1.0, 0.0], [0.0, -1.0]])

    score_id = compute_maxsim(q, d_identical)
    score_orth = compute_maxsim(q, d_orthogonal)

    # MaxSim on identical tokens: 1.0 + 1.0 = 2.0
    assert np.isclose(score_id, 2.0)
    assert score_id > score_orth


def test_rank_by_maxsim():
    q = np.array([[1.0, 0.0]])
    d1 = np.array([[1.0, 0.0]])   # sim = 1.0
    d2 = np.array([[0.5, 0.866]]) # sim = 0.5
    d3 = np.array([[0.0, 1.0]])   # sim = 0.0

    ranked = rank_by_maxsim(q, [d3, d1, d2], ["doc3", "doc1", "doc2"], k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == "doc1"
    assert ranked[1][0] == "doc2"
