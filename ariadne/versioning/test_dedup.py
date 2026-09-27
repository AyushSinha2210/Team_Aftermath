"""Unit tests for near-duplicate deduplication in ariadne.versioning.dedup."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ariadne.versioning.dedup import collapse_duplicates, find_near_duplicates


def _make_unit_vector(angle: float, dim: int = 4) -> np.ndarray:
    """Creates an L2-normalized vector in R^dim with given angle in the first 2 dimensions."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = np.cos(angle, dtype=np.float32)
    vec[1] = np.sin(angle, dtype=np.float32)
    return vec


def test_near_identical_embeddings_cluster_together() -> None:
    """Two near-identical embeddings (cosine sim > threshold) correctly cluster together."""
    v1 = _make_unit_vector(0.0)
    v2 = _make_unit_vector(0.05)  # cos(0.05) ~ 0.9987 > 0.95
    sim = float(np.dot(v1, v2))
    assert sim > 0.95

    doc_ids = ["doc_a", "doc_b"]
    embeddings = np.stack([v1, v2])

    clusters = find_near_duplicates(doc_ids, embeddings, threshold=0.95)
    assert len(clusters) == 1
    assert clusters[0] == {"doc_a", "doc_b"}


def test_dissimilar_embeddings_do_not_cluster() -> None:
    """Two dissimilar embeddings do NOT cluster (each forms a singleton)."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Orthogonal, cos sim = 0.0
    sim = float(np.dot(v1, v2))
    assert sim < 0.95

    doc_ids = ["doc_x", "doc_y"]
    embeddings = np.stack([v1, v2])

    clusters = find_near_duplicates(doc_ids, embeddings, threshold=0.95)
    assert len(clusters) == 2
    cluster_sets = [set(c) for c in clusters]
    assert {"doc_x"} in cluster_sets
    assert {"doc_y"} in cluster_sets


def test_three_way_cluster_transitivity() -> None:
    """A three-way cluster (A~B, B~C, but A and C individually just barely under

    threshold with each other) still correctly groups as one cluster via
    transitivity.
    """
    # Angle separation of 0.25 rad:
    # cos(0.25) ~ 0.9689 >= 0.95 (A~B and B~C connect)
    # cos(0.50) ~ 0.8776 < 0.95  (A and C do not directly connect)
    va = _make_unit_vector(0.0)
    vb = _make_unit_vector(0.25)
    vc = _make_unit_vector(0.50)

    sim_ab = float(np.dot(va, vb))
    sim_bc = float(np.dot(vb, vc))
    sim_ac = float(np.dot(va, vc))

    assert sim_ab >= 0.95, f"Expected sim_ab >= 0.95, got {sim_ab}"
    assert sim_bc >= 0.95, f"Expected sim_bc >= 0.95, got {sim_bc}"
    assert sim_ac < 0.95, f"Expected sim_ac < 0.95, got {sim_ac}"

    doc_ids = ["doc_A", "doc_B", "doc_C"]
    embeddings = np.stack([va, vb, vc])

    clusters = find_near_duplicates(doc_ids, embeddings, threshold=0.95)
    assert len(clusters) == 1, f"Expected 1 cluster via transitivity, got {len(clusters)}"
    assert clusters[0] == {"doc_A", "doc_B", "doc_C"}


def test_collapse_duplicates_synthetic_set() -> None:
    """collapse_duplicates() returns correct num_collapsed count on a small synthetic set with one duplicate pair."""
    # doc1 and doc2 are near-duplicates
    v1 = _make_unit_vector(0.0)
    v2 = _make_unit_vector(0.02)  # cos(0.02) ~ 0.9998
    # doc3 is distinct
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    doc_ids = ["doc1", "doc2", "doc3"]
    embeddings = np.stack([v1, v2, v3])
    texts = {
        "doc1": "short",
        "doc2": "a much longer piece of code for the duplicate",
        "doc3": "distinct logic",
    }

    result = collapse_duplicates(doc_ids, embeddings, texts, threshold=0.95)

    assert result["num_clusters"] == 2
    assert result["num_collapsed"] == 1
    # Canonical for {doc1, doc2} should be doc1 (shorter text)
    assert "doc1" in result["canonical_to_versions"]
    assert result["canonical_to_versions"]["doc1"] == ["doc1", "doc2"]
    assert "doc3" in result["canonical_to_versions"]
    assert result["canonical_to_versions"]["doc3"] == ["doc3"]


def test_threshold_read_from_config_changes_behavior() -> None:
    """Threshold is read from config.yaml, not hardcoded (test that changing it changes behavior)."""
    # Angle separation with cos sim = 0.90:
    # Under default config threshold (0.95), these should NOT cluster.
    # Under relaxed config threshold (0.85), these SHOULD cluster.
    v1 = _make_unit_vector(0.0)
    theta = float(np.arccos(0.90))
    v2 = _make_unit_vector(theta)

    sim = float(np.dot(v1, v2))
    assert 0.899 < sim < 0.901

    doc_ids = ["doc1", "doc2"]
    embeddings = np.stack([v1, v2])

    # 1. With default config (0.95 threshold)
    with patch(
        "ariadne.versioning.dedup.load_config",
        return_value={"versioning": {"dedup_cosine_threshold": 0.95}},
    ):
        clusters_strict = find_near_duplicates(doc_ids, embeddings, threshold=None)
        assert len(clusters_strict) == 2

    # 2. With modified config (0.85 threshold)
    with patch(
        "ariadne.versioning.dedup.load_config",
        return_value={"versioning": {"dedup_cosine_threshold": 0.85}},
    ):
        clusters_relaxed = find_near_duplicates(doc_ids, embeddings, threshold=None)
        assert len(clusters_relaxed) == 1
        assert clusters_relaxed[0] == {"doc1", "doc2"}
