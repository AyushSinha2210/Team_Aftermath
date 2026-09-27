"""Unit tests for evolutionary retrieval and cross-version query resolution."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ariadne.versioning.evolutionary_retrieval import (
    rank_across_versions,
    resolve_version_diff,
)


def _mock_encode(texts: list[str]) -> np.ndarray:
    """Mock encode function returning a fixed query unit vector along dimension 0."""
    # Returns [1.0, 0.0, 0.0] for any query
    vecs = np.zeros((len(texts), 3), dtype=np.float32)
    vecs[:, 0] = 1.0
    return vecs


def test_dedup_collapsing_prevents_duplicate_inflation() -> None:
    """rank_across_versions() on a synthetic set with one canonical having 3 near-

    duplicate versions and others having 1 each — confirm the 3-version canonical
    appears ONCE in the output (not 3 times), proving dedup collapsing prevents duplicate inflation.
    """
    canonical_to_versions = {
        "canon_multi": ["canon_multi_v1", "canon_multi_v2", "canon_multi_v3"],
        "canon_single_1": ["canon_single_1"],
        "canon_single_2": ["canon_single_2"],
    }

    embeddings = {
        "canon_multi": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        "canon_single_1": np.array([0.5, 0.5, 0.0], dtype=np.float32),
        "canon_single_2": np.array([0.1, 0.9, 0.0], dtype=np.float32),
    }

    with patch("ariadne.versioning.evolutionary_retrieval.encode", side_effect=_mock_encode):
        results = rank_across_versions("find function", canonical_to_versions, embeddings)

    # Output should contain exactly 3 ranked entries (one per canonical), NOT 5
    assert len(results) == 3

    ranked_ids = [r["canonical_id"] for r in results]
    assert ranked_ids.count("canon_multi") == 1
    assert ranked_ids.count("canon_single_1") == 1
    assert ranked_ids.count("canon_single_2") == 1

    # Verify multi-version canonical preserves all versions
    multi_entry = next(r for r in results if r["canonical_id"] == "canon_multi")
    assert multi_entry["version_count"] == 3
    assert set(multi_entry["all_versions"]) == {
        "canon_multi_v1",
        "canon_multi_v2",
        "canon_multi_v3",
    }


def test_ranking_order_correct_most_similar_first() -> None:
    """Ranking order is correct (most similar canonical first) on a small hand-built case with known order."""
    # Query vector will be [1.0, 0.0, 0.0] via mock encode
    # Embeddings:
    # A has cos sim ~ 0.95
    # B has cos sim ~ 0.70
    # C has cos sim ~ 0.20
    canonical_to_versions = {
        "doc_low": ["doc_low"],
        "doc_high": ["doc_high"],
        "doc_mid": ["doc_mid"],
    }
    embeddings = {
        "doc_high": np.array([0.95, 0.31, 0.0], dtype=np.float32),
        "doc_mid": np.array([0.70, 0.71, 0.0], dtype=np.float32),
        "doc_low": np.array([0.20, 0.98, 0.0], dtype=np.float32),
    }

    with patch("ariadne.versioning.evolutionary_retrieval.encode", side_effect=_mock_encode):
        results = rank_across_versions("search query", canonical_to_versions, embeddings)

    ranked_ids = [r["canonical_id"] for r in results]
    assert ranked_ids == ["doc_high", "doc_mid", "doc_low"]

    scores = [r["score"] for r in results]
    assert scores[0] > scores[1] > scores[2]


def test_resolve_version_diff_identifies_added_and_removed() -> None:
    """resolve_version_diff() correctly identifies newly-added and removed canonicals."""
    old_ranking = [
        {"canonical_id": "doc1", "score": 0.9},
        {"canonical_id": "doc_to_remove", "score": 0.8},
        {"canonical_id": "doc2", "score": 0.7},
    ]

    new_ranking = [
        {"canonical_id": "doc_newly_added", "score": 0.95},
        {"canonical_id": "doc1", "score": 0.85},
        {"canonical_id": "doc2", "score": 0.75},
    ]

    diff = resolve_version_diff(old_ranking, new_ranking)

    assert diff["new_canonicals"] == ["doc_newly_added"]
    assert diff["disappeared_canonicals"] == ["doc_to_remove"]


def test_resolve_version_diff_flags_significant_rank_change() -> None:
    """resolve_version_diff() correctly flags a significant rank change."""
    # doc_drastic was rank 1 in old, drops to rank 4 in new (|1 - 4| = 3 >= 2)
    # doc_stable was rank 2 in old, moves to rank 3 in new (|2 - 3| = 1 < 2)
    old_ranking = [
        {"canonical_id": "doc_drastic", "score": 0.95},  # rank 1
        {"canonical_id": "doc_stable", "score": 0.85},   # rank 2
        {"canonical_id": "doc_other1", "score": 0.75},   # rank 3
        {"canonical_id": "doc_other2", "score": 0.65},   # rank 4
    ]

    new_ranking = [
        {"canonical_id": "doc_other1", "score": 0.90},   # rank 1
        {"canonical_id": "doc_other2", "score": 0.85},   # rank 2
        {"canonical_id": "doc_stable", "score": 0.80},   # rank 3 (delta: 2 - 3 = -1)
        {"canonical_id": "doc_drastic", "score": 0.70},  # rank 4 (delta: 1 - 4 = -3)
    ]

    diff = resolve_version_diff(old_ranking, new_ranking, rank_change_threshold=2)

    flagged_ids = [change["canonical_id"] for change in diff["significant_rank_changes"]]
    assert "doc_drastic" in flagged_ids
    assert "doc_stable" not in flagged_ids

    drastic_change = next(
        c for c in diff["significant_rank_changes"] if c["canonical_id"] == "doc_drastic"
    )
    assert drastic_change["old_rank"] == 1
    assert drastic_change["new_rank"] == 4
    assert drastic_change["rank_delta"] == -3
