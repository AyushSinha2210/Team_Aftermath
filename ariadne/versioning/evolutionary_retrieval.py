"""Evolutionary retrieval across multiple codebase versions and commit histories.

This module provides cross-version query resolution and ranking without duplicate inflation
by querying canonical deduplicated cluster representatives, alongside version-diff ranking analytics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ariadne.finetuning.embedder import encode


def rank_across_versions(
    query: str,
    canonical_to_versions: Dict[str, List[str]],
    embeddings: Dict[str, np.ndarray],
    version_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Ranks canonical code representatives across versions against a query.

    Instead of scoring every version individually (which would cause near-duplicate
    inflation in search results), this scores only the canonical representative of each
    cluster, while preserving full lineage back to all historical/branch versions.

    Args:
        query: Natural language or code search query string.
        canonical_to_versions: Mapping from canonical doc_id to list of all member doc_ids.
        embeddings: Mapping from doc_id to its embedding vector.
        version_metadata: Optional mapping from doc_id to additional metadata dict.

    Returns:
        List of ranked canonical result dicts ordered by cosine similarity descending:
            [
                {
                    "canonical_id": str,
                    "score": float,
                    "all_versions": List[str],
                    "version_count": int,
                    "metadata": Dict[str, Any],  # if version_metadata provided
                },
                ...
            ]
    """
    if not canonical_to_versions:
        return []

    # Filter canonicals that exist in embeddings
    valid_canonicals = [cid for cid in canonical_to_versions if cid in embeddings]
    if not valid_canonicals:
        return []

    # Encode query using the project bi-encoder
    query_emb = encode([query])
    query_vec = np.asarray(query_emb[0], dtype=np.float32)
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm > 0:
        query_vec = query_vec / q_norm

    # Stack canonical embeddings and compute vectorized cosine similarities
    canonical_matrix = np.array(
        [embeddings[cid] for cid in valid_canonicals],
        dtype=np.float32,
    )
    # Ensure canonical vectors are normalized
    c_norms = np.linalg.norm(canonical_matrix, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1.0
    canonical_matrix = canonical_matrix / c_norms

    scores = canonical_matrix @ query_vec

    results: List[Dict[str, Any]] = []
    for cid, score in zip(valid_canonicals, scores):
        all_versions = canonical_to_versions.get(cid, [cid])
        item: Dict[str, Any] = {
            "canonical_id": cid,
            "score": float(score),
            "all_versions": all_versions,
            "version_count": len(all_versions),
        }
        if version_metadata is not None:
            item["metadata"] = version_metadata.get(cid, {})
        results.append(item)

    # Sort descending by cosine similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def resolve_version_diff(
    old_ranking: List[Dict[str, Any]],
    new_ranking: List[Dict[str, Any]],
    rank_change_threshold: int = 2,
) -> Dict[str, Any]:
    """Compares cross-version retrieval rankings between two points in history.

    Identifies newly added canonicals, disappeared canonicals, and canonicals whose
    ranking position shifted significantly (>= rank_change_threshold positions).

    Rank positions are 1-indexed (position 1 is highest scoring).

    Args:
        old_ranking: Ranking list from rank_across_versions() on prior version.
        new_ranking: Ranking list from rank_across_versions() on current version.
        rank_change_threshold: Minimum absolute rank shift to be flagged as significant.
            Default is 2 (e.g. moving from rank 1 to rank 3 or vice-versa).

    Returns:
        Dictionary containing:
            - 'new_canonicals': List[str] IDs present in new_ranking but not old_ranking.
            - 'disappeared_canonicals': List[str] IDs present in old_ranking but not new_ranking.
            - 'significant_rank_changes': List[Dict] with canonical_id, old_rank, new_rank, rank_delta.
            - 'rank_change_threshold': int threshold applied.
    """
    old_positions: Dict[str, int] = {
        item["canonical_id"]: idx + 1 for idx, item in enumerate(old_ranking)
    }
    new_positions: Dict[str, int] = {
        item["canonical_id"]: idx + 1 for idx, item in enumerate(new_ranking)
    }

    old_ids = set(old_positions.keys())
    new_ids = set(new_positions.keys())

    new_canonicals = sorted(list(new_ids - old_ids))
    disappeared_canonicals = sorted(list(old_ids - new_ids))

    # Track significant rank changes for canonicals existing in both
    common_ids = old_ids & new_ids
    significant_rank_changes: List[Dict[str, Any]] = []

    for cid in common_ids:
        old_rank = old_positions[cid]
        new_rank = new_positions[cid]
        rank_delta = old_rank - new_rank  # Positive means improved rank, negative means dropped
        if abs(rank_delta) >= rank_change_threshold:
            significant_rank_changes.append(
                {
                    "canonical_id": cid,
                    "old_rank": old_rank,
                    "new_rank": new_rank,
                    "rank_delta": rank_delta,
                }
            )

    # Sort significant rank changes by largest absolute delta descending
    significant_rank_changes.sort(key=lambda x: abs(x["rank_delta"]), reverse=True)

    return {
        "new_canonicals": new_canonicals,
        "disappeared_canonicals": disappeared_canonicals,
        "significant_rank_changes": significant_rank_changes,
        "rank_change_threshold": rank_change_threshold,
    }
