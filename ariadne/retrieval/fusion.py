"""Weighted reciprocal rank fusion of retrieval result lists."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    if rrf_k < 0 or (limit is not None and limit < 0):
        raise ValueError("rrf_k and limit must be nonnegative")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(weight < 0 for weight in weights):
        raise ValueError("Supply one nonnegative weight per ranking")
    scores: dict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights):
        seen: set[str] = set()
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += weight / (rrf_k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
    return ordered if limit is None else ordered[:limit]


def min_max_normalize(scores: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalizes ranking scores into the [0.0, 1.0] range.

    Args:
        scores: Sequence of (doc_id, score) pairs.

    Returns:
        Dict mapping doc_id -> normalized score in [0.0, 1.0].
    """
    if not scores:
        return {}

    raw_values = [score for _, score in scores]
    min_val = min(raw_values)
    max_val = max(raw_values)

    if max_val == min_val:
        return {doc_id: 1.0 for doc_id, _ in scores}

    span = max_val - min_val
    return {doc_id: (score - min_val) / span for doc_id, score in scores}


def convex_score_fusion(
    dense_ranking: Sequence[tuple[str, float]],
    sparse_ranking: Sequence[tuple[str, float]],
    alpha: float = 0.85,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Combines normalized dense and sparse scores via convex combination.

    Formula: S(d) = alpha * Dense(d) + (1 - alpha) * Sparse(d)

    Args:
        dense_ranking: Ordered list of (doc_id, dense_score).
        sparse_ranking: Ordered list of (doc_id, bm25_score).
        alpha: Weight for dense score in [0.0, 1.0]. Defaults to 0.85.
        limit: Optional maximum number of results to return.

    Returns:
        Sorted list of (doc_id, fused_score) tuples.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0.0, 1.0]")

    norm_dense = min_max_normalize(dense_ranking)
    norm_sparse = min_max_normalize(sparse_ranking)

    all_ids = set(norm_dense) | set(norm_sparse)
    fused_scores: dict[str, float] = {}

    for doc_id in all_ids:
        d_score = norm_dense.get(doc_id, 0.0)
        s_score = norm_sparse.get(doc_id, 0.0)
        fused_scores[doc_id] = (alpha * d_score) + ((1.0 - alpha) * s_score)

    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered if limit is None else ordered[:limit]
