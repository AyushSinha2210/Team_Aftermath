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
