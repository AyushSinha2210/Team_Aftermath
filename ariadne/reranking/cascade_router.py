"""Confidence-gated dynamic cascade router for adaptive retrieval and reranking.

Inspects top bi-encoder candidate margins (Score_1 - Score_2). High-confidence queries
bypass expensive/noisy cross-encoder reranking, preserving Config A's high NDCG.
Ambiguous queries (margin < tau) are escalated to second-tier rerankers.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ariadne.reranking.cross_encoder import rerank


class CascadeRouter:
    """Routes retrieval candidates dynamically based on dense confidence margins."""

    def __init__(
        self,
        confidence_threshold: float = 0.05,
        rerank_fn: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> None:
        """Initializes the cascade router.

        Args:
            confidence_threshold: Minimum margin (Score_1 - Score_2) to bypass reranking.
            rerank_fn: Optional callable for reranking. Defaults to cross_encoder.rerank.
        """
        self.confidence_threshold = confidence_threshold
        self.rerank_fn = rerank_fn or rerank

    def compute_margin(self, candidates: List[Dict[str, Any]]) -> float:
        """Calculates the top-2 score margin from candidate scores.

        Args:
            candidates: Ranked list of candidate dictionaries with score fields.

        Returns:
            Float margin >= 0.0. If fewer than 2 candidates, returns infinity (bypass).
        """
        if len(candidates) < 2:
            return float("inf")

        def _get_score(candidate: Dict[str, Any]) -> float:
            for key in ("dense_score", "fusion_score", "score", "sim"):
                if key in candidate and candidate[key] is not None:
                    return float(candidate[key])
            return 0.0

        top1 = _get_score(candidates[0])
        top2 = _get_score(candidates[1])
        return max(0.0, top1 - top2)

    def route(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        score_cache: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Dict[str, Any]:
        """Dynamically routes candidates through fast-path or reranking tier.

        Args:
            query: User search query.
            candidates: Candidate list from dense or hybrid retrieval.
            score_cache: Optional score cache for the reranker.

        Returns:
            Dict containing:
                - 'candidates': Final ordered candidate list.
                - 'decision': 'fast_path' or 'escalated'.
                - 'margin': float score margin between top-1 and top-2.
                - 'latency_ms': Routing execution time in milliseconds.
        """
        start_time = time.perf_counter()
        if not candidates:
            return {
                "candidates": [],
                "decision": "fast_path",
                "margin": float("inf"),
                "latency_ms": (time.perf_counter() - start_time) * 1000,
            }

        margin = self.compute_margin(candidates)

        # High confidence -> Fast-path bypass
        if margin >= self.confidence_threshold:
            return {
                "candidates": candidates,
                "decision": "fast_path",
                "margin": margin,
                "latency_ms": (time.perf_counter() - start_time) * 1000,
            }

        # Low confidence -> Escalate to reranker
        try:
            reranked = self.rerank_fn(query, candidates, score_cache=score_cache)
        except TypeError:
            reranked = self.rerank_fn(query, candidates)

        return {
            "candidates": reranked,
            "decision": "escalated",
            "margin": margin,
            "latency_ms": (time.perf_counter() - start_time) * 1000,
        }
