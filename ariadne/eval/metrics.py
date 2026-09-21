"""Retrieval evaluation metrics module.

Provides vectorized calculation of NDCG@k, MRR@k, and Recall@k
mirroring the official MTEB / BEIR benchmark scoring specifications.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Set

import numpy as np


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configures structured logger for metrics calculation.

    Args:
        log_level: Desired logging verbosity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("ariadne.eval.metrics")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def compute_mrr_at_k(ranks: List[int], k: int = 10) -> float:
    """Computes Mean Reciprocal Rank (MRR@k).

    Args:
        ranks: List of 1-based ranks of the first relevant document for each query.
               Use rank > k (or 0) if no relevant document was found within top-k.
        k: Cutoff threshold.

    Returns:
        Float MRR score in [0.0, 1.0].
    """
    if not ranks:
        return 0.0

    reciprocal_ranks = [
        1.0 / rank if (1 <= rank <= k) else 0.0
        for rank in ranks
    ]
    return float(np.mean(reciprocal_ranks))


def compute_ndcg_at_k(ranks: List[int], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG@k) for single-positive relevance.

    For datasets with binary relevance and single ground-truth target per query,
    IDCG@k is 1.0 (at rank 1), so NDCG@k simplifies to 1.0 / log2(rank + 1).

    Args:
        ranks: List of 1-based ranks of the relevant document for each query.
        k: Cutoff threshold.

    Returns:
        Float NDCG@k score in [0.0, 1.0].
    """
    if not ranks:
        return 0.0

    ndcg_scores = [
        1.0 / np.log2(rank + 1) if (1 <= rank <= k) else 0.0
        for rank in ranks
    ]
    return float(np.mean(ndcg_scores))


def compute_recall_at_k(ranks: List[int], k: int = 10) -> float:
    """Computes Recall@k (hit rate in top-k).

    Args:
        ranks: List of 1-based ranks of the relevant document for each query.
        k: Cutoff threshold.

    Returns:
        Float Recall@k score in [0.0, 1.0].
    """
    if not ranks:
        return 0.0

    hits = [1.0 if (1 <= rank <= k) else 0.0 for rank in ranks]
    return float(np.mean(hits))


def evaluate_ranking(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
    qrels: Dict[str, Set[str]],
    top_k: int = 10,
) -> Dict[str, float]:
    """Evaluates retrieval accuracy across all queries against candidate corpus embeddings.

    Computes cosine similarities via matrix multiplication on normalized vectors,
    identifies top ranks, and outputs standard evaluation metrics.

    Args:
        query_embeddings: np.ndarray of shape (N_queries, D), L2-normalized.
        corpus_embeddings: np.ndarray of shape (N_corpus, D), L2-normalized.
        query_ids: List of N_queries query IDs.
        corpus_ids: List of N_corpus corpus document IDs.
        qrels: Mapping from query ID to set of relevant corpus document IDs.
        top_k: Maximum cutoff threshold for metrics.

    Returns:
        Dictionary mapping metric names (e.g. 'ndcg@10', 'mrr@10') to float values.
    """
    num_queries = len(query_ids)
    if num_queries == 0 or len(corpus_ids) == 0:
        return {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@1": 0.0, "recall@10": 0.0}

    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
    ranks: List[int] = []

    # Batch cosine similarity computation: Q @ C.T
    similarity_matrix = np.matmul(query_embeddings, corpus_embeddings.T)

    for i, qid in enumerate(query_ids):
        relevant_cids = qrels.get(qid, set())
        if not relevant_cids:
            continue

        relevant_indices = {
            corpus_id_to_idx[cid] for cid in relevant_cids if cid in corpus_id_to_idx
        }
        if not relevant_indices:
            ranks.append(top_k + 1)
            continue

        query_sims = similarity_matrix[i]
        top_candidates = np.argsort(-query_sims)

        found_rank = top_k + 1
        for rank_idx, doc_idx in enumerate(top_candidates[:top_k], start=1):
            if doc_idx in relevant_indices:
                found_rank = rank_idx
                break
        ranks.append(found_rank)

    return {
        "ndcg@10": compute_ndcg_at_k(ranks, k=10),
        "mrr@10": compute_mrr_at_k(ranks, k=10),
        "recall@1": compute_recall_at_k(ranks, k=1),
        "recall@5": compute_recall_at_k(ranks, k=5),
        "recall@10": compute_recall_at_k(ranks, k=10),
    }
