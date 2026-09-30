"""Late-interaction token-level retrieval engine (MaxSim / ColBERT style).

Computes sum-of-maximum inner products across token representations:
    Score(Q, D) = sum_{i in Q} max_{j in D} (E_{q, i} . E_{d, j}^T)

Provides cross-encoder level contextual interaction with bi-encoder pre-indexing speed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np


def compute_maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Computes the MaxSim late-interaction score between query and document token embeddings.

    Args:
        query_tokens: np.ndarray of shape (N_q, D), L2-normalized token embeddings.
        doc_tokens: np.ndarray of shape (N_d, D), L2-normalized token embeddings.

    Returns:
        Float MaxSim relevance score.
    """
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return 0.0

    # Token-to-token cosine similarity matrix: shape (N_q, N_d)
    similarity_matrix = np.matmul(query_tokens, doc_tokens.T)

    # For each query token, take maximum similarity across all document tokens
    max_similarities = np.max(similarity_matrix, axis=1)

    # Sum max similarities across all query tokens
    return float(np.sum(max_similarities))


def rank_by_maxsim(
    query_tokens: np.ndarray,
    corpus_tokens: Sequence[np.ndarray],
    corpus_ids: Sequence[str],
    k: int = 50,
) -> List[Tuple[str, float]]:
    """Ranks corpus documents against query tokens using MaxSim late interaction.

    Args:
        query_tokens: np.ndarray of shape (N_q, D) for the query.
        corpus_tokens: Sequence of np.ndarray of shape (N_d, D) for each document.
        corpus_ids: List of document identifiers matching corpus_tokens.
        k: Maximum number of top documents to return.

    Returns:
        List of (doc_id, maxsim_score) tuples ordered by relevance descending.
    """
    if k <= 0 or not corpus_ids:
        return []

    scores: List[Tuple[str, float]] = []
    for doc_id, doc_toks in zip(corpus_ids, corpus_tokens):
        score = compute_maxsim(query_tokens, doc_toks)
        scores.append((str(doc_id), score))

    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:k]
