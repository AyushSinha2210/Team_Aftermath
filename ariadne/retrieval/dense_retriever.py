"""Cosine-similarity retrieval over normalized code embeddings."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


def default_encode(texts: list[str]) -> np.ndarray:
    # Import lazily so the index can also be tested with a deterministic stub.
    from ariadne.finetuning.embedder import encode

    return encode(texts)


class DenseRetriever:
    def __init__(
        self,
        encoder: Callable[[list[str]], np.ndarray] = default_encode,
        query_cache_size: int = 512,
    ):
        self.encoder = encoder
        self.ids: list[str] = []
        self.texts: dict[str, str] = {}
        self.embeddings: np.ndarray | None = None
        self.query_cache_size = query_cache_size
        self.query_cache: dict[str, np.ndarray] = {}

    def index(self, corpus: Mapping[str, str]) -> None:
        self.ids = list(corpus)
        self.texts = dict(corpus)
        if not self.ids:
            self.embeddings = None
            return
        vectors = np.asarray(self.encoder([corpus[id_] for id_ in self.ids]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.ids):
            raise ValueError("Encoder must return one 2D row per corpus document")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.embeddings = vectors / np.maximum(norms, 1e-12)

    def retrieve(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        if k <= 0 or self.embeddings is None:
            return []

        if query in self.query_cache:
            vector_1d = self.query_cache[query]
        else:
            vector = np.asarray(self.encoder([query]), dtype=np.float32)
            if vector.shape != (1, self.embeddings.shape[1]):
                raise ValueError("Query embedding dimension differs from corpus embeddings")
            vector_1d = vector[0]
            if len(self.query_cache) >= self.query_cache_size:
                # Evict oldest entry
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            self.query_cache[query] = vector_1d

        return self.retrieve_vector(vector_1d, k)

    def retrieve_vector(self, vector: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        if k <= 0 or self.embeddings is None:
            return []
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.embeddings.shape[1],):
            raise ValueError("Query embedding dimension differs from corpus embeddings")
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.embeddings @ vector
        count = min(k, len(self.ids))
        threshold = float(np.partition(scores, -count)[-count])
        candidates = np.flatnonzero(scores >= threshold)
        order = sorted(candidates, key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in order[:count]]

    def retrieve_vectors(self, vectors: np.ndarray, k: int = 50) -> list[list[tuple[str, float]]]:
        """Performs batch retrieval over multiple query vectors via single matrix multiplication."""
        if k <= 0 or self.embeddings is None:
            return []
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[None, :]
        if mat.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension differs from corpus embeddings")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        normed_mat = mat / np.maximum(norms, 1e-12)
        # Batch dot-product: (N_docs, D) @ (D, N_queries) -> (N_docs, N_queries)
        score_matrix = self.embeddings @ normed_mat.T

        results: list[list[tuple[str, float]]] = []
        count = min(k, len(self.ids))
        for q_idx in range(normed_mat.shape[0]):
            scores = score_matrix[:, q_idx]
            threshold = float(np.partition(scores, -count)[-count])
            candidates = np.flatnonzero(scores >= threshold)
            order = sorted(candidates, key=lambda i: (-float(scores[i]), self.ids[i]))
            results.append([(self.ids[i], float(scores[i])) for i in order[:count]])
        return results

    def retrieve_batch(self, queries: list[str], k: int = 50) -> list[list[tuple[str, float]]]:
        """Encodes and retrieves multiple queries in a single vectorized batch."""
        if not queries or k <= 0 or self.embeddings is None:
            return []
        vectors = np.asarray(self.encoder(queries), dtype=np.float32)
        return self.retrieve_vectors(vectors, k=k)

