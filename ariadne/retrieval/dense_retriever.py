"""Cosine-similarity retrieval over Person A's normalized code embeddings."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


def default_encode(texts: list[str]) -> np.ndarray:
    # Import lazily so the index can also be tested with a deterministic stub.
    from ariadne.finetuning.embedder import encode

    return encode(texts)


class DenseRetriever:
    def __init__(self, encoder: Callable[[list[str]], np.ndarray] = default_encode):
        self.encoder = encoder
        self.ids: list[str] = []
        self.texts: dict[str, str] = {}
        self.embeddings: np.ndarray | None = None

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
        vector = np.asarray(self.encoder([query]), dtype=np.float32)
        if vector.shape != (1, self.embeddings.shape[1]):
            raise ValueError("Query embedding dimension differs from corpus embeddings")
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.embeddings @ vector[0]
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in order[:k]]
