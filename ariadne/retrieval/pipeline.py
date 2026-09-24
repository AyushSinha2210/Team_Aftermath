"""Hybrid code search. Person C can call ``pipeline.retrieve(query, k=50)``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import yaml

from ariadne.retrieval.dense_retriever import DenseRetriever, default_encode
from ariadne.retrieval.fusion import reciprocal_rank_fusion
from ariadne.retrieval.sparse_retriever import SparseRetriever


class HybridPipeline:
    def __init__(
        self,
        corpus: Mapping[str, str],
        *,
        encoder: Callable[[list[str]], np.ndarray] = default_encode,
        config_path: Path | None = None,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["retrieval"]
        self.config = config
        self.corpus = dict(corpus)
        self.dense = DenseRetriever(encoder)
        self.sparse = SparseRetriever(config["bm25_k1"], config["bm25_b"])
        self.dense.index(self.corpus)
        self.sparse.index(self.corpus)

    def retrieve(
        self, query: str, k: int = 50, *, dense_vector: np.ndarray | None = None
    ) -> list[dict[str, object]]:
        if k <= 0:
            return []
        dense_k = max(k, int(self.config["dense_top_k"]))
        dense = (
            self.dense.retrieve(query, dense_k)
            if dense_vector is None
            else self.dense.retrieve_vector(dense_vector, dense_k)
        )
        sparse = self.sparse.retrieve(query, max(k, int(self.config["sparse_top_k"])))
        fused = reciprocal_rank_fusion(
            [dense, sparse],
            weights=[float(self.config["dense_weight"]), float(self.config["sparse_weight"])],
            rrf_k=int(self.config["rrf_k"]),
            limit=k,
        )
        dense_scores, sparse_scores = dict(dense), dict(sparse)
        return [
            {
                "id": doc_id,
                "text": self.corpus[doc_id],
                "fusion_score": score,
                "dense_score": dense_scores.get(doc_id),
                "sparse_score": sparse_scores.get(doc_id),
                "sources": [source for source, found in (("dense", doc_id in dense_scores), ("sparse", doc_id in sparse_scores)) if found],
            }
            for doc_id, score in fused
        ]
