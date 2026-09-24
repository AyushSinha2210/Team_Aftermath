"""MTEB SearchProtocol adapter for the actual dense + BM25 + RRF pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from mteb.models.model_meta import ModelMeta

from ariadne.retrieval.dense_retriever import default_encode
from ariadne.retrieval.pipeline import HybridPipeline


class HybridSearchModel:
    def __init__(self, encoder: Callable[[list[str]], np.ndarray] = default_encode):
        self.encoder = encoder
        self.pipeline: HybridPipeline | None = None
        self.mteb_model_meta = ModelMeta.create_empty(
            {"name": "ariadne/hybrid-rrf", "revision": "local", "framework": ["Sentence Transformers"], "model_type": ["hybrid"]}
        )

    def index(
        self,
        corpus,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs,
        num_proc: int | None,
    ) -> None:
        documents: dict[str, str] = {}
        for row in corpus:
            doc_id = str(row["id"])
            if doc_id in documents:
                raise ValueError(f"Duplicate corpus ID: {doc_id}")
            body = row.get("body") or row.get("text")
            if not isinstance(body, str) or not body:
                raise ValueError(f"Corpus document {doc_id} has no code text")
            documents[doc_id] = "\n".join(
                text for text in (row.get("title"), body) if text
            )
        self.pipeline = HybridPipeline(documents, encoder=self.encoder)

    def search(
        self,
        queries,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs,
        top_ranked=None,
        num_proc: int | None,
    ) -> dict[str, dict[str, float]]:
        if self.pipeline is None:
            raise RuntimeError("Call index before search")
        results: dict[str, dict[str, float]] = {}
        query_rows = list(queries)
        if not query_rows:
            return results
        query_vectors = np.asarray(
            self.encoder([str(row["text"]) for row in query_rows]), dtype=np.float32
        )
        if query_vectors.ndim != 2 or query_vectors.shape[0] != len(query_rows):
            raise ValueError("Encoder must return one embedding per query")
        for row, vector in zip(query_rows, query_vectors):
            query_id = str(row["id"])
            search_k = len(self.pipeline.corpus) if top_ranked is not None else top_k
            candidates = self.pipeline.retrieve(str(row["text"]), k=search_k, dense_vector=vector)
            if top_ranked is not None:
                allowed = set(top_ranked.get(query_id, []))
                candidates = [candidate for candidate in candidates if candidate["id"] in allowed][:top_k]
            results[query_id] = {
                str(candidate["id"]): float(candidate["fusion_score"])
                for candidate in candidates
            }
        return results
