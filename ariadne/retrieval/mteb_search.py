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
            documents[doc_id] = "\n".join(
                str(row[key]) for key in ("title", "body") if row.get(key)
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
        for row in queries:
            query_id = str(row["id"])
            search_k = len(self.pipeline.corpus) if top_ranked is not None else top_k
            candidates = self.pipeline.retrieve(str(row["text"]), k=search_k)
            if top_ranked is not None:
                allowed = set(top_ranked.get(query_id, []))
                candidates = [candidate for candidate in candidates if candidate["id"] in allowed][:top_k]
            results[query_id] = {
                str(candidate["id"]): float(candidate["fusion_score"])
                for candidate in candidates
            }
        return results
