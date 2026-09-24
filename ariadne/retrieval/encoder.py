"""MTEB dense encoder contract for AppsRetrieval (MTEB 2.21.3)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta


class PrePostPipelineEncoder(AbsEncoder):
    """Wrap Person A's encoder for MTEB's batch-oriented embedding API.

    BM25 and rank fusion cannot be expressed as independent embeddings. The
    hybrid SearchProtocol adapter lives in ``mteb_search.py``.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        embed: Callable[..., np.ndarray] | None = None,
    ) -> None:
        if embed is None:
            from ariadne.finetuning.embedder import encode

            embed = encode
        self.embed = embed
        self.model_name_or_path = model_name_or_path
        self.mteb_model_meta = ModelMeta.create_empty(
            {"name": "ariadne/dense", "revision": "local", "framework": ["Sentence Transformers"]}
        )

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type=None,
        **kwargs,
    ) -> np.ndarray:
        texts = [str(text) for batch in inputs for text in batch["text"]]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encode_kwargs = {"batch_size": int(kwargs.get("batch_size", 32))}
        if self.model_name_or_path is not None:
            encode_kwargs["model_name_or_path"] = self.model_name_or_path
        vectors = np.asarray(self.embed(texts, **encode_kwargs), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding backend returned the wrong number of rows")
        return vectors
