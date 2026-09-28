"""MTEB dense encoder contract for AppsRetrieval (compatible with mteb==1.12.50)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Union

import numpy as np

Corpus = Union[List[Dict[str, str]], Dict[str, List[str]], Dict[str, Dict[str, str]], Sequence[str]]


class PrePostPipelineEncoder:
    """PrePostPipelineEncoder implementing MTEB 1.12.50's EncoderWithQueryCorpusEncode protocol.

    Wraps Person A's fine-tuned bi-encoder for dense retrieval evaluation without
    inheriting from AbsEncoder, satisfying structural typing via Protocol.
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
        self.mteb_model_meta = None
        try:
            from mteb.model_meta import ModelMeta

            self.mteb_model_meta = ModelMeta.create_empty(
                {"name": "ariadne/dense", "revision": "local", "framework": ["Sentence Transformers"]}
            )
        except Exception:
            pass

    def encode(
        self,
        sentences: Sequence[str],
        *,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes given sentences using Person A's embedder."""
        if not sentences:
            return np.empty((0, 384), dtype=np.float32)
        texts = [str(s) for s in sentences]
        encode_kwargs: dict[str, Any] = {"batch_size": int(kwargs.get("batch_size", 32))}
        if self.model_name_or_path is not None:
            encode_kwargs["model_name_or_path"] = self.model_name_or_path
        vectors = np.asarray(self.embed(texts, **encode_kwargs), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding backend returned the wrong number of rows")
        return vectors

    def encode_queries(
        self,
        queries: Sequence[str],
        *,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes retrieval queries."""
        return self.encode(queries, prompt_name=prompt_name, **kwargs)

    def encode_corpus(
        self,
        corpus: Corpus,
        *,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes corpus documents, extracting text/title from dictionary records."""
        if not corpus:
            return np.empty((0, 384), dtype=np.float32)
        if isinstance(corpus, list):
            sentences = [
                (f"{doc.get('title', '')} {doc.get('text', '')}").strip()
                if isinstance(doc, dict) and doc.get("title")
                else (doc.get("text", "") if isinstance(doc, dict) else str(doc))
                for doc in corpus
            ]
        elif isinstance(corpus, dict):
            if "text" in corpus:
                texts = corpus["text"]
                titles = corpus.get("title", [""] * len(texts))
                sentences = [
                    (f"{t} {txt}").strip() if t else str(txt)
                    for t, txt in zip(titles, texts)
                ]
            else:
                sentences = [
                    (f"{doc.get('title', '')} {doc.get('text', '')}").strip()
                    if isinstance(doc, dict) and doc.get("title")
                    else (doc.get("text", "") if isinstance(doc, dict) else str(doc))
                    for doc in corpus.values()
                ]
        else:
            sentences = [str(item) for item in corpus]
        return self.encode(sentences, prompt_name=prompt_name, **kwargs)

