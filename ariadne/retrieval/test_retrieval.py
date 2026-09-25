"""Behavior checks for the Person B handoff interface."""

from __future__ import annotations

import numpy as np

from ariadne.retrieval.dense_retriever import DenseRetriever
from ariadne.retrieval.fusion import reciprocal_rank_fusion
from ariadne.retrieval.pipeline import HybridPipeline
from ariadne.retrieval.sparse_retriever import SparseRetriever, tokenize


def fake_encode(texts: list[str]) -> np.ndarray:
    return np.asarray(
        [[float("sort" in text.lower()), float("binary" in text.lower())] for text in texts],
        dtype=np.float32,
    )


def test_code_tokenization_and_bm25() -> None:
    assert "binary" in tokenize("binarySearch")
    assert "binary" in tokenize("binary_search")
    sparse = SparseRetriever()
    sparse.index({"a": "def binary_search(items): pass", "b": "def quicksort(items): pass"})
    assert sparse.retrieve("binary search", 1)[0][0] == "a"
    assert sparse.retrieve("missing term") == []


def test_dense_uses_cosine_and_deterministic_ties() -> None:
    dense = DenseRetriever(fake_encode)
    dense.index({"z": "binary search", "a": "binary tree", "b": "sort"})
    assert [item[0] for item in dense.retrieve("binary", 2)] == ["a", "z"]
    assert dense.retrieve("sort", 1)[0][0] == "b"


def test_rrf_deduplicates_and_uses_both_lists() -> None:
    fused = reciprocal_rank_fusion(
        [[("a", 9), ("b", 1), ("a", 0)], [("b", 7), ("c", 2)]],
        rrf_k=1,
    )
    assert [doc_id for doc_id, _ in fused] == ["b", "a", "c"]
    assert len({doc_id for doc_id, _ in fused}) == 3


def test_pipeline_handoff_shape_and_sources() -> None:
    pipeline = HybridPipeline(
        {"sort": "def quickSort(items): pass", "binary": "def binary_search(items): pass"},
        encoder=fake_encode,
    )
    result = pipeline.retrieve("binary search", k=2)
    assert result[0]["id"] == "binary"
    assert result[0]["sources"] == ["dense", "sparse"]
    assert set(result[0]) >= {"id", "text", "fusion_score", "dense_score", "sparse_score"}
    assert pipeline.retrieve("binary", k=0) == []
