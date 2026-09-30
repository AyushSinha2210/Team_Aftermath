"""Memory footprint profiling for dense vector stores and sparse inverted indices."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class MemoryReport:
	dense_embeddings_bytes: int
	dense_embeddings_mb: float
	sparse_postings_bytes: int
	sparse_postings_mb: float
	total_mb: float
	num_documents: int
	embedding_dim: int


def _get_deep_size(obj: Any) -> int:
	"""Approximates memory size in bytes of nested Python containers."""
	size = sys.getsizeof(obj)
	if isinstance(obj, dict):
		size += sum(_get_deep_size(k) + _get_deep_size(v) for k, v in obj.items())
	elif isinstance(obj, (list, tuple, set)):
		size += sum(_get_deep_size(i) for i in obj)
	return size


def profile_index_memory(
	dense_retriever: Any,
	sparse_retriever: Any,
) -> MemoryReport:
	"""Calculates memory footprint in bytes and megabytes for dense and sparse indices.

	Args:
		dense_retriever: DenseRetriever instance with .embeddings attribute.
		sparse_retriever: SparseRetriever instance with .postings attribute.

	Returns:
		MemoryReport dataclass.
	"""
	# 1. Dense vector store
	dense_bytes = 0
	dim = 0
	n_docs = 0
	emb = getattr(dense_retriever, "embeddings", None)
	if emb is not None and hasattr(emb, "nbytes"):
		dense_bytes = int(emb.nbytes)
		if emb.ndim == 2:
			n_docs, dim = emb.shape

	# 2. Sparse postings
	postings = getattr(sparse_retriever, "postings", {})
	sparse_bytes = _get_deep_size(postings)

	if n_docs == 0:
		ids = getattr(sparse_retriever, "ids", [])
		n_docs = len(ids)

	dense_mb = round(dense_bytes / (1024 * 1024), 3)
	sparse_mb = round(sparse_bytes / (1024 * 1024), 3)
	total_mb = round(dense_mb + sparse_mb, 3)

	return MemoryReport(
		dense_embeddings_bytes=dense_bytes,
		dense_embeddings_mb=dense_mb,
		sparse_postings_bytes=sparse_bytes,
		sparse_postings_mb=sparse_mb,
		total_mb=total_mb,
		num_documents=n_docs,
		embedding_dim=dim,
	)
