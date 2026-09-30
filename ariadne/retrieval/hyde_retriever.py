"""HyDE-augmented dense retriever for code retrieval."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from ariadne.retrieval.code_hyde import average_hyde_embeddings, expand_hyde_queries


class CodeHyDERetriever:
	"""Wraps a dense vector retriever with hypothetical code generation and multi-representation centroid search."""

	def __init__(self, base_dense_retriever: Any) -> None:
		self.base_retriever = base_dense_retriever

	def retrieve(
		self,
		query: str,
		top_k: int = 10,
		use_hyde: bool = True,
	) -> List[Tuple[str, float]]:
		"""Performs dense retrieval with optional Code HyDE centroid expansion.

		Args:
			query: Natural language query.
			top_k: Number of candidates to return.
			use_hyde: If True, expands query into code hypotheses; if False, uses standard query.

		Returns:
			List of (doc_id, score) pairs.
		"""
		if not use_hyde:
			return self.base_retriever.retrieve(query, top_k=top_k)

		queries = expand_hyde_queries(query)
		if not queries:
			return []

		model = getattr(self.base_retriever, "model", None)
		if model is not None and hasattr(model, "encode"):
			embeddings = [
				np.asarray(model.encode(q), dtype=float)
				for q in queries
			]
			centroid = average_hyde_embeddings(embeddings)
			if hasattr(self.base_retriever, "retrieve_by_vector"):
				return self.base_retriever.retrieve_by_vector(centroid, top_k=top_k)

		# Fallback to standard base retrieval if vector-level query is not supported
		return self.base_retriever.retrieve(query, top_k=top_k)
