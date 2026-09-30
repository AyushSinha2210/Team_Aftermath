"""Dynamic incremental inverted index with single-document insert, update, and delete."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from ariadne.retrieval.code_tokenizer import tokenize_code


class IncrementalSparseIndex:
	"""BM25 inverted index supporting online O(doc_length) updates without full rebuild."""

	def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
		self.k1 = k1
		self.b = b
		self.docs: Dict[str, str] = {}
		self.term_counts: Dict[str, Counter[str]] = {}
		self.lengths: Dict[str, int] = {}
		self.doc_freq: Counter[str] = Counter()
		self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {doc_id: tf}
		self._total_length: int = 0

	@property
	def avg_length(self) -> float:
		n = len(self.docs)
		return (self._total_length / n) if n > 0 else 0.0

	def insert(self, doc_id: str, text: str) -> None:
		"""Inserts a new document into the inverted index."""
		if doc_id in self.docs:
			self.update(doc_id, text)
			return

		tokens = tokenize_code(text)
		counts = Counter(tokens)
		doc_len = sum(counts.values())

		self.docs[doc_id] = text
		self.term_counts[doc_id] = counts
		self.lengths[doc_id] = doc_len
		self._total_length += doc_len

		for term, tf in counts.items():
			self.doc_freq[term] += 1
			self.inverted_index[term][doc_id] = tf

	def remove(self, doc_id: str) -> None:
		"""Removes a document from the inverted index in O(doc_terms) time."""
		if doc_id not in self.docs:
			return

		counts = self.term_counts.pop(doc_id)
		doc_len = self.lengths.pop(doc_id)
		self._total_length -= doc_len
		self.docs.pop(doc_id)

		for term in counts:
			self.doc_freq[term] -= 1
			if self.doc_freq[term] <= 0:
				del self.doc_freq[term]
			if term in self.inverted_index:
				self.inverted_index[term].pop(doc_id, None)
				if not self.inverted_index[term]:
					del self.inverted_index[term]

	def update(self, doc_id: str, new_text: str) -> None:
		"""Updates an existing document by removing old terms and inserting new terms."""
		self.remove(doc_id)
		self.insert(doc_id, new_text)

	def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
		"""Scores documents using BM25 formula over the dynamic postings."""
		if not self.docs or top_k <= 0:
			return []

		query_terms = set(tokenize_code(query))
		n = len(self.docs)
		avg_l = max(self.avg_length, 1.0)
		scores: Dict[str, float] = defaultdict(float)

		for term in query_terms:
			if term not in self.inverted_index:
				continue
			df = self.doc_freq[term]
			idf = math.log1p((n - df + 0.5) / (df + 0.5))
			postings = self.inverted_index[term]

			for doc_id, tf in postings.items():
				doc_len = self.lengths[doc_id]
				length_factor = 1.0 - self.b + self.b * (doc_len / avg_l)
				scores[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * length_factor)

		ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
		return ordered[:top_k]
