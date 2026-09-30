"""Fast document chunk fingerprinting and near-duplicate detection."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set, Tuple


def normalize_code_for_fingerprint(code: str) -> str:
	"""Normalizes code by stripping whitespace, comments, and non-alphanumeric chars."""
	# Remove single-line comments
	no_comments = re.sub(r"(#|//).*$", "", code, flags=re.MULTILINE)
	# Collapse whitespace
	tokens = re.findall(r"[a-zA-Z0-9_]+", no_comments.lower())
	return " ".join(tokens)


def compute_chunk_hash(code: str) -> str:
	"""Computes a 64-bit MD5 hex fingerprint of normalized code."""
	norm = normalize_code_for_fingerprint(code)
	return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


class ChunkDeduplicator:
	"""Rejects duplicate or near-identical code chunks during repository indexing."""

	def __init__(self) -> None:
		self._fingerprints: Dict[str, str] = {}  # fingerprint -> first_doc_id
		self.duplicates_skipped: int = 0

	def filter_unique_chunks(
		self, corpus: Dict[str, str]
	) -> Tuple[Dict[str, str], List[str]]:
		"""Filters a corpus, retaining only the first occurrence of each unique code chunk.

		Args:
			corpus: Dict mapping doc_id -> raw code snippet.

		Returns:
			Tuple of (unique_corpus, list_of_skipped_duplicate_doc_ids).
		"""
		unique_corpus: Dict[str, str] = {}
		skipped_ids: List[str] = []

		for doc_id, text in corpus.items():
			fp = compute_chunk_hash(text)
			if fp in self._fingerprints:
				self.duplicates_skipped += 1
				skipped_ids.append(doc_id)
			else:
				self._fingerprints[fp] = doc_id
				unique_corpus[doc_id] = text

		return unique_corpus, skipped_ids
