"""Tombstone compaction and garbage collection for incremental index stores."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


class TombstoneCompactor:
	"""Purges deleted document tombstones and defragments vector stores."""

	def __init__(self) -> None:
		self.tombstones: Set[str] = set()

	def mark_deleted(self, doc_id: str) -> None:
		"""Marks a document ID as deleted."""
		self.tombstones.add(doc_id)

	def is_deleted(self, doc_id: str) -> bool:
		"""Checks if a document ID is tombstoned."""
		return doc_id in self.tombstones

	def compact(
		self,
		entries: Dict[str, Tuple[str, np.ndarray]],
	) -> Tuple[Dict[str, Tuple[str, np.ndarray]], int]:
		"""Compacts index entries by purging all tombstoned documents.

		Args:
			entries: Dict mapping doc_id -> (content_hash, embedding_vector).

		Returns:
			Tuple of (compacted_entries_dict, purged_count).
		"""
		purged_count = 0
		compacted: Dict[str, Tuple[str, np.ndarray]] = {}

		for doc_id, val in entries.items():
			if doc_id in self.tombstones:
				purged_count += 1
			else:
				compacted[doc_id] = val

		# Reset tombstones after successful compaction
		self.tombstones.clear()
		return compacted, purged_count
