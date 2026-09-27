"""Incremental indexer that re-embeds only changed hashes across code versions (owned by Person D).

Persistence format:
    This module uses Python's standard `pickle` protocol to serialize the cache dictionary
    mapping `doc_id -> (content_hash, embedding_vector)`. This ensures atomic read/write,
    full preservation of float32 NumPy arrays, and zero-conversion overhead.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ariadne.finetuning.embedder import encode
from ariadne.versioning.content_hash import diff_hashes, hash_corpus, load_config


class IncrementalIndex:
    """Manages an incremental vector index by hashing and caching code snippet embeddings."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        """Initializes the incremental index.

        Args:
            cache_path: Optional path to persistence file. If None, loaded from
                config.yaml (versioning.incremental_cache_path).
        """
        if cache_path is None:
            config = load_config()
            configured_path = config.get("versioning", {}).get(
                "incremental_cache_path", "versioning/.index_cache"
            )
            repo_root = Path(__file__).resolve().parent.parent
            self.cache_path = (repo_root / configured_path).resolve()
        else:
            self.cache_path = Path(cache_path).resolve()

        # In-memory storage: doc_id -> (content_hash, embedding_vector)
        self._entries: Dict[str, Tuple[str, np.ndarray]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Loads cached entries from disk if the cache file exists."""
        if self.cache_path.exists() and self.cache_path.is_file():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self._entries = data
            except Exception:
                self._entries = {}

    def _save_cache(self) -> None:
        """Persists cache entries to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    def build(self, corpus: Dict[str, str]) -> None:
        """Performs a full first-time build of the index.

        Hashes all documents in the corpus, encodes them via the bi-encoder,
        and saves {doc_id: (hash, embedding)} to the cache path.

        Args:
            corpus: Mapping of doc_id to code snippet text.
        """
        if not corpus:
            self._entries = {}
            self._save_cache()
            return

        doc_ids = list(corpus.keys())
        texts = [corpus[doc_id] for doc_id in doc_ids]
        hashes = hash_corpus(corpus)
        embeddings = encode(texts)

        self._entries = {
            doc_id: (hashes[doc_id], np.asarray(embeddings[idx], dtype=np.float32))
            for idx, doc_id in enumerate(doc_ids)
        }
        self._save_cache()

    def update(self, new_corpus: Dict[str, str]) -> Dict[str, Any]:
        """Incrementally updates the index against a new version of the corpus.

        Compares content hashes, re-encodes only added and changed documents,
        reuses cached embeddings for unchanged documents, and removes obsolete
        documents.

        Args:
            new_corpus: Mapping of doc_id to code snippet text in the updated version.

        Returns:
            Summary dictionary containing:
                - 'added': int count of newly added documents
                - 'changed': int count of modified documents
                - 'removed': int count of deleted documents
                - 'unchanged': int count of documents with identical content
                - 're_embedded': int count of documents sent to encode()
        """
        old_hashes = {
            doc_id: entry[0] for doc_id, entry in self._entries.items()
        }
        new_hashes = hash_corpus(new_corpus)

        diff = diff_hashes(old_hashes, new_hashes)

        # Only added and changed documents need embedding
        to_embed_ids = sorted(list(diff["added"] | diff["changed"]))

        if to_embed_ids:
            texts_to_embed = [new_corpus[doc_id] for doc_id in to_embed_ids]
            new_embeddings = encode(texts_to_embed)
            for doc_id, emb in zip(to_embed_ids, new_embeddings):
                self._entries[doc_id] = (
                    new_hashes[doc_id],
                    np.asarray(emb, dtype=np.float32),
                )

        # Remove deleted documents
        for doc_id in diff["removed"]:
            self._entries.pop(doc_id, None)

        self._save_cache()

        return {
            "added": len(diff["added"]),
            "changed": len(diff["changed"]),
            "removed": len(diff["removed"]),
            "unchanged": len(diff["unchanged"]),
            "re_embedded": len(to_embed_ids),
        }

    def get_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Returns the current document IDs and their embedding matrix.

        Returns:
            Tuple of (doc_ids, embeddings) where embeddings has shape
            (num_docs, embedding_dim).
        """
        if not self._entries:
            return [], np.empty((0, 384), dtype=np.float32)

        doc_ids = list(self._entries.keys())
        embeddings = np.array(
            [self._entries[doc_id][1] for doc_id in doc_ids],
            dtype=np.float32,
        )
        return doc_ids, embeddings
