"""Unit tests for ariadne.versioning.incremental_index."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ariadne.versioning.incremental_index import IncrementalIndex


def _mock_encode(texts: list[str], **kwargs) -> np.ndarray:
    """Deterministic mock embedder returning vectors based on string length."""
    vectors = []
    for text in texts:
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = float(len(text))
        vectors.append(vec)
    return np.asarray(vectors, dtype=np.float32)


def test_build_produces_correct_cache(tmp_path: Path) -> None:
    """Verifies that build() encodes the corpus and creates a persistent cache."""
    cache_file = tmp_path / "test_cache.pkl"
    index = IncrementalIndex(cache_path=cache_file)

    corpus = {
        "doc1": "def func_a(): return 1",
        "doc2": "def func_b(): return 2",
        "doc3": "def func_c(): return 3",
    }

    with patch("ariadne.versioning.incremental_index.encode", side_effect=_mock_encode) as mock_encode:
        index.build(corpus)
        assert mock_encode.call_count == 1
        # Should pass all 3 document texts
        args, _ = mock_encode.call_args
        assert len(args[0]) == 3

    assert cache_file.exists()
    doc_ids, embeddings = index.get_embeddings()
    assert set(doc_ids) == {"doc1", "doc2", "doc3"}
    assert embeddings.shape == (3, 384)


def test_update_reuses_unchanged_and_only_embeds_changed_and_added(tmp_path: Path) -> None:
    """Verifies that update() only calls encode() on added and changed docs, NOT unchanged ones."""
    cache_file = tmp_path / "test_cache.pkl"
    index = IncrementalIndex(cache_path=cache_file)

    initial_corpus = {
        "doc_unchanged": "def unchanged(): return 100",
        "doc_to_change": "def change_me(): return 'v1'",
        "doc_to_remove": "def remove_me(): return 'old'",
    }

    with patch("ariadne.versioning.incremental_index.encode", side_effect=_mock_encode):
        index.build(initial_corpus)

    updated_corpus = {
        "doc_unchanged": "def unchanged(): return 100",  # unchanged
        "doc_to_change": "def change_me(): return 'v2'",  # changed
        "doc_new": "def brand_new(): return 'fresh'",     # added
    }

    with patch("ariadne.versioning.incremental_index.encode", side_effect=_mock_encode) as mock_encode:
        summary = index.update(updated_corpus)

        assert summary == {
            "added": 1,
            "changed": 1,
            "removed": 1,
            "unchanged": 1,
            "re_embedded": 2,
        }

        # encode() must be called exactly once for the 2 re-embedded docs
        assert mock_encode.call_count == 1
        encoded_texts = mock_encode.call_args[0][0]
        assert len(encoded_texts) == 2
        # The unchanged document text must NOT be sent to encode()
        assert initial_corpus["doc_unchanged"] not in encoded_texts
        assert updated_corpus["doc_to_change"] in encoded_texts
        assert updated_corpus["doc_new"] in encoded_texts


def test_update_removes_obsolete_documents(tmp_path: Path) -> None:
    """Verifies that docs absent from new_corpus are pruned from the index."""
    cache_file = tmp_path / "test_cache.pkl"
    index = IncrementalIndex(cache_path=cache_file)

    corpus = {
        "keep": "def keep(): pass",
        "drop": "def drop(): pass",
    }

    with patch("ariadne.versioning.incremental_index.encode", side_effect=_mock_encode):
        index.build(corpus)

    new_corpus = {"keep": "def keep(): pass"}

    with patch("ariadne.versioning.incremental_index.encode", side_effect=_mock_encode) as mock_encode:
        summary = index.update(new_corpus)
        assert summary["removed"] == 1
        assert summary["re_embedded"] == 0
        mock_encode.assert_not_called()

    doc_ids, embeddings = index.get_embeddings()
    assert doc_ids == ["keep"]
    assert embeddings.shape == (1, 384)


def test_get_embeddings_consistency_after_persistence(tmp_path: Path) -> None:
    """Verifies that a re-instantiated IncrementalIndex correctly loads existing state."""
    cache_file = tmp_path / "test_cache.pkl"
    index1 = IncrementalIndex(cache_path=cache_file)

    corpus = {
        "a": "def a(): pass",
        "b": "def b(): pass",
    }
    with patch("ariadne.versioning.incremental_index.encode", side_effect=_mock_encode):
        index1.build(corpus)

    # Re-open cache in a new index instance
    index2 = IncrementalIndex(cache_path=cache_file)
    doc_ids, embeddings = index2.get_embeddings()

    assert set(doc_ids) == {"a", "b"}
    assert embeddings.shape == (2, 384)
