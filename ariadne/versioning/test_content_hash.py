"""Unit tests for ariadne.versioning.content_hash."""

from __future__ import annotations

from ariadne.versioning.content_hash import (
    diff_hashes,
    get_hash_algorithm,
    hash_content,
    hash_corpus,
)


def test_hash_content_deterministic() -> None:
    """Verifies that hashing the same content is deterministic across calls."""
    snippet = "def compute_sum(a, b):\n    return a + b\n"
    hash1 = hash_content(snippet)
    hash2 = hash_content(snippet)
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # sha256 hex length
    assert hash1 == hash2


def test_hash_content_whitespace_normalization() -> None:
    """Verifies that trivial whitespace differences hash identically."""
    base_code = "def process(items):\n    result = []\n    for item in items:\n        result.append(item * 2)\n    return result"
    
    # Extra trailing spaces on lines, CRLF line endings, extra leading/trailing blank lines
    dirty_code = "  \r\ndef process(items):   \r\n    result = []  \r\n    for item in items: \t \r\n        result.append(item * 2)  \r\n    return result\r\n\r\n  "
    
    # Semantically different code
    different_code = "def process(items):\n    result = []\n    for item in items:\n        result.append(item * 3)\n    return result"

    assert hash_content(base_code) == hash_content(dirty_code)
    assert hash_content(base_code) != hash_content(different_code)


def test_hash_algorithm_from_config() -> None:
    """Verifies that the hash algorithm is retrieved from config.yaml."""
    algo = get_hash_algorithm()
    assert algo == "sha256"


def test_hash_corpus() -> None:
    """Verifies that hash_corpus hashes an entire dictionary of documents."""
    corpus = {
        "fn_add": "def add(x, y): return x + y",
        "fn_sub": "def sub(x, y): return x - y",
    }
    corpus_hashes = hash_corpus(corpus)

    assert set(corpus_hashes.keys()) == {"fn_add", "fn_sub"}
    assert corpus_hashes["fn_add"] == hash_content(corpus["fn_add"])
    assert corpus_hashes["fn_sub"] == hash_content(corpus["fn_sub"])


def test_diff_hashes_buckets() -> None:
    """Verifies that diff_hashes correctly categorizes added, removed, changed, and unchanged docs."""
    old_corpus = {
        "doc_unchanged": "def unchanged():\n    return 42",
        "doc_changed": "def to_change():\n    return 'v1'",
        "doc_removed": "def to_remove():\n    pass",
    }
    new_corpus = {
        "doc_unchanged": "def unchanged():  \r\n    return 42 \r\n",  # whitespace diff only
        "doc_changed": "def to_change():\n    return 'v2'",
        "doc_added": "def newly_added():\n    return True",
    }

    old_hashes = hash_corpus(old_corpus)
    new_hashes = hash_corpus(new_corpus)

    diff = diff_hashes(old_hashes, new_hashes)

    assert diff["added"] == {"doc_added"}
    assert diff["removed"] == {"doc_removed"}
    assert diff["changed"] == {"doc_changed"}
    assert diff["unchanged"] == {"doc_unchanged"}
