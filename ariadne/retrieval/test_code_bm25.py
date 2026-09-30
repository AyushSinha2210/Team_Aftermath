"""Unit tests for CodeBM25Retriever."""

from __future__ import annotations

import pytest

from ariadne.retrieval.code_bm25 import CodeBM25Retriever


def test_code_bm25_empty():
    retriever = CodeBM25Retriever()
    retriever.index({})
    assert retriever.retrieve("binary search") == []


def test_code_bm25_invalid_params():
    with pytest.raises(ValueError):
        CodeBM25Retriever(k1=-1.0)
    with pytest.raises(ValueError):
        CodeBM25Retriever(b=1.5)


def test_code_bm25_identifier_boost():
    corpus = {
        "doc1": "def is_valid_email(s):\n    return '@' in s",
        "doc2": "def process_data(data):\n    return data",
    }
    retriever = CodeBM25Retriever()
    retriever.index(corpus)

    results = retriever.retrieve("email validation")
    assert len(results) > 0
    assert results[0][0] == "doc1"


def test_code_bm25_keyword_penalty():
    corpus = {
        "doc_keywords": "def pass return if else while break",
        "doc_identifier": "def compute_fibonacci(n): pass",
    }
    retriever = CodeBM25Retriever()
    retriever.index(corpus)

    # Searching for compute_fibonacci should rank doc_identifier top
    results = retriever.retrieve("compute_fibonacci")
    assert results[0][0] == "doc_identifier"
