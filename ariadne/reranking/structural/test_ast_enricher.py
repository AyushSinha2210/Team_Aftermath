"""Tests for AST-enriched structural preprocessor."""

from __future__ import annotations

from ariadne.reranking.structural.ast_enricher import (
    enrich_code_snippet,
    enrich_corpus,
    extract_python_metadata,
    format_structural_prefix,
)


def test_extract_python_metadata_valid_function():
    code = """def binary_search(arr, target):
    \"\"\"Finds target index in sorted array.\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
    return -1
"""
    meta = extract_python_metadata(code)
    assert "binary_search(arr, target)" in meta["functions"]
    assert "Finds target index in sorted array." in meta["docstring"]
    assert "len" in meta["calls"]


def test_extract_python_metadata_class():
    code = """class TokenValidator:
    def validate(self, token):
        return check_signature(token)
"""
    meta = extract_python_metadata(code)
    assert "TokenValidator" in meta["classes"]
    assert "validate(token)" in meta["functions"]
    assert "check_signature" in meta["calls"]


def test_extract_python_metadata_syntax_error_fallback():
    # Incomplete syntax
    code = "def incomplete_func(a, b):\n    if a >"
    meta = extract_python_metadata(code)
    assert "incomplete_func(a, b)" in meta["functions"]


def test_enrich_code_snippet():
    code = "def is_even(n):\n    return n % 2 == 0"
    enriched = enrich_code_snippet(code)
    assert "# SIGNATURE: is_even(n)" in enriched
    assert code in enriched


def test_enrich_corpus():
    corpus = {"c1": "def foo(): pass", "c2": "def bar(): pass"}
    enriched = enrich_corpus(corpus)
    assert "c1" in enriched and "c2" in enriched
    assert "# SIGNATURE: foo()" in enriched["c1"]
    assert "# SIGNATURE: bar()" in enriched["c2"]
