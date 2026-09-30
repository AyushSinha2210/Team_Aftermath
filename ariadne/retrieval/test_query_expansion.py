"""Tests for query expansion module."""

from __future__ import annotations

from ariadne.retrieval.query_expansion import expand_code_query


def test_expand_query_empty():
    assert expand_code_query("") == ""
    assert expand_code_query("   ") == ""


def test_expand_query_palindrome():
    expanded = expand_code_query("find if string is palindrome")
    assert "is_palindrome" in expanded
    assert "find if string is palindrome" in expanded


def test_expand_query_binary_search():
    expanded = expand_code_query("binary search in sorted array")
    assert "bisect" in expanded or "mid" in expanded


def test_expand_query_unmatched():
    query = "unmatched xyzzy phrase"
    assert expand_code_query(query) == query
