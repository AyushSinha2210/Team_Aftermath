"""Tests for code-aware tokenizer."""

from __future__ import annotations

from ariadne.retrieval.code_tokenizer import split_identifier, tokenize_code


def test_split_identifier_snake_case():
    assert split_identifier("is_valid_email") == ["is", "valid", "email", "is_valid_email"]
    assert split_identifier("binary_search") == ["binary", "search", "binary_search"]
    assert split_identifier("single") == ["single"]


def test_split_identifier_camel_case():
    assert split_identifier("isPalindrome") == ["is", "palindrome", "ispalindrome"]
    assert split_identifier("getHTTPResponse") == ["get", "http", "response", "gethttpresponse"]
    assert split_identifier("OAuth2Token") == ["oauth", "2", "token", "oauth2token"]


def test_split_identifier_empty_or_underscores():
    assert split_identifier("") == []
    assert split_identifier("___") == ["___"]


def test_tokenize_code_basic():
    code = "def binary_search(arr, target):\n    return -1"
    tokens = tokenize_code(code)
    assert "def" in tokens
    assert "binary" in tokens
    assert "search" in tokens
    assert "binary_search" in tokens
    assert "target" in tokens
    assert "return" in tokens


def test_tokenize_code_empty():
    assert tokenize_code("") == []
