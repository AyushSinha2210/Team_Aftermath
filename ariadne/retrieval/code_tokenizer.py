"""Code-aware tokenizer for structural and lexical code retrieval (CodeBM25).

Splits identifiers across camelCase, PascalCase, and snake_case boundaries,
normalizes programming tokens, and isolates AST identifier symbols from syntactic noise.
"""

from __future__ import annotations

import re
from typing import List, Set, Sequence

_DIGIT_LETTER_REGEX = re.compile(r"([a-zA-Z])(\d+)")
_LETTER_DIGIT_REGEX = re.compile(r"(\d+)([a-zA-Z])")
_UPPER_SPLIT_REGEX = re.compile(r"([A-Z]{2,})([A-Z][a-z])")
_CAMEL_SPLIT_REGEX = re.compile(r"([a-z\d])([A-Z])")
_PUNCT_SPLIT_REGEX = re.compile(r"[^a-zA-Z0-9_]")
_UNDERSCORE_SPLIT_REGEX = re.compile(r"_+")
_WHITESPACE_REGEX = re.compile(r"\s+")

# Python & generic programming language reserved keywords
COMMON_CODE_KEYWORDS: Set[str] = {
    "def", "class", "return", "import", "from", "as", "if", "elif", "else",
    "for", "while", "break", "continue", "pass", "try", "except", "finally",
    "raise", "with", "yield", "lambda", "assert", "global", "nonlocal",
    "async", "await", "function", "const", "let", "var", "public", "private",
    "protected", "static", "void", "int", "float", "bool", "str", "self", "this",
    "true", "false", "none", "null", "undefined",
}


def split_identifier(identifier: str) -> List[str]:
    """Splits a programming identifier into constituent sub-tokens.

    Handles snake_case, camelCase, PascalCase, acronyms, and digit boundaries.

    Examples:
        >>> split_identifier("is_valid_email")
        ['is', 'valid', 'email', 'is_valid_email']
        >>> split_identifier("getHTTPResponseCode")
        ['get', 'http', 'response', 'code', 'gethttpresponsecode']
        >>> split_identifier("OAuth2Token")
        ['oauth', '2', 'token', 'oauth2token']
        >>> split_identifier("binary_search")
        ['binary', 'search', 'binary_search']

    Args:
        identifier: A single variable, function, class, or method name.

    Returns:
        List of lowercase sub-tokens plus the full normalized identifier.
    """
    if not identifier:
        return []

    # Strip leading/trailing underscores
    cleaned = identifier.strip("_")
    if not cleaned:
        return [identifier.lower()]

    # Insert boundaries for digits and camelCase / acronyms
    s = _DIGIT_LETTER_REGEX.sub(r"\1_\2", cleaned)
    s = _LETTER_DIGIT_REGEX.sub(r"\1_\2", s)
    s = _UPPER_SPLIT_REGEX.sub(r"\1_\2", s)
    s = _CAMEL_SPLIT_REGEX.sub(r"\1_\2", s)
    parts = [part.lower() for part in _UNDERSCORE_SPLIT_REGEX.split(s) if part]

    lower_full = cleaned.lower()
    results: List[str] = []
    for part in parts:
        if part not in results:
            results.append(part)

    # Preserve full identifier if composite
    if len(parts) > 1 and lower_full not in results:
        results.append(lower_full)

    return results


def tokenize_code(
    text: str,
    split_subtokens: bool = True,
    preserve_case: bool = False,
    min_length: int = 1,
) -> List[str]:
    """Tokenizes code or query text into clean, searchable lexical terms.

    Args:
        text: Raw code snippet or search query.
        split_subtokens: Whether to decompose compound identifiers into parts.
        preserve_case: Whether to retain original casing (default False).
        min_length: Minimum character length for tokens.

    Returns:
        List of extracted token strings.
    """
    if not text:
        return []

    # Replace punctuation and code symbols with spaces
    sanitized = _PUNCT_SPLIT_REGEX.sub(" ", text)
    raw_tokens = [tok for tok in _WHITESPACE_REGEX.split(sanitized) if tok]

    tokens: List[str] = []
    for token in raw_tokens:
        if len(token) < min_length:
            continue

        if split_subtokens and ("_" in token or any(c.isupper() for c in token)):
            sub_tokens = split_identifier(token)
            for sub in sub_tokens:
                if len(sub) >= min_length:
                    tokens.append(sub if preserve_case else sub.lower())
        else:
            tokens.append(token if preserve_case else token.lower())

    return tokens
