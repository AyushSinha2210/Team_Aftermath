"""Lightweight rule-based query expansion for natural-language-to-code alignment.

Bridges vocabulary gaps by mapping natural language concepts to idiomatic
code tokens, method names, and syntax idioms for hybrid retrieval.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

# Common algorithmic and development concept code synonyms
CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "palindrome": ["is_palindrome", "reversed", "[::-1]"],
    "binary search": ["binary_search", "bisect", "left", "right", "mid"],
    "sort": ["sorted", "quicksort", "mergesort", "key"],
    "json": ["json.loads", "json.dumps", "json"],
    "regex": ["re.search", "re.match", "re.compile", "pattern"],
    "file": ["open", "read", "readlines", "with open"],
    "matrix": ["grid", "row", "col", "matrix"],
    "prime": ["is_prime", "primes", "sieve"],
    "factorial": ["math.factorial", "factorial"],
    "tree": ["root", "left", "right", "TreeNode"],
    "graph": ["visited", "adj", "dfs", "bfs", "queue"],
    "jwt": ["token", "verify", "decode", "payload"],
    "hash": ["hashlib", "sha256", "md5", "hexdigest"],
    "random": ["random.choice", "randint", "sample"],
    "flatten": ["itertools.chain", "nested", "flatten"],
    "linked list": ["ListNode", "head", "next"],
}


def expand_code_query(query: str, max_expansions: int = 5) -> str:
    """Expands a search query with programming idioms and identifier variants.

    Args:
        query: Natural language or hybrid query.
        max_expansions: Maximum number of expansion terms to append.

    Returns:
        Expanded query string.
    """
    if not query or not query.strip():
        return ""

    query_lower = query.lower()
    added_terms: List[str] = []

    for concept, synonyms in CONCEPT_SYNONYMS.items():
        if concept in query_lower:
            for syn in synonyms:
                if syn.lower() not in query_lower and syn not in added_terms:
                    added_terms.append(syn)
                    if len(added_terms) >= max_expansions:
                        break
        if len(added_terms) >= max_expansions:
            break

    if not added_terms:
        return query

    return f"{query} {' '.join(added_terms)}"
