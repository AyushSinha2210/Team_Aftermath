"""Small CPU BM25 index with code-aware tokenization."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping


_PARTS = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Split snake_case and camelCase while preserving exact identifiers."""
    tokens: list[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+", text):
        tokens.append(word.lower())
        for piece in word.split("_"):
            tokens.extend(part.lower() for part in _PARTS.findall(piece) if part.lower() != word.lower())
    return tokens


class SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.texts: dict[str, str] = {}
        self.counts: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.doc_freq: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.avg_length = 0.0

    def index(self, corpus: Mapping[str, str]) -> None:
        self.ids = list(corpus)
        self.texts = dict(corpus)
        self.counts = [Counter(tokenize(corpus[id_])) for id_ in self.ids]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.doc_freq = Counter(token for counts in self.counts for token in counts)
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, counts in enumerate(self.counts):
            for token, frequency in counts.items():
                postings[token].append((index, frequency))
        self.postings = dict(postings)

    def retrieve(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        if k <= 0 or not self.ids:
            return []
        query_terms = set(tokenize(query))
        n = len(self.ids)
        scores: dict[int, float] = defaultdict(float)
        for token in query_terms:
            postings = self.postings.get(token, [])
            if not postings:
                continue
            df = self.doc_freq[token]
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                length_factor = 1 - self.b + self.b * self.lengths[i] / max(self.avg_length, 1)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_factor)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], self.ids[item[0]]))
        return [(self.ids[i], score) for i, score in ordered[:k]]
