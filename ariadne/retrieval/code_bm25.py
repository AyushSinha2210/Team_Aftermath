"""High-precision CodeBM25 retriever with AST-weighted tokenization.

Applies domain-specific term weighting:
- Function and class identifier boost (3.0x)
- Docstring summary terms boost (2.0x)
- Generic syntax keywords penalty (0.5x)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Dict, List, Sequence, Tuple

from ariadne.retrieval.code_tokenizer import COMMON_CODE_KEYWORDS, tokenize_code


class CodeBM25Retriever:
    """Specialized BM25 retriever for source code corpora with AST identifier boosting."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        identifier_boost: float = 2.5,
        keyword_penalty: float = 0.5,
    ) -> None:
        """Initializes CodeBM25Retriever.

        Args:
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            identifier_boost: Weight multiplier for non-keyword identifiers.
            keyword_penalty: Weight multiplier for common language keywords.
        """
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.identifier_boost = identifier_boost
        self.keyword_penalty = keyword_penalty

        self.ids: List[str] = []
        self.texts: Dict[str, str] = {}
        self.counts: List[Counter[str]] = []
        self.lengths: List[int] = []
        self.doc_freq: Counter[str] = Counter()
        self.postings: Dict[str, List[Tuple[int, float]]] = {}
        self.avg_length: float = 0.0

    def _weigh_tokens(self, tokens: Sequence[str]) -> Counter[str]:
        """Calculates weighted frequencies for tokens based on code semantics."""
        weighted = Counter()
        for tok in tokens:
            weight = self.keyword_penalty if tok in COMMON_CODE_KEYWORDS else self.identifier_boost
            weighted[tok] += weight
        return weighted

    def index(self, corpus: Mapping[str, str]) -> None:
        """Indexes source code documents with weighted term frequencies."""
        self.ids = list(corpus)
        self.texts = dict(corpus)
        if not self.ids:
            return

        self.counts = []
        self.lengths = []
        self.doc_freq = Counter()
        raw_postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

        for index, doc_id in enumerate(self.ids):
            tokens = tokenize_code(self.texts[doc_id])
            w_counts = self._weigh_tokens(tokens)
            self.counts.append(w_counts)
            self.lengths.append(len(tokens))
            for term, freq in w_counts.items():
                self.doc_freq[term] += 1
                raw_postings[term].append((index, freq))

        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.postings = dict(raw_postings)

    def retrieve(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        """Retrieves top-k documents ranked by CodeBM25 score."""
        if k <= 0 or not self.ids:
            return []

        query_terms = set(tokenize_code(query))
        n = len(self.ids)
        scores: Dict[int, float] = defaultdict(float)

        for term in query_terms:
            postings = self.postings.get(term, [])
            if not postings:
                continue
            df = self.doc_freq[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            if idf <= 0:
                continue

            for doc_idx, freq in postings:
                length = self.lengths[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (length / (self.avg_length or 1.0)))
                scores[doc_idx] += idf * (freq * (self.k1 + 1.0)) / denom

        ordered = sorted(scores.items(), key=lambda item: (-item[1], self.ids[item[0]]))
        return [(self.ids[idx], float(score)) for idx, score in ordered[:k]]
