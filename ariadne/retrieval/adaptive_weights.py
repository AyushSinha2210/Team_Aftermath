"""Adaptive fusion weight estimation based on query syntax and lexical density."""

from __future__ import annotations

import re
from typing import Tuple

from ariadne.retrieval.code_tokenizer import split_identifier


def compute_syntax_density(query: str) -> float:
	"""Computes a syntax density score in [0.0, 1.0] for a search query.

	Queries with code operators, snake_case, camelCase, or programming keywords
	have high density, whereas natural language questions have low density.

	Args:
		query: Raw search query string.

	Returns:
		Density float between 0.0 (pure NL) and 1.0 (pure code symbol).
	"""
	if not query or not query.strip():
		return 0.0

	tokens = query.strip().split()
	if not tokens:
		return 0.0

	code_indicators = 0

	# 1. Operators & punctuation common in code
	op_pattern = re.compile(r"[_\(\)\{\}\[\]\:\;\=\-\>\<\.\&\|\!\@\#\$\%\^\*]")
	# 2. Programming keywords
	keyword_pattern = re.compile(
		r"^(def|class|function|return|import|export|from|async|await|const|let|var|if|else|for|while|try|catch|except|raise|throw)$",
		re.IGNORECASE,
	)

	for token in tokens:
		is_code_like = False
		if op_pattern.search(token):
			is_code_like = True
		elif keyword_pattern.match(token):
			is_code_like = True
		elif "_" in token:
			is_code_like = True
		else:
			subwords = split_identifier(token)
			if len(subwords) > 1:
				is_code_like = True

		if is_code_like:
			code_indicators += 1

	return min(1.0, code_indicators / len(tokens))


def get_adaptive_fusion_weights(
	query: str,
	base_dense_weight: float = 0.85,
	min_dense_weight: float = 0.40,
) -> Tuple[float, float]:
	"""Computes dynamic (dense_weight, sparse_weight) tailored to the query's code density.

	High syntax queries (e.g. symbol lookups) boost sparse BM25 weighting.
	Conceptual questions prioritize dense semantic vector search.

	Args:
		query: User query string.
		base_dense_weight: Baseline dense weight for natural language (default 0.85).
		min_dense_weight: Lower bound for dense weight on pure code queries (default 0.40).

	Returns:
		Tuple of (dense_weight, sparse_weight), where dense_weight + sparse_weight == 1.0.
	"""
	density = compute_syntax_density(query)
	# Interpolate between base_dense_weight (density=0) and min_dense_weight (density=1)
	dense_w = base_dense_weight - density * (base_dense_weight - min_dense_weight)
	dense_w = max(min_dense_weight, min(base_dense_weight, dense_w))
	sparse_w = 1.0 - dense_w
	return round(dense_w, 4), round(sparse_w, 4)
