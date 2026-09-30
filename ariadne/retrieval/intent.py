"""Query intent classification for code retrieval search engines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QueryIntent(str, Enum):
	"""Types of search intents in a code repository."""
	SYMBOL_LOOKUP = "symbol_lookup"
	CONCEPTUAL = "conceptual"
	ERROR_DEBUG = "error_debug"


@dataclass(frozen=True)
class QueryIntentInfo:
	intent: QueryIntent
	confidence: float
	detected_symbol: Optional[str] = None
	detected_error_pattern: Optional[str] = None


_ERROR_KEYWORDS = sorted(
	[
		"assertionerror",
		"nullpointer",
		"syntaxerror",
		"typeerror",
		"valueerror",
		"keyerror",
		"exception",
		"traceback",
		"stack trace",
		"panicked at",
		"segfault",
		"exit code",
		"failure",
		"failed",
		"error",
	],
	key=lambda k: -len(k),
)

_CONCEPT_MARKERS = [
	"how to",
	"how do",
	"why does",
	"explain",
	"architecture",
	"overview",
	"pattern",
	"best practice",
	"usage of",
	"implementation of",
]


def classify_query_intent(query: str) -> QueryIntentInfo:
	"""Classifies a search query into SYMBOL_LOOKUP, CONCEPTUAL, or ERROR_DEBUG.

	Args:
		query: User search query string.

	Returns:
		QueryIntentInfo with the determined intent and confidence.
	"""
	q = query.strip()
	if not q:
		return QueryIntentInfo(intent=QueryIntent.CONCEPTUAL, confidence=0.5)

	lower_q = q.lower()

	# 1. Check for error / debugging patterns
	for err_kw in _ERROR_KEYWORDS:
		if err_kw in lower_q:
			return QueryIntentInfo(
				intent=QueryIntent.ERROR_DEBUG,
				confidence=0.90,
				detected_error_pattern=err_kw,
			)

	# 2. Check for conceptual markers
	for concept_kw in _CONCEPT_MARKERS:
		if concept_kw in lower_q:
			return QueryIntentInfo(
				intent=QueryIntent.CONCEPTUAL,
				confidence=0.85,
			)

	# 3. Check for direct code identifier / symbol lookup
	# Single identifier like `formatDate`, `get_user_by_id`, or `AuthService.login`
	tokens = q.split()
	if len(tokens) == 1:
		token = tokens[0]
		if re.match(r"^[a-zA-Z_][a-zA-Z0-9_\.]*(\(\))?$", token):
			return QueryIntentInfo(
				intent=QueryIntent.SYMBOL_LOOKUP,
				confidence=0.95,
				detected_symbol=token,
			)

	if len(tokens) <= 3 and any(ch in q for ch in ["(", "::", "->", "."]):
		return QueryIntentInfo(
			intent=QueryIntent.SYMBOL_LOOKUP,
			confidence=0.85,
			detected_symbol=q,
		)

	# Default to conceptual
	return QueryIntentInfo(intent=QueryIntent.CONCEPTUAL, confidence=0.70)
