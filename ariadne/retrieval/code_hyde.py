"""Code HyDE (Hypothetical Document Embeddings) for bridging NL query to code embedding space."""

from __future__ import annotations

import re
from typing import List

import numpy as np

from ariadne.retrieval.code_tokenizer import split_identifier


def _to_snake_case(text: str) -> str:
	words = re.findall(r"[a-zA-Z0-9]+", text.lower())
	return "_".join(words[:5]) if words else "handler"


def generate_hyde_code(query: str, language: str = "python") -> str:
	"""Generates a hypothetical code snippet matching the query's semantic intent.

	Dense code models perform significantly better in code-to-code similarity space
	than in natural-language-to-code space.

	Args:
		query: Natural language query (e.g. 'verify user password hash with bcrypt').
		language: Target language ('python' or 'javascript').

	Returns:
		Synthesized hypothetical code snippet.
	"""
	clean_q = query.strip()
	fn_name = _to_snake_case(clean_q)

	if language.lower() in ("javascript", "typescript", "js", "ts"):
		return (
			f"// {clean_q}\n"
			f"async function {fn_name}(params) {{\n"
			f"  // Implementation for {clean_q}\n"
			f"  try {{\n"
			f"    return await execute_{fn_name}(params);\n"
			f"  }} catch (err) {{\n"
			f"    throw err;\n"
			f"  }}\n"
			f"}}"
		)

	# Default Python
	return (
		f'def {fn_name}(*args, **kwargs):\n'
		f'\t"""{clean_q}\n\n'
		f'\tReturns:\n'
		f'\t\tResult for {clean_q}.\n'
		f'\t"""\n'
		f'\tpass\n'
	)


def expand_hyde_queries(query: str) -> List[str]:
	"""Generates query variations including raw query and hypothetical code representations."""
	raw = query.strip()
	if not raw:
		return []

	py_hyde = generate_hyde_code(raw, language="python")
	js_hyde = generate_hyde_code(raw, language="javascript")
	return [raw, py_hyde, js_hyde]


def average_hyde_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
	"""Averages a list of embedding vectors and L2-normalizes the resulting centroid.

	Args:
		embeddings: List of 1D numpy embedding vectors.

	Returns:
		Unit-normalized centroid vector.
	"""
	if not embeddings:
		return np.array([], dtype=float)

	stacked = np.vstack(embeddings)
	centroid = np.mean(stacked, axis=0)
	norm = np.linalg.norm(centroid)
	if norm == 0.0:
		return centroid
	return centroid / norm
