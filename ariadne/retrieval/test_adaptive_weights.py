import pytest

from ariadne.retrieval.adaptive_weights import (
	compute_syntax_density,
	get_adaptive_fusion_weights,
)


def test_syntax_density_nl_vs_code():
	nl_query = "how to sort a list in ascending order"
	code_query = "def _sort_items(arr: List[int]) -> None:"

	nl_density = compute_syntax_density(nl_query)
	code_density = compute_syntax_density(code_query)

	assert nl_density < 0.2
	assert code_density > 0.6


def test_syntax_density_empty():
	assert compute_syntax_density("") == 0.0
	assert compute_syntax_density("   ") == 0.0


def test_adaptive_weights_shift():
	nl_query = "authenticate user credentials"
	code_query = "auth.verify_token(token)"

	dense_nl, sparse_nl = get_adaptive_fusion_weights(nl_query)
	dense_code, sparse_code = get_adaptive_fusion_weights(code_query)

	# Code query should have higher sparse weight (and lower dense weight) than NL query
	assert sparse_code > sparse_nl
	assert dense_code < dense_nl
	assert round(dense_nl + sparse_nl, 2) == 1.0
	assert round(dense_code + sparse_code, 2) == 1.0
