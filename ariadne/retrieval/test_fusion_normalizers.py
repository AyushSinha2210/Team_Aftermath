import pytest

from ariadne.retrieval.fusion import sigmoid_normalize, z_score_normalize


def test_z_score_normalize():
	scores = [("d1", 10.0), ("d2", 20.0), ("d3", 30.0)]
	normalized = z_score_normalize(scores)

	assert "d1" in normalized and "d2" in normalized and "d3" in normalized
	assert normalized["d1"] < normalized["d2"] < normalized["d3"]
	# Mean of standardized scores should be approximately 0
	mean_z = sum(normalized.values()) / 3
	assert abs(mean_z) < 1e-6


def test_z_score_constant_scores():
	scores = [("d1", 5.0), ("d2", 5.0)]
	normalized = z_score_normalize(scores)
	assert normalized["d1"] == 0.0
	assert normalized["d2"] == 0.0


def test_sigmoid_normalize():
	scores = [("d1", 1.0), ("d2", 5.0), ("d3", 10.0)]
	normalized = sigmoid_normalize(scores)

	# All values should be bounded between 0 and 1
	for val in normalized.values():
		assert 0.0 < val < 1.0

	assert normalized["d1"] < normalized["d2"] < normalized["d3"]
