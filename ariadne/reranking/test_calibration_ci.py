import numpy as np

from ariadne.reranking.calibration import estimate_confidence_interval


def test_confidence_interval_bounds():
	scores = np.array([1.0, 1.2, 0.9, 1.1, 1.05, 0.95, 1.15, 0.98])
	res = estimate_confidence_interval(scores, confidence_level=0.95, random_state=42)

	assert res["lower_bound"] <= res["mean"] <= res["upper_bound"]
	assert 0.8 < res["mean"] < 1.3
	assert res["lower_bound"] < res["upper_bound"]


def test_confidence_interval_empty():
	res = estimate_confidence_interval(np.array([]))
	assert res["mean"] == 0.0
	assert res["lower_bound"] == 0.0
	assert res["upper_bound"] == 0.0
