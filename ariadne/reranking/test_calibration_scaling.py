import numpy as np
import pytest

from ariadne.reranking.calibration import (
	compute_ece,
	fit_temperature_scaling,
	temperature_scale,
)


def test_temperature_scale_sums_to_one():
	scores = np.array([2.5, 1.0, 0.2])
	probs = temperature_scale(scores, temperature=1.0)
	assert np.isclose(np.sum(probs), 1.0)
	assert probs[0] > probs[1] > probs[2]


def test_temperature_scale_temperature_softening():
	scores = np.array([5.0, 1.0])
	probs_cold = temperature_scale(scores, temperature=0.5)
	probs_hot = temperature_scale(scores, temperature=5.0)

	# Higher temperature should soften the distribution (bring max probability lower)
	assert probs_cold[0] > probs_hot[0]
	assert np.isclose(np.sum(probs_hot), 1.0)


def test_temperature_scale_invalid_temperature():
	with pytest.raises(ValueError):
		temperature_scale(np.array([1.0, 2.0]), temperature=0.0)


def test_compute_ece_perfect_calibration():
	probs = np.array([0.1, 0.9])
	labels = np.array([0.0, 1.0])
	ece = compute_ece(probs, labels, n_bins=10)
	assert ece < 0.2


def test_compute_ece_empty():
	assert compute_ece(np.array([]), np.array([])) == 0.0


def test_fit_temperature_scaling():
	np.random.seed(42)
	logits = np.array([3.0, -2.0, 1.5, -1.0, 4.0, -3.0])
	labels = np.array([1, 0, 1, 0, 1, 0])
	opt_temp = fit_temperature_scaling(logits, labels)
	assert opt_temp > 0.0
	assert isinstance(opt_temp, float)
