import numpy as np
import pytest

from ariadne.reranking.calibration import IsotonicCalibrator, PlattCalibrator


def test_platt_calibrator_monotonicity():
	scores = np.array([-2.0, -0.5, 0.5, 2.0, 4.0])
	labels = np.array([0, 0, 1, 1, 1])

	calibrator = PlattCalibrator().fit(scores, labels)
	assert calibrator.is_fitted

	test_scores = np.array([-1.0, 0.0, 1.0, 3.0])
	probs = calibrator.predict_proba(test_scores)

	# Probs must be in [0, 1] and strictly monotonically increasing with score
	assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
	assert np.all(np.diff(probs) >= 0.0)


def test_isotonic_calibrator_bounds():
	scores = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
	labels = np.array([0, 0, 1, 1, 1])

	calibrator = IsotonicCalibrator().fit(scores, labels)
	assert calibrator.is_fitted

	test_scores = np.array([0.0, 0.15, 0.6, 1.0])
	probs = calibrator.predict_proba(test_scores)

	assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
	assert np.all(np.diff(probs) >= 0.0)
