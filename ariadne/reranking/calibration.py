"""Confidence-based abstention and calibration for retrieval scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
	"""Loads configuration dictionary from config.yaml.

	Args:
		config_path: Optional path to config.yaml. If None, infers from repository root.

	Returns:
		Parsed YAML configuration dictionary.
	"""
	if config_path is None:
		repo_root = Path(__file__).resolve().parent.parent
		config_path = repo_root / "config.yaml"

	if not config_path.exists():
		raise FileNotFoundError(f"Configuration file not found at: {config_path}")

	with open(config_path, "r", encoding="utf-8") as file:
		config: Dict[str, Any] = yaml.safe_load(file)
	return config


def calibrate(reranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Determines whether reranked candidates have enough score separation.

	Args:
		reranked_candidates: Candidates returned by the cross-encoder reranker.

	Returns:
		A calibration result containing abstention status, normalized variance,
		and the top candidate when confidence is sufficient.
	"""
	config = load_config()
	reranking_config = config.get("reranking", {})
	rerank_top_k = int(reranking_config.get("rerank_top_k", len(reranked_candidates)))
	threshold = float(reranking_config.get("calibration_threshold", 0.35))
	reranked_portion = reranked_candidates[:rerank_top_k]
	scores = np.asarray(
		[float(candidate["rerank_score"]) for candidate in reranked_portion],
		dtype=float,
	)

	if scores.size < 2:
		# Confidence cannot be determined from fewer than two rerank scores, so abstain.
		return {
			"should_abstain": True,
			"confidence_variance": 0.0,
			"top_candidate": None,
		}

	shifted_scores = scores - np.max(scores)
	probs = np.exp(shifted_scores)
	probs /= np.sum(probs)
	assert np.isclose(np.sum(probs), 1.0), "Softmax probabilities must sum to 1"
	softmax_variance = float(np.var(probs))
	# Entropy of probs is a documented alternative worth exploring later (Gemini review).
	# Normalize by the maximum variance of an n-item probability distribution so the
	# existing threshold remains on a comparable [0, 1] confidence scale. The
	# configured 0.35 threshold was tuned for the old sigmoid signal and needs
	# empirical re-tuning against this softmax signal.
	max_softmax_variance = (scores.size - 1) / (scores.size * scores.size)
	normalized_variance = softmax_variance / max_softmax_variance
	should_abstain = normalized_variance < threshold

	return {
		"should_abstain": should_abstain,
		"confidence_variance": normalized_variance,
		"top_candidate": None if should_abstain else reranked_portion[0],
	}


def temperature_scale(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
	"""Applies temperature scaling to raw reranker logits.

	Args:
		scores: 1D array of raw logit scores.
		temperature: Temperature parameter T > 0. T > 1 softens probabilities; T < 1 sharpens them.

	Returns:
		Scaled softmax probability distribution.
	"""
	if temperature <= 0:
		raise ValueError(f"Temperature must be strictly positive, got {temperature}")
	arr = np.asarray(scores, dtype=float)
	if arr.size == 0:
		return np.array([], dtype=float)
	scaled = (arr - np.max(arr)) / temperature
	exp_scores = np.exp(scaled)
	denom = np.sum(exp_scores)
	if denom == 0:
		return np.ones_like(arr) / arr.size
	return exp_scores / denom


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
	"""Computes the Expected Calibration Error (ECE) across prediction bins.

	Args:
		probs: Predicted confidence probabilities in [0, 1].
		labels: Binary ground-truth labels (0 or 1).
		n_bins: Number of equal-width probability bins.

	Returns:
		ECE value in [0, 1].
	"""
	p = np.asarray(probs, dtype=float)
	y = np.asarray(labels, dtype=float)
	if p.size == 0 or p.size != y.size:
		return 0.0

	bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
	ece = 0.0
	n_total = p.size

	for i in range(n_bins):
		bin_lower = bin_boundaries[i]
		bin_upper = bin_boundaries[i + 1]
		if i == n_bins - 1:
			in_bin = (p >= bin_lower) & (p <= bin_upper)
		else:
			in_bin = (p >= bin_lower) & (p < bin_upper)

		prop_in_bin = np.mean(in_bin)
		if prop_in_bin > 0:
			acc_in_bin = np.mean(y[in_bin])
			conf_in_bin = np.mean(p[in_bin])
			ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

	return float(ece)


def fit_temperature_scaling(
	val_logits: np.ndarray,
	val_labels: np.ndarray,
	init_temp: float = 1.0,
) -> float:
	"""Optimizes temperature T using negative log likelihood on validation logits.

	Args:
		val_logits: 1D array of reranker logits.
		val_labels: 1D array of binary ground-truth labels (0 or 1).
		init_temp: Initial temperature guess.

	Returns:
		Optimized temperature scalar T > 0.
	"""
	from scipy.optimize import minimize_scalar

	logits = np.asarray(val_logits, dtype=float)
	labels = np.asarray(val_labels, dtype=float)

	def nll_obj(t: float) -> float:
		if t <= 1e-4:
			return 1e9
		# Binary cross-entropy with temperature-scaled logit: p = 1 / (1 + exp(-logits / t))
		scaled = np.clip(logits / t, -50.0, 50.0)
		p = 1.0 / (1.0 + np.exp(-scaled))
		eps = 1e-12
		p_clipped = np.clip(p, eps, 1.0 - eps)
		loss = -np.mean(labels * np.log(p_clipped) + (1.0 - labels) * np.log(1.0 - p_clipped))
		return float(loss)

	res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
	return float(res.x) if res.success else init_temp


class PlattCalibrator:
	"""Parametric calibration via logistic regression (Platt scaling)."""

	def __init__(self) -> None:
		self.a: float = 1.0
		self.b: float = 0.0
		self.is_fitted: bool = False

	def fit(self, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
		"""Fits Platt scaling parameters a and b minimizing binary cross-entropy.

		Args:
			scores: 1D array of uncalibrated scores.
			labels: 1D array of binary ground-truth labels {0, 1}.

		Returns:
			Self (fitted calibrator).
		"""
		from scipy.optimize import minimize

		x = np.asarray(scores, dtype=float)
		y = np.asarray(labels, dtype=float)

		def nll(params: np.ndarray) -> float:
			a, b = params
			logits = np.clip(a * x + b, -50.0, 50.0)
			p = 1.0 / (1.0 + np.exp(-logits))
			eps = 1e-12
			p_clipped = np.clip(p, eps, 1.0 - eps)
			return float(-np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))

		res = minimize(nll, x0=[1.0, 0.0], method="L-BFGS-B")
		if res.success:
			self.a, self.b = float(res.x[0]), float(res.x[1])
		self.is_fitted = True
		return self

	def predict_proba(self, scores: np.ndarray) -> np.ndarray:
		"""Predicts calibrated posterior probabilities P(y=1|score)."""
		x = np.asarray(scores, dtype=float)
		logits = np.clip(self.a * x + self.b, -50.0, 50.0)
		return 1.0 / (1.0 + np.exp(-logits))


class IsotonicCalibrator:
	"""Non-parametric monotonic calibration via Isotonic Regression."""

	def __init__(self) -> None:
		from sklearn.isotonic import IsotonicRegression

		self._ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
		self.is_fitted: bool = False

	def fit(self, scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
		"""Fits isotonic regression mapping scores monotonically to empirical frequencies."""
		x = np.asarray(scores, dtype=float)
		y = np.asarray(labels, dtype=float)
		self._ir.fit(x, y)
		self.is_fitted = True
		return self

	def predict_proba(self, scores: np.ndarray) -> np.ndarray:
		"""Predicts calibrated probabilities via monotonic piecewise linear interpolation."""
		x = np.asarray(scores, dtype=float)
		return np.asarray(self._ir.predict(x), dtype=float)


def estimate_confidence_interval(
	scores: np.ndarray,
	confidence_level: float = 0.95,
	n_bootstrap: int = 1000,
	random_state: Optional[int] = None,
) -> Dict[str, float]:
	"""Estimates bootstrap confidence interval for retrieval scores mean and median.

	Args:
		scores: 1D array of scores.
		confidence_level: Desired coverage probability (e.g., 0.95).
		n_bootstrap: Number of bootstrap resamples.
		random_state: Seed for reproducibility.

	Returns:
		Dictionary containing mean, median, lower_bound, and upper_bound.
	"""
	arr = np.asarray(scores, dtype=float)
	if arr.size == 0:
		return {
			"mean": 0.0,
			"median": 0.0,
			"lower_bound": 0.0,
			"upper_bound": 0.0,
		}

	rng = np.random.default_rng(random_state)
	resamples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
	resampled_means = np.mean(resamples, axis=1)

	alpha = 1.0 - confidence_level
	lower_pct = 100.0 * (alpha / 2.0)
	upper_pct = 100.0 * (1.0 - alpha / 2.0)

	lower_bound = float(np.percentile(resampled_means, lower_pct))
	upper_bound = float(np.percentile(resampled_means, upper_pct))

	return {
		"mean": float(np.mean(arr)),
		"median": float(np.median(arr)),
		"lower_bound": lower_bound,
		"upper_bound": upper_bound,
	}



