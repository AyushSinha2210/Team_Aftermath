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
