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

	# Raw logits are unbounded (about -11.5 to 5.6 observed), so sigmoid maps them to (0, 1).
	sigmoid_scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -709.0, 709.0)))
	sigmoid_variance = float(np.var(sigmoid_scores))
	# Normalize by 0.25, the maximum variance for scores bounded to [0, 1].
	normalized_variance = sigmoid_variance / 0.25
	should_abstain = normalized_variance < threshold

	return {
		"should_abstain": should_abstain,
		"confidence_variance": normalized_variance,
		"top_candidate": None if should_abstain else reranked_portion[0],
	}
