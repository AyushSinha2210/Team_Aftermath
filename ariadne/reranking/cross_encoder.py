"""CPU-optimized cross-encoder reranker for top-k fused candidate shortlist."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_CACHED_MODEL: Any = None
_CACHED_MODEL_NAME: Optional[str] = None


def setup_logger(log_level: str = "INFO") -> logging.Logger:
	"""Configures and returns a structured logger.

	Args:
		log_level: Desired logging verbosity level.

	Returns:
		Configured logging.Logger instance.
	"""
	logger = logging.getLogger("ariadne.reranking.cross_encoder")
	logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
	if not logger.handlers:
		handler = logging.StreamHandler(sys.stdout)
		formatter = logging.Formatter(
			fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	return logger


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


def load_cross_encoder() -> Any:
	"""Retrieves or initializes the cached CPU CrossEncoder model instance.

	Returns:
		Loaded sentence-transformers CrossEncoder instance.
	"""
	global _CACHED_MODEL, _CACHED_MODEL_NAME

	config = load_config()
	logger = setup_logger(config.get("global", {}).get("log_level", "INFO"))
	model_name = config.get("reranking", {}).get(
		"cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
	)

	if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == model_name:
		return _CACHED_MODEL

	from sentence_transformers import CrossEncoder

	logger.info("Loading cross-encoder model '%s' on device 'cpu'...", model_name)
	_CACHED_MODEL = CrossEncoder(model_name, device="cpu")
	_CACHED_MODEL_NAME = model_name
	return _CACHED_MODEL


def _truncate_text(model: Any, query: str, text: str) -> str:
	"""Truncates candidate text to the model tokenizer's pair length when available."""
	tokenizer = getattr(model, "tokenizer", None)
	max_length = getattr(model, "max_length", None)
	if max_length is None and tokenizer is not None:
		max_length = getattr(tokenizer, "model_max_length", None)

	if tokenizer is None or not isinstance(max_length, int) or max_length <= 0:
		return text

	encoded_query = tokenizer.encode(query, add_special_tokens=False)
	available_tokens = max_length - len(encoded_query) - 3
	if available_tokens <= 0:
		return text[:1]

	encoded_text = tokenizer.encode(text, add_special_tokens=False)
	if len(encoded_text) <= available_tokens:
		return text

	decoded = tokenizer.decode(encoded_text[:available_tokens], skip_special_tokens=True)
	return decoded or text[:1]


def rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Reranks the configured top-k candidates with a CPU cross-encoder.

	Args:
		query: User query to score against candidate text.
		candidates: Candidate dictionaries containing ``id``, ``text``, and ``fusion_score``.

	Returns:
		Reranked top-k candidates followed by the untouched remainder.
	"""
	if not candidates:
		return []

	config = load_config()
	rerank_top_k = int(config.get("reranking", {}).get("rerank_top_k", len(candidates)))
	reranked_candidates = candidates[:rerank_top_k]
	remainder = candidates[rerank_top_k:]
	model = load_cross_encoder()
	pairs = [(query, _truncate_text(model, query, candidate["text"])) for candidate in reranked_candidates]
	scores = model.predict(pairs)

	scored_candidates = []
	for candidate, score in zip(reranked_candidates, scores):
		candidate["rerank_score"] = float(score)
		scored_candidates.append(candidate)

	scored_candidates.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
	return scored_candidates + remainder
