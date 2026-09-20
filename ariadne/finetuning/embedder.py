"""Bi-encoder inference and embedding module.

Provides a unified encode(texts: list[str]) -> np.ndarray entrypoint for
retrieval pipelines (Person B/C/D), backed by an off-the-shelf or fine-tuned
CPU-optimized SentenceTransformer model configured via config.yaml.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Module-level model cache to prevent expensive re-instantiation across inference calls
_CACHED_MODEL: Any = None
_CACHED_MODEL_NAME_OR_PATH: Optional[str] = None


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        log_level: Desired logging verbosity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("ariadne.finetuning.embedder")
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

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def get_embedder(
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """Retrieves or initializes a cached SentenceTransformer model instance.

    Args:
        model_name_or_path: HuggingFace model ID or local checkpoint path.
            If None, reads from config.yaml (checking checkpoint_path then base_model).
        device: Target inference device (e.g., 'cpu'). If None, pulled from config.yaml.

    Returns:
        Loaded SentenceTransformer instance.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME_OR_PATH

    from sentence_transformers import SentenceTransformer

    config = load_config()
    logger = setup_logger(config.get("global", {}).get("log_level", "INFO"))

    # Resolve model path: check fine-tuned checkpoint first, fall back to base model
    if model_name_or_path is None:
        best_ckpt = config.get("finetuning", {}).get("best_checkpoint_path")
        repo_root = Path(__file__).resolve().parent.parent
        if best_ckpt and (repo_root / best_ckpt).exists():
            model_name_or_path = str(repo_root / best_ckpt)
            logger.info("Using fine-tuned checkpoint: %s", model_name_or_path)
        else:
            model_name_or_path = config.get("model", {}).get(
                "name", "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Using baseline base model: %s", model_name_or_path)

    if device is None:
        device = config.get("global", {}).get("device", "cpu")

    # Return cached instance if model target matches
    if _CACHED_MODEL is not None and _CACHED_MODEL_NAME_OR_PATH == model_name_or_path:
        return _CACHED_MODEL

    logger.info("Loading embedder model '%s' on device '%s'...", model_name_or_path, device)
    model = SentenceTransformer(model_name_or_path, device=device)

    max_seq_length = config.get("model", {}).get("max_seq_length", 512)
    model.max_seq_length = max_seq_length

    _CACHED_MODEL = model
    _CACHED_MODEL_NAME_OR_PATH = model_name_or_path
    return _CACHED_MODEL


def encode(
    texts: List[str],
    batch_size: Optional[int] = None,
    normalize_embeddings: Optional[bool] = None,
    model_name_or_path: Optional[str] = None,
) -> np.ndarray:
    """Encodes a list of text strings into normalized dense embedding vectors.

    This is the primary embedder entrypoint contract consumed by Person B
    (dense retriever), Person C (reranker candidate scorer), and Person D
    (versioning incremental index).

    Args:
        texts: List of query or code text strings to encode.
        batch_size: Batch size for forward pass. Pulled from config.yaml if None.
        normalize_embeddings: Whether to L2-normalize vectors. Pulled from config.yaml if None.
        model_name_or_path: Optional explicit model or checkpoint to encode with.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim) with float32 precision.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    config = load_config()
    model = get_embedder(model_name_or_path=model_name_or_path)

    if batch_size is None:
        batch_size = int(config.get("finetuning", {}).get("eval_batch_size", 32))

    if normalize_embeddings is None:
        normalize_embeddings = bool(config.get("model", {}).get("normalize_embeddings", True))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    # Self-test verification harness
    test_texts = [
        "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
        "How to implement binary search in Python with logarithmic complexity?",
        "def fibonacci(n): a, b = 0, 1; [a := b, b := a + b for _ in range(n)]; return a",
        "Find minimum element in a rotated sorted array",
        "class Node: def __init__(self, val=0, next=None): self.val = val; self.next = next",
    ]
    cfg = load_config()
    log = setup_logger("INFO")
    log.info("Running baseline encode verification on %d sample texts...", len(test_texts))
    vectors = encode(test_texts)
    log.info("Encoding successful! Resulting vector array shape: %s, dtype: %s", vectors.shape, vectors.dtype)
    assert vectors.shape == (5, cfg.get("model", {}).get("embedding_dim", 384)), f"Unexpected shape: {vectors.shape}"
    log.info("Baseline bi-encoder fallback verified successfully!")
