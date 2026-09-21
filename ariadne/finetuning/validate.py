"""Validation evaluation harness for the Ariadne bi-encoder.

Strictly evaluates against the CoIR apps validation split (data/raw/valid) only.
Computes NDCG@10, MRR@10, and Recall@k, and logs timestamped results to eval/results/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import yaml
from datasets import Dataset, load_from_disk

# Ensure repository root and workspace root are on sys.path
_repo_root = Path(__file__).resolve().parent.parent
_workspace_root = _repo_root.parent
for p in [str(_workspace_root), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ariadne.eval.metrics import evaluate_ranking
from ariadne.finetuning.embedder import encode, get_embedder


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configures structured logger for validation.

    Args:
        log_level: Desired log level string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("ariadne.finetuning.validate")
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
    """Loads central configuration.

    Args:
        config_path: Optional path to config.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = _repo_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_validation(
    model_name_or_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    save_results: bool = True,
) -> Dict[str, float]:
    """Runs evaluation on the valid split only.

    Args:
        model_name_or_path: Optional path or HuggingFace ID of model to evaluate.
            If None, evaluates the current embedder configuration in config.yaml.
        config: Optional pre-loaded config dict.
        save_results: If True, writes results JSON to eval/results/.

    Returns:
        Dictionary of calculated metrics (NDCG@10, MRR@10, Recall@k).
    """
    if config is None:
        config = load_config()

    logger = setup_logger(config.get("global", {}).get("log_level", "INFO"))
    data_cfg = config.get("data", {})
    raw_dir = _repo_root / data_cfg.get("raw_dir", "data/raw")
    valid_split_name = data_cfg.get("valid_split", "valid")
    valid_path = raw_dir / valid_split_name

    # STRICT SAFETY CHECK: Ensure test split is never accessed
    assert "test" not in str(valid_path).lower(), "SAFETY VIOLATION: Test split path referenced!"

    logger.info("=" * 70)
    logger.info("Starting validation harness against '%s' split ONLY...", valid_split_name)
    logger.info("Validation dataset path: %s", valid_path)

    if not valid_path.exists():
        raise FileNotFoundError(
            f"Validation dataset split not found at {valid_path}. Run download.sh first."
        )

    # Load validation split from disk
    valid_dataset: Dataset = load_from_disk(str(valid_path))
    logger.info("Validation records loaded: %d", len(valid_dataset))

    query_ids: List[str] = valid_dataset["query_id"]
    corpus_ids: List[str] = valid_dataset["corpus_id"]
    queries: List[str] = valid_dataset["query"]
    codes: List[str] = valid_dataset["code"]

    # Build ground-truth qrels: query_id -> set of relevant corpus_ids
    qrels: Dict[str, Set[str]] = {}
    for qid, cid in zip(query_ids, corpus_ids):
        if qid not in qrels:
            qrels[qid] = set()
        qrels[qid].add(cid)

    # Candidate corpus pool: unique documents from validation split
    cid_to_code: Dict[str, str] = {}
    for cid, code in zip(corpus_ids, codes):
        if cid not in cid_to_code:
            cid_to_code[cid] = code
    unique_corpus_ids: List[str] = list(cid_to_code.keys())
    unique_codes: List[str] = [cid_to_code[cid] for cid in unique_corpus_ids]

    eval_batch_size = int(config.get("finetuning", {}).get("eval_batch_size", 32))

    logger.info("Encoding %d validation queries...", len(queries))
    query_embeddings = encode(queries, batch_size=eval_batch_size, model_name_or_path=model_name_or_path)

    logger.info("Encoding %d candidate code snippets...", len(unique_codes))
    corpus_embeddings = encode(unique_codes, batch_size=eval_batch_size, model_name_or_path=model_name_or_path)

    logger.info("Computing retrieval metrics (NDCG@10, MRR@10, Recall@k)...")
    metrics = evaluate_ranking(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        query_ids=query_ids,
        corpus_ids=unique_corpus_ids,
        qrels=qrels,
        top_k=10,
    )

    model_label = model_name_or_path or config.get("model", {}).get("name", "baseline")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("Validation Results for '%s':", model_label)
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info("  %-12s: %.4f", k, v)
    logger.info("=" * 60)

    if save_results:
        results_dir = _repo_root / "eval" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "timestamp": timestamp,
            "model": model_label,
            "split": valid_split_name,
            "num_queries": len(queries),
            "num_candidates": len(unique_corpus_ids),
            "metrics": metrics,
        }
        result_file = results_dir / f"eval_valid_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)
        logger.info("Validation metrics saved to: %s", result_file)

    return metrics


if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else None
    run_validation(model_name_or_path=target_model)
