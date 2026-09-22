"""Validation evaluation harness for the Ariadne bi-encoder.

Strictly evaluates against the CoIR apps validation split (data/raw/valid) only.
Computes NDCG@10, MRR@10, and Recall@k, and logs timestamped results to eval/results/.

SAFETY RAIL: Enforces strict runtime assertions preventing any reference to the 'test' split.
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


def verify_safety_rails(dataset_path: Path, split_name: str) -> None:
    """Explicit runtime safety rail asserting that no test split path is ever referenced.

    This is an active architectural safety constraint to guarantee zero data leakage
    before the final official benchmark evaluation in Phase 6.

    Args:
        dataset_path: Path to dataset split to be evaluated.
        split_name: Name of split declared in configuration or arguments.

    Raises:
        AssertionError: If any reference to 'test' is detected.
    """
    norm_path = dataset_path.as_posix().lower()
    norm_split = split_name.strip().lower()

    # Safety Rail 1: Forbid 'test' split name
    if norm_split == "test":
        raise AssertionError(
            f"CRITICAL SAFETY RAIL VIOLATION: Validation harness was called with split='test'. "
            f"Evaluating against the test split is strictly prohibited until Phase 6!"
        )

    # Safety Rail 2: Forbid 'test' path pattern
    if "/test" in norm_path or norm_path.endswith("test") or "\\test" in str(dataset_path).lower():
        raise AssertionError(
            f"CRITICAL SAFETY RAIL VIOLATION: Dataset path references 'test' split: {dataset_path}. "
            f"Evaluating against the test split is strictly prohibited until Phase 6!"
        )

    # Safety Rail 3: Positively enforce that target is valid split
    if norm_split != "valid":
        raise AssertionError(
            f"CRITICAL SAFETY RAIL VIOLATION: Expected target split 'valid', but got '{split_name}'."
        )

    if "valid" not in norm_path:
        raise AssertionError(
            f"CRITICAL SAFETY RAIL VIOLATION: Target path '{dataset_path}' does not resolve to 'valid'."
        )


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
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """Runs evaluation strictly on the valid split.

    Enforces runtime safety rails, loads candidate documents and queries from
    data/raw/valid, encodes embeddings, and computes ranking metrics.

    Args:
        model_name_or_path: Optional path or HuggingFace ID of model to evaluate.
            If None, evaluates the current embedder configuration in config.yaml.
        config: Optional pre-loaded config dict.
        save_results: If True, writes results JSON to eval/results/.
        batch_size: Optional batch size for encoding.

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

    # =========================================================================
    # MANDATORY SAFETY RAIL CHECK
    # =========================================================================
    verify_safety_rails(valid_path, valid_split_name)

    logger.info("=" * 70)
    logger.info("Starting Validation Harness (Safety Rails Verified: 'valid' split ONLY)")
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

    if batch_size is None:
        eval_batch_size = int(config.get("finetuning", {}).get("eval_batch_size", 32))
    else:
        eval_batch_size = batch_size

    logger.info("Target model checkpoint: %s", model_name_or_path or "default from config")
    logger.info("Encoding %d validation queries...", len(queries))
    query_embeddings = encode(
        queries,
        batch_size=eval_batch_size,
        model_name_or_path=model_name_or_path,
    )

    logger.info("Encoding %d candidate code snippets...", len(unique_codes))
    corpus_embeddings = encode(
        unique_codes,
        batch_size=eval_batch_size,
        model_name_or_path=model_name_or_path,
    )

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

    # Format output block
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


def test_safety_rail_enforcement() -> None:
    """Unit test for safety rail enforcement function.

    Verifies that any attempt to evaluate against 'test' raises AssertionError.
    """
    logger = setup_logger("INFO")
    logger.info("Running internal safety rail validation self-test...")

    # Test 1: valid path passes
    valid_test_path = _repo_root / "data" / "raw" / "valid"
    verify_safety_rails(valid_test_path, "valid")

    # Test 2: 'test' split name triggers assertion
    try:
        verify_safety_rails(_repo_root / "data" / "raw" / "test", "test")
        raise RuntimeError("Safety rail failed to catch test split!")
    except AssertionError as e:
        logger.info("Safety Rail Test 1 Passed: Caught prohibited split name -> %s", e)

    # Test 3: 'test' in path triggers assertion
    try:
        verify_safety_rails(_repo_root / "data" / "raw" / "test", "valid")
        raise RuntimeError("Safety rail failed to catch test in path!")
    except AssertionError as e:
        logger.info("Safety Rail Test 2 Passed: Caught prohibited path -> %s", e)

    logger.info("All safety rail checks verified successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone validation harness for Ariadne bi-encoder checkpoints."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path or identifier of model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_flag",
        type=str,
        default=None,
        help="Path or identifier of model checkpoint to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for query and document encoding.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Do not save result JSON to eval/results/.",
    )
    parser.add_argument(
        "--test-safety-rail",
        action="store_true",
        default=False,
        help="Run self-test asserting safety rails raise on test split.",
    )
    args = parser.parse_args()

    if args.test_safety_rail:
        test_safety_rail_enforcement()
        sys.exit(0)

    target_ckpt = args.checkpoint_flag or args.checkpoint
    run_validation(
        model_name_or_path=target_ckpt,
        save_results=not args.no_save,
        batch_size=args.batch_size,
    )
