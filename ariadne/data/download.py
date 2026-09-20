"""Data download module for CoIR apps dataset splits.

Pulls CoIR-Retrieval/apps (queries, corpus, qrels) from HuggingFace,
constructs paired datasets for train, valid, and test, and caches them
locally in ariadne/data/raw/ in an idempotent manner.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prevent local directory from shadowing standard library inspect module
_script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _script_dir:
    sys.path.pop(0)
import inspect as _stdlib_inspect  # noqa: F401

import json
import logging
from typing import Any, Dict, List

import yaml
from datasets import Dataset, load_dataset


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configures structured logger.

    Args:
        log_level: Logging level name ('INFO', 'DEBUG', etc.).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("ariadne.data.download")
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


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads central configuration.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_already_downloaded(raw_dir: Path, splits: List[str]) -> bool:
    """Checks if all required dataset splits already exist on disk.

    Args:
        raw_dir: Directory where raw dataset splits are stored.
        splits: List of split directory names to verify.

    Returns:
        True if all splits exist and are non-empty directories, False otherwise.
    """
    if not raw_dir.exists():
        return False

    for split in splits:
        split_path = raw_dir / split
        if not split_path.exists() or not any(split_path.iterdir()):
            return False

    return True


def build_pairs_dataset(
    qrels: Any,
    queries_dict: Dict[str, str],
    corpus_dict: Dict[str, str],
) -> Dataset:
    """Combines qrel mappings with query and code text into a Dataset.

    Args:
        qrels: HuggingFace Dataset containing 'query-id', 'corpus-id', and 'score'.
        queries_dict: Mapping from query ID to natural language query text.
        corpus_dict: Mapping from corpus ID to code snippet text.

    Returns:
        HuggingFace Dataset containing paired query and code data.
    """
    records: Dict[str, List[Any]] = {
        "query_id": [],
        "corpus_id": [],
        "query": [],
        "code": [],
        "score": [],
    }

    for row in qrels:
        qid = row["query-id"]
        cid = row["corpus-id"]
        score = row.get("score", 1)

        q_text = queries_dict.get(qid, "")
        c_text = corpus_dict.get(cid, "")

        records["query_id"].append(qid)
        records["corpus_id"].append(cid)
        records["query"].append(q_text)
        records["code"].append(c_text)
        records["score"].append(score)

    return Dataset.from_dict(records)


def download_and_prepare(config_file: Path) -> None:
    """Downloads dataset from HuggingFace, constructs splits, and saves to disk.

    Args:
        config_file: Path to config.yaml.
    """
    config = load_config(config_file)
    logger = setup_logger(config.get("global", {}).get("log_level", "INFO"))

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "CoIR-Retrieval/apps")
    raw_dir = config_file.parent / data_cfg.get("raw_dir", "data/raw")
    train_split_name = data_cfg.get("train_split", "train")
    valid_split_name = data_cfg.get("valid_split", "valid")
    test_split_name = data_cfg.get("test_split", "test")
    val_ratio = float(data_cfg.get("validation_ratio", 0.1))
    seed = int(config.get("global", {}).get("seed", 42))

    required_splits = [train_split_name, valid_split_name, test_split_name]

    logger.info("Checking if raw data is already present at: %s", raw_dir)
    if is_already_downloaded(raw_dir, required_splits):
        logger.info(
            "Dataset already present and verified at %s. Skipping download (idempotent).",
            raw_dir,
        )
        return

    logger.info("Starting fresh download for dataset '%s'...", dataset_name)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full queries and corpus
    logger.info("Loading queries collection from HuggingFace...")
    hf_queries = load_dataset(dataset_name, "queries", split="queries")
    logger.info("Total queries loaded: %d", len(hf_queries))
    queries_dict = {item["_id"]: item["text"] for item in hf_queries}

    logger.info("Loading code corpus collection from HuggingFace...")
    hf_corpus = load_dataset(dataset_name, "corpus", split="corpus")
    logger.info("Total code corpus snippets loaded: %d", len(hf_corpus))
    corpus_dict = {item["_id"]: item["text"] for item in hf_corpus}

    # Save corpus and queries collections for downstream modules (Persons B/C/D)
    corpus_dir = raw_dir / "corpus"
    queries_dir = raw_dir / "queries"
    logger.info("Caching raw corpus to %s...", corpus_dir)
    hf_corpus.save_to_disk(str(corpus_dir))
    logger.info("Caching raw queries to %s...", queries_dir)
    hf_queries.save_to_disk(str(queries_dir))

    # 2. Load qrels for train and test
    logger.info("Loading qrels (relevance judgments) for train split...")
    train_qrels = load_dataset(dataset_name, "default", split="train")
    logger.info("Total train qrels: %d", len(train_qrels))

    logger.info("Loading qrels (relevance judgments) for test split...")
    test_qrels = load_dataset(dataset_name, "default", split="test")
    logger.info("Total test qrels: %d", len(test_qrels))

    # 3. Build paired query-code datasets
    logger.info("Constructing paired train dataset...")
    full_train_dataset = build_pairs_dataset(train_qrels, queries_dict, corpus_dict)

    logger.info("Constructing paired test dataset...")
    test_dataset = build_pairs_dataset(test_qrels, queries_dict, corpus_dict)

    # 4. Partition train into train and valid deterministically
    logger.info(
        "Partitioning %d training pairs into train (%.0f%%) and valid (%.0f%%) with seed=%d...",
        len(full_train_dataset),
        (1.0 - val_ratio) * 100,
        val_ratio * 100,
        seed,
    )
    split_dataset = full_train_dataset.train_test_split(
        test_size=val_ratio,
        seed=seed,
        shuffle=True,
    )
    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    logger.info("Partitioned train split count: %d", len(train_dataset))
    logger.info("Partitioned valid split count: %d", len(valid_dataset))
    logger.info("Test split count: %d", len(test_dataset))

    # 5. Save all three splits to disk
    train_out = raw_dir / train_split_name
    valid_out = raw_dir / valid_split_name
    test_out = raw_dir / test_split_name

    logger.info("Saving '%s' split to %s...", train_split_name, train_out)
    train_dataset.save_to_disk(str(train_out))

    logger.info("Saving '%s' split to %s...", valid_split_name, valid_out)
    valid_dataset.save_to_disk(str(valid_out))

    logger.info("Saving '%s' split to %s...", test_split_name, test_out)
    test_dataset.save_to_disk(str(test_out))

    # Write metadata summary
    meta = {
        "dataset_name": dataset_name,
        "seed": seed,
        "validation_ratio": val_ratio,
        "splits": {
            train_split_name: len(train_dataset),
            valid_split_name: len(valid_dataset),
            test_split_name: len(test_dataset),
            "corpus": len(hf_corpus),
            "queries": len(hf_queries),
        },
        "columns": train_dataset.column_names,
    }
    with open(raw_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Data download and preparation completed successfully!")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config_path = repo_root / "config.yaml"
    download_and_prepare(config_path)
