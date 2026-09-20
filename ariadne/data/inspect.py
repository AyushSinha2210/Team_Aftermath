"""Data inspection script for CoIR apps dataset splits.

Sanity-checks the schema, logs split sizes, and formats sample records
from train, validation, and test splits pulled from config.yaml.
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
import os
from typing import Any, Dict, List

import yaml
from datasets import Dataset, DatasetDict, load_from_disk


def setup_structured_logging(log_level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger for data inspection.

    Args:
        log_level: Logging level string ('INFO', 'DEBUG', etc.).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("ariadne.data.inspect")
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
    """Loads and validates central config.yaml.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Parsed configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def inspect_split(
    split_name: str,
    split_path: Path,
    num_samples: int,
    logger: logging.Logger,
) -> None:
    """Inspects a single dataset split, logging size and sample records.

    Args:
        split_name: Name of the split (e.g., 'train', 'valid', 'test').
        split_path: Filesystem path to the saved Arrow dataset on disk.
        num_samples: Number of sample records to display.
        logger: Structured logger instance.
    """
    logger.info("=" * 70)
    logger.info("Inspecting split: %s", split_name)
    logger.info("Path: %s", split_path)

    if not split_path.exists():
        logger.warning(
            "Split path does not exist on disk: %s. Run download.sh first.",
            split_path,
        )
        return

    try:
        dataset: Dataset = load_from_disk(str(split_path))
    except Exception as exc:
        logger.error("Failed to load split '%s' from disk: %s", split_name, exc)
        return

    split_size = len(dataset)
    logger.info("Split '%s' record count: %d", split_name, split_size)
    logger.info("Column names: %s", dataset.column_names)

    # Display requested number of samples
    samples_to_show = min(num_samples, split_size)
    for idx in range(samples_to_show):
        record = dataset[idx]
        logger.info("-" * 40)
        logger.info("Sample #%d from '%s':", idx + 1, split_name)
        logger.info("  Query ID : %s", record.get("query_id", "N/A"))
        logger.info("  Corpus ID: %s", record.get("corpus_id", "N/A"))
        logger.info("  Score    : %s", record.get("score", "N/A"))

        query_text = str(record.get("query", ""))
        code_text = str(record.get("code", ""))

        if len(query_text) > 250:
            query_display = query_text[:250] + f"... [truncated, total {len(query_text)} chars]"
        else:
            query_display = query_text

        if len(code_text) > 250:
            code_display = code_text[:250] + f"... [truncated, total {len(code_text)} chars]"
        else:
            code_display = code_text

        logger.info("  Query Text:\n    %s", query_display.replace("\n", "\n    "))
        logger.info("  Code Snippet:\n    %s", code_display.replace("\n", "\n    "))


def main() -> None:
    """Main entrypoint: parses configuration and inspects all splits."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config_file = repo_root / "config.yaml"

    config = load_config(config_file)
    logger = setup_structured_logging(config.get("global", {}).get("log_level", "INFO"))

    raw_dir = repo_root / config.get("data", {}).get("raw_dir", "data/raw")
    splits: List[str] = [
        config.get("data", {}).get("train_split", "train"),
        config.get("data", {}).get("valid_split", "valid"),
        config.get("data", {}).get("test_split", "test"),
    ]

    logger.info("Starting dataset inspection across splits: %s", splits)
    logger.info("Base raw directory: %s", raw_dir)

    for split in splits:
        split_dir = raw_dir / split
        inspect_split(split_name=split, split_path=split_dir, num_samples=2, logger=logger)

    logger.info("=" * 70)
    logger.info("Dataset inspection completed successfully.")


if __name__ == "__main__":
    main()
