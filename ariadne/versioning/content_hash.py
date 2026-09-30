"""Content hashing module to detect changed vs unchanged functions across commits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Set

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
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file) or {}
    return config


def get_hash_algorithm(config_path: Optional[Path] = None) -> str:
    """Retrieves the configured hash algorithm from versioning config."""
    config = load_config(config_path)
    return config.get("versioning", {}).get("hash_algorithm", "sha256")


def _normalize_text(text: str) -> str:
    """Normalizes trivial whitespace differences in code text.

    Converts line endings to \n, strips trailing whitespace from each line,
    and strips leading/trailing whitespace from the entire block.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def hash_content(text: str, algorithm: Optional[str] = None) -> str:
    """Returns a hex digest of the given text using the configured hash algorithm.

    Normalizes trivial whitespace differences (strip leading/trailing whitespace,
    normalize line endings to \n) before hashing so identical code with minor
    formatting differences hashes identically.

    Args:
        text: Source code or snippet text to hash.
        algorithm: Optional hash algorithm name. If None, read from config.yaml.

    Returns:
        Hexadecimal hash string.
    """
    if algorithm is None:
        algorithm = get_hash_algorithm()

    normalized = _normalize_text(text)
    hasher = hashlib.new(algorithm)
    hasher.update(normalized.encode("utf-8"))
    return hasher.hexdigest()


def hash_corpus(corpus: Dict[str, str]) -> Dict[str, str]:
    """Hashes each document in the corpus.

    Args:
        corpus: Mapping of doc_id to document text.

    Returns:
        Mapping of doc_id to content hash.
    """
    return {doc_id: hash_content(text) for doc_id, text in corpus.items()}


def diff_hashes(
    old_hashes: Dict[str, str], new_hashes: Dict[str, str]
) -> Dict[str, Set[str]]:
    """Compares two corpus hash dicts to detect added, removed, changed, and unchanged docs.

    Args:
        old_hashes: Previous corpus hashes mapping doc_id to hash.
        new_hashes: Current corpus hashes mapping doc_id to hash.

    Returns:
        Dictionary with keys:
            - 'added': doc_ids only in new_hashes
            - 'removed': doc_ids only in old_hashes
            - 'changed': doc_ids in both but with different hashes
            - 'unchanged': doc_ids in both with identical hashes
    """
    old_keys = set(old_hashes.keys())
    new_keys = set(new_hashes.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys
    common = old_keys & new_keys

    changed = {doc_id for doc_id in common if old_hashes[doc_id] != new_hashes[doc_id]}
    unchanged = {
        doc_id for doc_id in common if old_hashes[doc_id] == new_hashes[doc_id]
    }

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }
