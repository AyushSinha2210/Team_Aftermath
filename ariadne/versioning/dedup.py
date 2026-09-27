"""Near-duplicate deduplication across versions via cosine similarity thresholds (owned by Person D).

This module identifies and collapses near-duplicate code snippets across commits and versions
using vectorized cosine similarity computation and connected-component clustering (Union-Find).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ariadne.versioning.content_hash import load_config


def get_dedup_threshold(config_path: Optional[Path] = None) -> float:
    """Retrieves the configured cosine similarity threshold from config.yaml.

    Args:
        config_path: Optional path to config.yaml.

    Returns:
        Cosine similarity threshold float (default 0.95).
    """
    config = load_config(config_path)
    return float(config.get("versioning", {}).get("dedup_cosine_threshold", 0.95))


class _UnionFind:
    """Disjoint-set data structure with path compression."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        path: List[int] = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def find_near_duplicates(
    doc_ids: List[str],
    embeddings: np.ndarray,
    threshold: Optional[float] = None,
) -> List[Set[str]]:
    """Groups document IDs into clusters based on cosine similarity thresholding.

    Every pair of documents with cosine similarity >= threshold are connected.
    Clusters are formed by transitive connected components (Union-Find).
    Documents with no near-duplicates form singleton clusters.

    Args:
        doc_ids: Parallel list of document IDs.
        embeddings: 2D numpy array of shape (N, D) containing L2-normalized embeddings.
        threshold: Cosine similarity threshold in [0.0, 1.0]. If None, read from
            config.yaml (versioning.dedup_cosine_threshold).

    Returns:
        List of sets of doc_ids, where each set represents a connected cluster.
    """
    if len(doc_ids) == 0:
        return []

    if len(doc_ids) != len(embeddings):
        raise ValueError(
            f"Length mismatch: {len(doc_ids)} doc_ids vs {len(embeddings)} embeddings"
        )

    if threshold is None:
        threshold = get_dedup_threshold()

    n = len(doc_ids)
    if n == 1:
        return [{doc_ids[0]}]

    # Ensure 2D float array
    emb = np.asarray(embeddings, dtype=np.float32)

    # Compute pairwise cosine similarity matrix via vectorized GEMM
    sim_matrix = emb @ emb.T

    # Find pairs (i, j) with i < j and sim >= threshold
    uf = _UnionFind(n)
    rows, cols = np.where(np.triu(sim_matrix >= threshold, k=1))
    for r, c in zip(rows, cols):
        uf.union(int(r), int(c))

    # Group document IDs by root representative
    clusters_by_root: Dict[int, Set[str]] = {}
    for idx, doc_id in enumerate(doc_ids):
        root = uf.find(idx)
        if root not in clusters_by_root:
            clusters_by_root[root] = set()
        clusters_by_root[root].add(doc_id)

    # Return clusters ordered deterministically by minimum doc_id in each cluster
    return sorted(list(clusters_by_root.values()), key=lambda c: sorted(list(c))[0])


def collapse_duplicates(
    doc_ids: List[str],
    embeddings: np.ndarray,
    texts: Dict[str, str],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Collapses near-duplicate clusters to canonical representatives.

    For each cluster, a canonical representative is chosen using:
      1. Shortest code text length (len(texts[doc_id])), prioritizing concise implementations.
      2. Alphabetical doc_id order as deterministic tie-breaker.

    The returned dictionary maps canonical doc_id to the list of all versions in its
    cluster (with canonical doc_id placed first), along with cluster summary counts.

    Args:
        doc_ids: Parallel list of document IDs.
        embeddings: 2D numpy array of shape (N, D) containing L2-normalized embeddings.
        texts: Mapping of doc_id to raw code snippet string.
        threshold: Cosine similarity threshold. If None, read from config.yaml.

    Returns:
        Dictionary containing:
            - 'canonical_to_versions': Dict[str, List[str]] mapping canonical doc_id to
              list of all doc_ids in its cluster including itself.
            - 'num_clusters': int total number of clusters.
            - 'num_collapsed': int total number of collapsed documents (total docs - num_clusters).
    """
    if len(doc_ids) == 0:
        return {
            "canonical_to_versions": {},
            "num_clusters": 0,
            "num_collapsed": 0,
        }

    clusters = find_near_duplicates(doc_ids, embeddings, threshold=threshold)

    canonical_to_versions: Dict[str, List[str]] = {}
    for cluster in clusters:
        # Pick canonical representative: shortest text length, then alphabetical doc_id
        canonical = min(
            cluster,
            key=lambda d: (len(texts.get(d, "")), d),
        )
        # Order versions with canonical first, followed by remaining sorted alphabetically
        other_versions = sorted([d for d in cluster if d != canonical])
        canonical_to_versions[canonical] = [canonical] + other_versions

    num_clusters = len(clusters)
    num_collapsed = len(doc_ids) - num_clusters

    return {
        "canonical_to_versions": canonical_to_versions,
        "num_clusters": num_clusters,
        "num_collapsed": num_collapsed,
    }
