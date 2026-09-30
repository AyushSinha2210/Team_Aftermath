"""Linear-time community detection and module clustering for code call graphs."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set


def detect_code_communities(
	call_graph: Dict[str, List[str]],
	max_iter: int = 15,
) -> Dict[str, int]:
	"""Partitions functions into cohesive modules using asynchronous Label Propagation.

	Args:
		call_graph: Dict mapping caller -> list of callees.
		max_iter: Maximum label propagation iterations.

	Returns:
		Dict mapping qualified function name -> community integer ID.
	"""
	all_nodes_set: Set[str] = set(call_graph.keys())
	for callees in call_graph.values():
		all_nodes_set.update(callees)
	nodes = sorted(list(all_nodes_set))

	if not nodes:
		return {}

	# Build undirected neighbor map
	neighbors: Dict[str, Set[str]] = {node: set() for node in nodes}
	for u, callees in call_graph.items():
		for v in callees:
			neighbors[u].add(v)
			neighbors[v].add(u)

	# Initial state: each node is its own community
	labels: Dict[str, int] = {node: i for i, node in enumerate(nodes)}

	for _ in range(max_iter):
		changed = False
		for node in nodes:
			nbrs = neighbors[node]
			if not nbrs:
				continue
			# Find most frequent label among neighbors
			nbr_labels = [labels[nbr] for nbr in nbrs]
			counter = Counter(nbr_labels)
			best_label = counter.most_common(1)[0][0]
			if labels[node] != best_label:
				labels[node] = best_label
				changed = True
		if not changed:
			break

	# Renumber community IDs to 0, 1, 2, ...
	unique_labels = sorted(list(set(labels.values())))
	remap = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_labels)}
	return {node: remap[lbl] for node, lbl in labels.items()}
