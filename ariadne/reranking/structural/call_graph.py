"""Build and query a JavaScript function call graph."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


class AmbiguousFunctionNameError(ValueError):
	"""Raised when a bare function name matches multiple qualified functions across files."""
	pass


class CallGraph(dict):
	"""Adjacency map representation of a function call graph.

	Attributes:
		ambiguous_calls: Dict mapping caller or call name to list of candidate qualified targets.
		short_to_qualified: Dict mapping bare function name to list of matching qualified names.
	"""

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.ambiguous_calls: Dict[str, List[str]] = {}
		self.short_to_qualified: Dict[str, List[str]] = {}


def _configured_depth() -> int:
	config_path = Path(__file__).resolve().parents[2] / "config.yaml"
	with config_path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	return int(config["reranking"]["structural"]["call_graph_depth"])


def _max_depth(max_depth: int | None) -> int:
	return _configured_depth() if max_depth is None else max_depth


def build_call_graph(parsed_functions: List[Dict[str, Any]]) -> CallGraph:
	"""Build an adjacency map keyed by qualified name (file:function).

	Bare call names are resolved to qualified targets if unambiguous across the
	repository. If a bare call matches functions in multiple files, all candidate
	qualified targets are included and the ambiguity is flagged in ambiguous_calls.
	"""
	short_to_qualified: Dict[str, List[str]] = {}
	for function in parsed_functions:
		short_name = function.get("short_name", function["name"].split(":")[-1])
		qual_name = function["name"]
		short_to_qualified.setdefault(short_name, []).append(qual_name)

	all_qualified_names = {function["name"] for function in parsed_functions}

	graph = CallGraph()
	graph.short_to_qualified = short_to_qualified

	for function in parsed_functions:
		qual_name = function["name"]
		resolved_calls: List[str] = []
		for call in function.get("calls", []):
			if call in short_to_qualified:
				candidates = short_to_qualified[call]
				if len(candidates) == 1:
					resolved_calls.append(candidates[0])
				else:
					# Ambiguous call: matches functions in multiple files
					resolved_calls.extend(candidates)
					graph.ambiguous_calls[f"{qual_name}->{call}"] = candidates
					graph.ambiguous_calls[call] = candidates
			elif call in all_qualified_names:
				resolved_calls.append(call)

		graph[qual_name] = resolved_calls

	return graph


def _resolve_function_name(call_graph: Dict[str, Any], name: str) -> str | None:
	"""Resolves a bare or qualified name to a unique node in the call graph.

	Raises AmbiguousFunctionNameError if a bare name matches multiple qualified nodes.
	"""
	# Exact match in graph keys
	if name in call_graph:
		return name

	# Use short_to_qualified index if present on CallGraph
	if hasattr(call_graph, "short_to_qualified") and name in call_graph.short_to_qualified:
		candidates = call_graph.short_to_qualified[name]
		if len(candidates) == 1:
			return candidates[0]
		if len(candidates) > 1:
			raise AmbiguousFunctionNameError(
				f"Bare function name '{name}' is ambiguous across files: matches multiple functions {candidates}. "
				"Specify a qualified 'file:function' name instead."
			)

	# Fallback: inspect keys ending with :name
	matching = [
		k for k in call_graph.keys()
		if isinstance(k, str) and (k == name or k.endswith(f":{name}"))
	]
	if len(matching) == 1:
		return matching[0]
	if len(matching) > 1:
		raise AmbiguousFunctionNameError(
			f"Bare function name '{name}' is ambiguous across files: matches multiple functions {matching}. "
			"Specify a qualified 'file:function' name instead."
		)

	return None


def resolve_call_path(
	call_graph: Dict[str, List[str]],
	source: str,
	target: str,
	max_depth: int | None = None,
) -> List[str] | None:
	"""Return the shortest path from source to target within the depth limit.

	Accepts bare names (if unambiguous) or qualified 'file:function' names.
	Raises AmbiguousFunctionNameError if source or target is ambiguous.
	"""
	resolved_source = _resolve_function_name(call_graph, source)
	resolved_target = _resolve_function_name(call_graph, target)

	if resolved_source is None or resolved_target is None:
		return None

	depth_limit = _max_depth(max_depth)
	queue = deque([(resolved_source, [resolved_source])])
	visited = {resolved_source}
	while queue:
		current, path = queue.popleft()
		if current == resolved_target:
			return path
		if len(path) - 1 >= depth_limit:
			continue
		for called_function in call_graph.get(current, []):
			if called_function not in visited:
				visited.add(called_function)
				queue.append((called_function, path + [called_function]))
	return None


def calls_within_depth(
	call_graph: Dict[str, List[str]],
	source: str,
	max_depth: int | None = None,
) -> Set[str]:
	"""Return functions reachable from source within the depth limit.

	Accepts bare names (if unambiguous) or qualified 'file:function' names.
	Raises AmbiguousFunctionNameError if source is ambiguous.
	"""
	resolved_source = _resolve_function_name(call_graph, source)
	if resolved_source is None:
		return set()

	depth_limit = _max_depth(max_depth)
	reachable: Set[str] = set()
	queue = deque([(resolved_source, 0)])
	visited = {resolved_source}
	while queue:
		current, depth = queue.popleft()
		if depth >= depth_limit:
			continue
		for called_function in call_graph.get(current, []):
			if called_function not in visited:
				visited.add(called_function)
				reachable.add(called_function)
				queue.append((called_function, depth + 1))
	return reachable


def resolve_call_path_bidirectional(
	call_graph: Dict[str, List[str]],
	source: str,
	target: str,
	max_depth: int | None = None,
) -> List[str] | None:
	"""Finds shortest path using bidirectional BFS with O(2 * b^(d/2)) complexity.

	Traverses forward from source and backward from target across inverted edges.
	"""
	resolved_source = _resolve_function_name(call_graph, source)
	resolved_target = _resolve_function_name(call_graph, target)

	if resolved_source is None or resolved_target is None:
		return None
	if resolved_source == resolved_target:
		return [resolved_source]

	depth_limit = _max_depth(max_depth)

	# Build inverted adjacency map for reverse traversal
	inverted_graph: Dict[str, List[str]] = {}
	for u, neighbors in call_graph.items():
		for v in neighbors:
			inverted_graph.setdefault(v, []).append(u)

	forward_parents: Dict[str, str | None] = {resolved_source: None}
	backward_parents: Dict[str, str | None] = {resolved_target: None}

	forward_queue = deque([resolved_source])
	backward_queue = deque([resolved_target])

	intersection: str | None = None

	while forward_queue and backward_queue:
		# Forward expansion step
		if forward_queue:
			curr_f = forward_queue.popleft()
			curr_f_depth = 0
			temp = curr_f
			while forward_parents[temp] is not None:
				temp = forward_parents[temp]  # type: ignore
				curr_f_depth += 1

			if curr_f_depth < depth_limit:
				for nxt in call_graph.get(curr_f, []):
					if nxt not in forward_parents:
						forward_parents[nxt] = curr_f
						forward_queue.append(nxt)
						if nxt in backward_parents:
							intersection = nxt
							break
			if intersection:
				break

		# Backward expansion step
		if backward_queue:
			curr_b = backward_queue.popleft()
			curr_b_depth = 0
			temp = curr_b
			while backward_parents[temp] is not None:
				temp = backward_parents[temp]  # type: ignore
				curr_b_depth += 1

			if curr_b_depth < depth_limit:
				for prev in inverted_graph.get(curr_b, []):
					if prev not in backward_parents:
						backward_parents[prev] = curr_b
						backward_queue.append(prev)
						if prev in forward_parents:
							intersection = prev
							break
			if intersection:
				break

	if intersection is None:
		return None

	# Reconstruct full path: source -> ... -> intersection -> ... -> target
	forward_path = []
	curr = intersection
	while curr is not None:
		forward_path.append(curr)
		curr = forward_parents[curr]
	forward_path.reverse()

	backward_path = []
	curr = backward_parents[intersection]
	while curr is not None:
		backward_path.append(curr)
		curr = backward_parents[curr]

	full_path = forward_path + backward_path
	if len(full_path) - 1 > depth_limit:
		return None
	return full_path


def compute_call_graph_pagerank(
	call_graph: Dict[str, List[str]],
	damping: float = 0.85,
	max_iter: int = 50,
	tol: float = 1e-5,
) -> Dict[str, float]:
	"""Computes PageRank centrality scores for all nodes in the call graph.

	Functions with high PageRank represent architectural nexus points called by many
	subsystems (e.g. database gateways, authentication, utility parsers).

	Args:
		call_graph: Dict mapping caller -> list of callees.
		damping: Probability of following an edge (default 0.85).
		max_iter: Maximum power iterations.
		tol: L1 convergence tolerance.

	Returns:
		Dict mapping qualified node name -> centrality probability in [0, 1].
	"""
	nodes = list(call_graph.keys())
	all_nodes_set = set(nodes)
	for callees in call_graph.values():
		all_nodes_set.update(callees)
	all_nodes = sorted(list(all_nodes_set))
	n = len(all_nodes)
	if n == 0:
		return {}

	# Incoming edge map: v -> list of u where u calls v
	incoming: Dict[str, List[str]] = {node: [] for node in all_nodes}
	out_degrees: Dict[str, int] = {node: len(call_graph.get(node, [])) for node in all_nodes}

	for u, callees in call_graph.items():
		for v in callees:
			incoming.setdefault(v, []).append(u)

	scores = {node: 1.0 / n for node in all_nodes}

	for _ in range(max_iter):
		next_scores = {}
		# Dangling sum from nodes with 0 outgoing calls
		dangling_sum = sum(scores[node] for node in all_nodes if out_degrees[node] == 0)
		base_score = (1.0 - damping) / n + (damping * dangling_sum) / n

		for node in all_nodes:
			inflow = sum(scores[parent] / out_degrees[parent] for parent in incoming[node] if out_degrees[parent] > 0)
			next_scores[node] = base_score + damping * inflow

		# Check convergence
		diff = sum(abs(next_scores[node] - scores[node]) for node in all_nodes)
		scores = next_scores
		if diff < tol:
			break

	return scores


