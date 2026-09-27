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
