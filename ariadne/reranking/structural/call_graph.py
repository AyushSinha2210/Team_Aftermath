"""Build and query a JavaScript function call graph."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


def _configured_depth() -> int:
	config_path = Path(__file__).resolve().parents[2] / "config.yaml"
	with config_path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)
	return int(config["reranking"]["structural"]["call_graph_depth"])


def _max_depth(max_depth: int | None) -> int:
	return _configured_depth() if max_depth is None else max_depth


def build_call_graph(parsed_functions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
	"""Build an adjacency map containing only calls to known functions."""
	function_names = {function["name"] for function in parsed_functions}
	return {
		function["name"]: [
			call for call in function.get("calls", []) if call in function_names
		]
		for function in parsed_functions
	}


def resolve_call_path(
	call_graph: Dict[str, List[str]],
	source: str,
	target: str,
	max_depth: int | None = None,
) -> List[str] | None:
	"""Return the shortest path from source to target within the depth limit."""
	if source not in call_graph or target not in call_graph:
		return None

	depth_limit = _max_depth(max_depth)
	queue = deque([(source, [source])])
	visited = {source}
	while queue:
		current, path = queue.popleft()
		if current == target:
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
	"""Return functions reachable from source within the depth limit."""
	if source not in call_graph:
		return set()

	depth_limit = _max_depth(max_depth)
	reachable: Set[str] = set()
	queue = deque([(source, 0)])
	visited = {source}
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
