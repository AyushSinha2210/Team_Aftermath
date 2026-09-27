"""Parse JavaScript functions and their direct call sites with Tree-sitter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language


_FUNCTION_NODE_TYPES = {
	"arrow_function",
	"function_declaration",
	"function_expression",
	"generator_function_declaration",
	"generator_function",
	"method_definition",
}


def _node_text(node: Node, source: bytes) -> str:
	return source[node.start_byte : node.end_byte].decode("utf-8")


def _is_function_node(node: Node) -> bool:
	return node.type in _FUNCTION_NODE_TYPES


def _function_name(node: Node, source: bytes) -> str:
	name_node = node.child_by_field_name("name")
	if name_node is not None:
		return _node_text(name_node, source)

	parent = node.parent
	if parent is not None and parent.type == "variable_declarator":
		variable_node = parent.child_by_field_name("name")
		if variable_node is not None:
			return _node_text(variable_node, source)

	if parent is not None and parent.type == "assignment_expression":
		variable_node = parent.child_by_field_name("left")
		if variable_node is not None:
			return _node_text(variable_node, source)

	return "<anonymous>"


def _call_name(node: Node, source: bytes) -> str | None:
	function_node = node.child_by_field_name("function")
	if function_node is None:
		return None
	if function_node.type == "identifier":
		return _node_text(function_node, source)
	return None


def _direct_calls(body: Node, source: bytes) -> List[str]:
	calls: List[str] = []

	def visit(node: Node) -> None:
		if node is not body and _is_function_node(node):
			return
		if node.type == "call_expression":
			name = _call_name(node, source)
			if name is not None:
				calls.append(name)
		for child in node.children:
			visit(child)

	visit(body)
	return calls


def parse_js_file(file_path: str) -> List[dict[str, Any]]:
	"""Return JavaScript functions and calls made directly in each body."""
	path_obj = Path(file_path)
	file_name = path_obj.name
	source = path_obj.read_bytes()
	parser = Parser()
	parser.set_language(get_language("javascript"))
	tree = parser.parse(source)
	functions: List[dict[str, Any]] = []

	def visit(node: Node) -> None:
		if _is_function_node(node):
			body = node.child_by_field_name("body")
			calls = _direct_calls(body, source) if body is not None else []
			bare_name = _function_name(node, source)
			functions.append(
				{
					"name": f"{file_name}:{bare_name}",
					"short_name": bare_name,
					"file": file_name,
					"start_line": node.start_point[0] + 1,
					"end_line": node.end_point[0] + 1,
					"calls": calls,
				}
			)
		for child in node.children:
			visit(child)

	visit(tree.root_node)
	return functions


def parse_repo(repo_dir: str, pattern: str = "*.js") -> List[dict[str, Any]]:
	"""Walk a directory, parse all matching JavaScript files, and return combined functions."""
	repo_path = Path(repo_dir)
	functions: List[dict[str, Any]] = []
	for file_path in sorted(repo_path.rglob(pattern)):
		functions.extend(parse_js_file(str(file_path)))
	return functions
