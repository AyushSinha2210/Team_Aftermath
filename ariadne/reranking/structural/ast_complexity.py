"""AST structural complexity analysis for code candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeComplexityMetrics:
	cyclomatic_complexity: int
	max_indent_depth: int
	line_count: int
	branch_count: int


_BRANCH_KEYWORDS = re.compile(
	r"\b(if|elif|else\s+if|for|while|case|catch|except|&&|\|\||\?)\b"
)


def compute_code_complexity(code_snippet: str) -> CodeComplexityMetrics:
	"""Computes fast structural complexity metrics for a code snippet.

	Args:
		code_snippet: Raw code string.

	Returns:
		CodeComplexityMetrics dataclass.
	"""
	if not code_snippet or not code_snippet.strip():
		return CodeComplexityMetrics(
			cyclomatic_complexity=1,
			max_indent_depth=0,
			line_count=0,
			branch_count=0,
		)

	lines = code_snippet.splitlines()
	non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith(("#", "//"))]

	# 1. Branch count and McCabe cyclomatic complexity: CC = 1 + branches
	branches = 0
	for line in non_empty_lines:
		matches = _BRANCH_KEYWORDS.findall(line)
		branches += len(matches)

	# 2. Maximum indentation / nesting depth
	max_depth = 0
	for line in non_empty_lines:
		stripped = line.lstrip()
		indent = len(line) - len(stripped)
		# Tabs or 4-spaces -> depth levels
		depth = line[:indent].count("\t") + (line[:indent].count(" ") // 4)
		if depth > max_depth:
			max_depth = depth

	return CodeComplexityMetrics(
		cyclomatic_complexity=1 + branches,
		max_indent_depth=max_depth,
		line_count=len(non_empty_lines),
		branch_count=branches,
	)
