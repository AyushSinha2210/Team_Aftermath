"""Git diff chunk impact analysis for pinpoint incremental indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class ChunkBoundary:
	chunk_id: str
	file_path: str
	start_line: int
	end_line: int


_HUNK_HEADER_RE = re.compile(
	r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_len>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))?\s+@@"
)


def parse_diff_modified_lines(unified_diff: str) -> Dict[str, Set[int]]:
	"""Parses a unified git diff and returns modified line numbers per file.

	Args:
		unified_diff: Standard unified git diff text.

	Returns:
		Dict mapping file_path -> set of modified line numbers.
	"""
	modified_lines: Dict[str, Set[int]] = {}
	current_file: str | None = None
	current_line: int = 0

	for line in unified_diff.splitlines():
		if line.startswith("+++ b/"):
			current_file = line[6:].strip()
			modified_lines.setdefault(current_file, set())
			continue
		elif line.startswith("+++ /dev/null"):
			current_file = None
			continue

		if current_file is None:
			continue

		hunk_match = _HUNK_HEADER_RE.match(line)
		if hunk_match:
			current_line = int(hunk_match.group("new_start"))
			continue

		if line.startswith("+") and not line.startswith("+++"):
			modified_lines[current_file].add(current_line)
			current_line += 1
		elif line.startswith("-") and not line.startswith("---"):
			# Deletion at current position
			modified_lines[current_file].add(current_line)
		elif line.startswith(" "):
			current_line += 1

	return modified_lines


def find_impacted_chunks(
	modified_lines: Dict[str, Set[int]],
	chunks: List[ChunkBoundary],
) -> List[str]:
	"""Determines which chunk IDs are touched by the modified lines in a diff.

	Args:
		modified_lines: Dict mapping file_path -> set of modified line numbers.
		chunks: List of defined ChunkBoundary objects.

	Returns:
		List of impacted chunk_ids that need re-embedding.
	"""
	impacted: Set[str] = set()

	for chunk in chunks:
		if chunk.file_path not in modified_lines:
			continue
		file_mod_lines = modified_lines[chunk.file_path]
		# Check if any modified line intersects [start_line, end_line]
		for line_num in file_mod_lines:
			if chunk.start_line <= line_num <= chunk.end_line:
				impacted.add(chunk.chunk_id)
				break

	return sorted(list(impacted))
