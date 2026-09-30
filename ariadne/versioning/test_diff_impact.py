from ariadne.versioning.diff_impact import (
	ChunkBoundary,
	find_impacted_chunks,
	parse_diff_modified_lines,
)


def test_parse_diff_modified_lines():
	diff = """--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,4 @@
 def login():
+    verify_token()
     return True
"""
	mod = parse_diff_modified_lines(diff)
	assert "src/auth.py" in mod
	assert 11 in mod["src/auth.py"]


def test_find_impacted_chunks():
	modified = {
		"src/auth.py": {11, 12},
		"src/utils.py": {5},
	}

	chunks = [
		ChunkBoundary(chunk_id="auth:login", file_path="src/auth.py", start_line=10, end_line=15),
		ChunkBoundary(chunk_id="auth:logout", file_path="src/auth.py", start_line=20, end_line=30),
		ChunkBoundary(chunk_id="utils:helper", file_path="src/utils.py", start_line=1, end_line=10),
		ChunkBoundary(chunk_id="other:foo", file_path="src/other.py", start_line=1, end_line=20),
	]

	impacted = find_impacted_chunks(modified, chunks)

	# auth:login (lines 10-15) and utils:helper (lines 1-10) should be impacted
	assert "auth:login" in impacted
	assert "utils:helper" in impacted
	assert "auth:logout" not in impacted
	assert "other:foo" not in impacted
