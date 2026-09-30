from ariadne.retrieval.fingerprint import (
	ChunkDeduplicator,
	compute_chunk_hash,
	normalize_code_for_fingerprint,
)


def test_normalize_code_ignores_comments_and_whitespace():
	code1 = "# Comment here\ndef add(a, b):\n    return a + b\n"
	code2 = "// Different comment\ndef add( a ,  b ):\n\treturn a+b"

	assert normalize_code_for_fingerprint(code1) == normalize_code_for_fingerprint(code2)
	assert compute_chunk_hash(code1) == compute_chunk_hash(code2)


def test_chunk_deduplicator():
	dedup = ChunkDeduplicator()
	corpus = {
		"c1": "def foo():\n    return 42",
		"c2": "def foo():\n    # identical code\n    return 42",
		"c3": "def bar():\n    return 99",
	}

	unique_corpus, skipped = dedup.filter_unique_chunks(corpus)

	assert len(unique_corpus) == 2
	assert "c1" in unique_corpus
	assert "c3" in unique_corpus
	assert "c2" in skipped
	assert dedup.duplicates_skipped == 1
