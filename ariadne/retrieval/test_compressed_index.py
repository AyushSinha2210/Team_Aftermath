import pytest

from ariadne.retrieval.compressed_index import (
	compress_postings,
	decode_vbyte,
	decompress_postings,
	encode_vbyte,
)


def test_vbyte_roundtrip():
	numbers = [0, 1, 127, 128, 255, 300, 16384, 1000000]
	encoded = encode_vbyte(numbers)
	decoded = decode_vbyte(encoded)
	assert decoded == numbers


def test_vbyte_negative_raises():
	with pytest.raises(ValueError):
		encode_vbyte([-1])


def test_postings_compression_roundtrip():
	postings = [(2, 1), (5, 3), (12, 1), (105, 7), (1000, 2)]
	comp_ids, comp_tfs = compress_postings(postings)

	assert len(comp_ids) > 0
	assert len(comp_tfs) > 0

	decompressed = decompress_postings(comp_ids, comp_tfs)
	assert decompressed == postings


def test_postings_compression_empty():
	comp_ids, comp_tfs = compress_postings([])
	assert comp_ids == b""
	assert comp_tfs == b""
	assert decompress_postings(b"", b"") == []
