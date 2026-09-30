"""Variable-byte and delta-gap integer compression for sparse index posting lists."""

from __future__ import annotations

from typing import List, Tuple


def encode_vbyte(numbers: List[int]) -> bytes:
	"""Encodes a list of unsigned integers into variable-byte compressed bytes.

	Each byte uses 7 bits for data and the most significant bit (MSB) as continuation flag (1=last byte).
	"""
	byte_list = bytearray()
	for n in numbers:
		if n < 0:
			raise ValueError("Only non-negative integers can be encoded with VByte")
		num = n
		curr_bytes = []
		while True:
			curr_bytes.append(num & 0x7F)
			if num < 128:
				break
			num >>= 7
		curr_bytes[0] |= 0x80  # Terminating byte flag
		byte_list.extend(reversed(curr_bytes))
	return bytes(byte_list)


def decode_vbyte(bytestream: bytes) -> List[int]:
	"""Decodes variable-byte compressed bytes back into a list of unsigned integers."""
	numbers = []
	n = 0
	for byte in bytestream:
		if byte & 0x80:
			n = (n << 7) | (byte & 0x7F)
			numbers.append(n)
			n = 0
		else:
			n = (n << 7) | (byte & 0x7F)
	return numbers


def compress_postings(postings: List[Tuple[int, int]]) -> Tuple[bytes, bytes]:
	"""Compresses posting list (doc_id, tf) pairs using delta-gap encoding for doc_ids and VByte for frequencies.

	Args:
		postings: Sorted list of (doc_id, tf) pairs where doc_ids are strictly ascending.

	Returns:
		Tuple of (compressed_doc_ids_bytes, compressed_tfs_bytes).
	"""
	if not postings:
		return b"", b""

	doc_ids = [doc_id for doc_id, _ in postings]
	tfs = [tf for _, tf in postings]

	# Delta encoding for strictly ascending doc_ids: d_0 = id_0, d_i = id_i - id_{i-1}
	deltas = [doc_ids[0]]
	for i in range(1, len(doc_ids)):
		deltas.append(doc_ids[i] - doc_ids[i - 1])

	comp_ids = encode_vbyte(deltas)
	comp_tfs = encode_vbyte(tfs)
	return comp_ids, comp_tfs


def decompress_postings(comp_ids: bytes, comp_tfs: bytes) -> List[Tuple[int, int]]:
	"""Decompresses delta-encoded and VByte-compressed posting lists."""
	if not comp_ids or not comp_tfs:
		return []

	deltas = decode_vbyte(comp_ids)
	tfs = decode_vbyte(comp_tfs)

	# Reconstruct original doc_ids from deltas
	doc_ids = []
	curr = 0
	for d in deltas:
		curr += d
		doc_ids.append(curr)

	return list(zip(doc_ids, tfs))
