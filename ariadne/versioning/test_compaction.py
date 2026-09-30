import numpy as np

from ariadne.versioning.compaction import TombstoneCompactor


def test_tombstone_compaction():
	compactor = TombstoneCompactor()
	entries = {
		"d1": ("hash1", np.zeros(4)),
		"d2": ("hash2", np.ones(4)),
		"d3": ("hash3", np.full(4, 2.0)),
	}

	compactor.mark_deleted("d2")
	assert compactor.is_deleted("d2")
	assert not compactor.is_deleted("d1")

	compacted, purged = compactor.compact(entries)

	assert purged == 1
	assert "d2" not in compacted
	assert "d1" in compacted and "d3" in compacted
	assert len(compactor.tombstones) == 0


def test_tombstone_compactor_empty():
	compactor = TombstoneCompactor()
	compacted, purged = compactor.compact({})
	assert len(compacted) == 0
	assert purged == 0
