import pytest

from ariadne.retrieval.fusion import linear_decay_fusion


def test_linear_decay_fusion_basic():
	r1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
	r2 = [("doc2", 10.0), ("doc1", 5.0), ("doc4", 1.0)]

	fused = linear_decay_fusion([r1, r2], decay_rate=0.1)
	doc_ids = [doc_id for doc_id, _ in fused]

	# Both doc1 and doc2 are in top ranks of both lists
	assert "doc1" in doc_ids[:2]
	assert "doc2" in doc_ids[:2]


def test_linear_decay_fusion_weights():
	r1 = [("doc1", 1.0)]
	r2 = [("doc2", 1.0)]

	fused = linear_decay_fusion([r1, r2], weights=[2.0, 0.5])
	assert fused[0][0] == "doc1"
	assert fused[0][1] == 2.0
	assert fused[1][0] == "doc2"
	assert fused[1][1] == 0.5


def test_linear_decay_fusion_invalid_params():
	with pytest.raises(ValueError):
		linear_decay_fusion([[("a", 1.0)]], decay_rate=-0.1)
