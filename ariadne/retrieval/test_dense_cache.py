import numpy as np

from ariadne.retrieval.dense_retriever import DenseRetriever


def test_query_cache_hit():
	call_count = [0]

	def counting_encoder(texts):
		call_count[0] += len(texts)
		return np.ones((len(texts), 4), dtype=np.float32)

	corpus = {"d1": "code chunk 1"}
	retriever = DenseRetriever(counting_encoder, query_cache_size=2)
	retriever.index(corpus)

	initial_calls = call_count[0]

	# First retrieval -> encoder called
	retriever.retrieve("my query", k=1)
	assert call_count[0] == initial_calls + 1

	# Second retrieval of identical query -> cache hit, no new encoder call!
	retriever.retrieve("my query", k=1)
	assert call_count[0] == initial_calls + 1


def test_query_cache_eviction():
	def dummy_encoder(texts):
		return np.ones((len(texts), 4), dtype=np.float32)

	corpus = {"d1": "code chunk 1"}
	retriever = DenseRetriever(dummy_encoder, query_cache_size=2)
	retriever.index(corpus)

	retriever.retrieve("q1", k=1)
	retriever.retrieve("q2", k=1)
	assert len(retriever.query_cache) == 2

	retriever.retrieve("q3", k=1)
	assert len(retriever.query_cache) == 2
	assert "q1" not in retriever.query_cache
	assert "q3" in retriever.query_cache
