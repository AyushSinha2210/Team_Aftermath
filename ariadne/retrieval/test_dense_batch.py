import numpy as np

from ariadne.retrieval.dense_retriever import DenseRetriever


def test_dense_retrieve_batch():
	def dummy_encoder(texts):
		# Create orthogonal-ish representations
		vectors = []
		for t in texts:
			if "apple" in t:
				vectors.append([1.0, 0.0, 0.0])
			elif "banana" in t:
				vectors.append([0.0, 1.0, 0.0])
			else:
				vectors.append([0.0, 0.0, 1.0])
		return np.array(vectors, dtype=np.float32)

	corpus = {
		"d1": "red apple",
		"d2": "yellow banana",
		"d3": "green citrus",
	}

	retriever = DenseRetriever(dummy_encoder)
	retriever.index(corpus)

	queries = ["apple fruit", "banana smoothie"]
	batch_results = retriever.retrieve_batch(queries, k=2)

	assert len(batch_results) == 2
	# Query 0 top result should be d1
	assert batch_results[0][0][0] == "d1"
	# Query 1 top result should be d2
	assert batch_results[1][0][0] == "d2"


def test_dense_retrieve_batch_empty():
	retriever = DenseRetriever()
	assert retriever.retrieve_batch([]) == []
