import numpy as np

from ariadne.reranking.cross_encoder import predict_batches, rerank


class MockBatchModel:
	def __init__(self):
		self.call_count = 0

	def predict(self, pairs, batch_size=16, show_progress_bar=False):
		self.call_count += 1
		# Return arbitrary scores
		return [float(len(p[0]) + len(p[1])) for p in pairs]


def test_predict_batches_empty():
	res = predict_batches(MockBatchModel(), [])
	assert len(res) == 0


def test_predict_batches_chunking():
	model = MockBatchModel()
	pairs = [("query", f"doc {i}") for i in range(35)]
	scores = predict_batches(model, pairs, batch_size=10)

	assert len(scores) == 35
	assert model.call_count == 1  # Underlying sentence-transformers handles batch_size


def test_predict_batches_callable_fallback():
	def callable_model(pairs):
		return [1.0] * len(pairs)

	pairs = [("query", f"doc {i}") for i in range(25)]
	scores = predict_batches(callable_model, pairs, batch_size=10)
	assert len(scores) == 25
	assert np.all(scores == 1.0)
