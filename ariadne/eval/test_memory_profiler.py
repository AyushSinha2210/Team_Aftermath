import numpy as np

from ariadne.eval.memory_profiler import profile_index_memory


class DummyDense:
	def __init__(self):
		# 100 docs, 384 dimensions, float32 (4 bytes each) -> 100 * 384 * 4 = 153,600 bytes
		self.embeddings = np.zeros((100, 384), dtype=np.float32)


class DummySparse:
	def __init__(self):
		self.ids = [f"d{i}" for i in range(100)]
		self.postings = {f"term_{i}": [(j, 1) for j in range(5)] for i in range(50)}


def test_profile_index_memory():
	dense = DummyDense()
	sparse = DummySparse()

	report = profile_index_memory(dense, sparse)

	assert report.num_documents == 100
	assert report.embedding_dim == 384
	assert report.dense_embeddings_bytes == 153600
	assert report.sparse_postings_bytes > 0
	assert report.total_mb > 0.0


def test_profile_index_memory_empty():
	class Empty:
		pass

	report = profile_index_memory(Empty(), Empty())
	assert report.num_documents == 0
	assert report.dense_embeddings_bytes == 0
