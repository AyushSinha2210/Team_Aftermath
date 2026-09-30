import numpy as np

from ariadne.retrieval.hyde_retriever import CodeHyDERetriever


class DummyModel:
	def encode(self, text):
		return np.array([float(len(text)), 1.0, 2.0])


class DummyBaseRetriever:
	def __init__(self):
		self.model = DummyModel()
		self.last_query = None
		self.last_vector = None

	def retrieve(self, query, top_k=10):
		self.last_query = query
		return [("doc_base", 0.95)]

	def retrieve_by_vector(self, vector, top_k=10):
		self.last_vector = vector
		return [("doc_vector", 0.98)]


def test_code_hyde_retriever_hyde_enabled():
	base = DummyBaseRetriever()
	wrapper = CodeHyDERetriever(base)

	res = wrapper.retrieve("find authentication handler", top_k=5, use_hyde=True)
	assert len(res) == 1
	assert res[0][0] == "doc_vector"
	assert base.last_vector is not None
	assert np.isclose(np.linalg.norm(base.last_vector), 1.0)


def test_code_hyde_retriever_hyde_disabled():
	base = DummyBaseRetriever()
	wrapper = CodeHyDERetriever(base)

	res = wrapper.retrieve("find authentication handler", top_k=5, use_hyde=False)
	assert len(res) == 1
	assert res[0][0] == "doc_base"
	assert base.last_query == "find authentication handler"
