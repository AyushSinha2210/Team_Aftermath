import time

from ariadne.eval.latency_benchmark import benchmark_retrieval_latency


def test_benchmark_retrieval_latency_basic():
	def dummy_retrieval(query: str):
		# Small busywork
		sum(i * i for i in range(1000))
		return [("doc1", 0.9)]

	queries = [f"query {i}" for i in range(20)]
	res = benchmark_retrieval_latency(dummy_retrieval, queries, warmup_queries=2)

	assert res.total_queries == 20
	assert res.qps > 0.0
	assert res.mean_ms > 0.0
	assert res.p50_ms <= res.p90_ms <= res.p95_ms <= res.p99_ms


def test_benchmark_retrieval_latency_empty():
	res = benchmark_retrieval_latency(lambda q: [], [])
	assert res.total_queries == 0
	assert res.qps == 0.0
	assert res.mean_ms == 0.0
