"""Latency profiling and percentile benchmarking for retrieval pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class LatencyBenchmarkResult:
	total_queries: int
	qps: float
	mean_ms: float
	p50_ms: float
	p90_ms: float
	p95_ms: float
	p99_ms: float


def benchmark_retrieval_latency(
	retrieval_fn: Callable[[str], Any],
	queries: List[str],
	warmup_queries: int = 5,
) -> LatencyBenchmarkResult:
	"""Measures retrieval latency percentiles (P50, P90, P95, P99) and QPS.

	Args:
		retrieval_fn: Callable taking a query string.
		queries: List of test query strings.
		warmup_queries: Number of preliminary queries executed to warm caches/JIT.

	Returns:
		LatencyBenchmarkResult with latency distributions in milliseconds.
	"""
	if not queries:
		return LatencyBenchmarkResult(
			total_queries=0,
			qps=0.0,
			mean_ms=0.0,
			p50_ms=0.0,
			p90_ms=0.0,
			p95_ms=0.0,
			p99_ms=0.0,
		)

	# 1. Warm-up
	warmup_subset = queries[:warmup_queries] if len(queries) >= warmup_queries else queries
	for q in warmup_subset:
		retrieval_fn(q)

	# 2. Timed benchmark
	durations_ms: List[float] = []
	t_start_total = time.perf_counter()

	for q in queries:
		t0 = time.perf_counter()
		retrieval_fn(q)
		t1 = time.perf_counter()
		durations_ms.append((t1 - t0) * 1000.0)

	t_end_total = time.perf_counter()
	total_time_s = max(t_end_total - t_start_total, 1e-6)
	qps = len(queries) / total_time_s

	arr = np.asarray(durations_ms, dtype=float)

	return LatencyBenchmarkResult(
		total_queries=len(queries),
		qps=float(round(qps, 2)),
		mean_ms=float(round(float(np.mean(arr)), 3)),
		p50_ms=float(round(float(np.percentile(arr, 50)), 3)),
		p90_ms=float(round(float(np.percentile(arr, 90)), 3)),
		p95_ms=float(round(float(np.percentile(arr, 95)), 3)),
		p99_ms=float(round(float(np.percentile(arr, 99)), 3)),
	)
