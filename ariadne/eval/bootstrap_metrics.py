"""Bootstrap statistical significance and confidence interval estimator for retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ariadne.eval.metrics import compute_mrr_at_k, compute_ndcg_at_k


@dataclass(frozen=True)
class MetricBootstrapSummary:
	metric_name: str
	point_estimate: float
	std_err: float
	ci_lower: float
	ci_upper: float


def bootstrap_metric_ci(
	per_query_scores: List[float],
	confidence_level: float = 0.95,
	n_bootstrap: int = 1000,
	random_state: Optional[int] = 42,
) -> MetricBootstrapSummary:
	"""Estimates bootstrap confidence interval for arbitrary per-query metric scores.

	Args:
		per_query_scores: List of metric values per query (e.g. NDCG@10 scores).
		confidence_level: Desired coverage probability (e.g. 0.95).
		n_bootstrap: Number of bootstrap resamples.
		random_state: Seed for reproducibility.

	Returns:
		MetricBootstrapSummary with point estimate, standard error, and CI bounds.
	"""
	arr = np.asarray(per_query_scores, dtype=float)
	if arr.size == 0:
		return MetricBootstrapSummary(
			metric_name="metric",
			point_estimate=0.0,
			std_err=0.0,
			ci_lower=0.0,
			ci_upper=0.0,
		)

	point_est = float(np.mean(arr))
	rng = np.random.default_rng(random_state)
	resamples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
	resampled_means = np.mean(resamples, axis=1)

	alpha = 1.0 - confidence_level
	lower_pct = 100.0 * (alpha / 2.0)
	upper_pct = 100.0 * (1.0 - alpha / 2.0)

	ci_lower = float(np.percentile(resampled_means, lower_pct))
	ci_upper = float(np.percentile(resampled_means, upper_pct))
	std_err = float(np.std(resampled_means, ddof=1))

	return MetricBootstrapSummary(
		metric_name="metric",
		point_estimate=round(point_est, 4),
		std_err=round(std_err, 4),
		ci_lower=round(ci_lower, 4),
		ci_upper=round(ci_upper, 4),
	)


def compute_eval_bootstrap(
	ranks: List[int],
	k: int = 10,
	n_bootstrap: int = 1000,
) -> Dict[str, MetricBootstrapSummary]:
	"""Computes NDCG@k and MRR@k with full bootstrap confidence intervals."""
	if not ranks:
		return {}

	# Per-query MRR
	per_query_mrr = [1.0 / r if (1 <= r <= k) else 0.0 for r in ranks]
	# Per-query NDCG
	per_query_ndcg = [1.0 / np.log2(r + 1) if (1 <= r <= k) else 0.0 for r in ranks]

	mrr_summary = bootstrap_metric_ci(per_query_mrr, n_bootstrap=n_bootstrap)
	ndcg_summary = bootstrap_metric_ci(per_query_ndcg, n_bootstrap=n_bootstrap)

	return {
		f"mrr@{k}": MetricBootstrapSummary(
			metric_name=f"mrr@{k}",
			point_estimate=mrr_summary.point_estimate,
			std_err=mrr_summary.std_err,
			ci_lower=mrr_summary.ci_lower,
			ci_upper=mrr_summary.ci_upper,
		),
		f"ndcg@{k}": MetricBootstrapSummary(
			metric_name=f"ndcg@{k}",
			point_estimate=ndcg_summary.point_estimate,
			std_err=ndcg_summary.std_err,
			ci_lower=ndcg_summary.ci_lower,
			ci_upper=ndcg_summary.ci_upper,
		),
	}
