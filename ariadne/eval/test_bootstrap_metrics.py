from ariadne.eval.bootstrap_metrics import (
	bootstrap_metric_ci,
	compute_eval_bootstrap,
)


def test_bootstrap_metric_ci():
	scores = [1.0, 0.5, 0.33, 1.0, 0.0, 0.5, 1.0]
	summary = bootstrap_metric_ci(scores, confidence_level=0.95, n_bootstrap=500)

	assert summary.ci_lower <= summary.point_estimate <= summary.ci_upper
	assert summary.std_err > 0.0


def test_compute_eval_bootstrap():
	ranks = [1, 2, 1, 4, 15, 1, 3]
	res = compute_eval_bootstrap(ranks, k=10, n_bootstrap=500)

	assert "mrr@10" in res
	assert "ndcg@10" in res

	mrr = res["mrr@10"]
	assert 0.0 < mrr.point_estimate <= 1.0
	assert mrr.ci_lower <= mrr.point_estimate <= mrr.ci_upper

	ndcg = res["ndcg@10"]
	assert 0.0 < ndcg.point_estimate <= 1.0
	assert ndcg.ci_lower <= ndcg.point_estimate <= ndcg.ci_upper


def test_compute_eval_bootstrap_empty():
	assert compute_eval_bootstrap([]) == {}
