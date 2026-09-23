from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np

from ariadne.reranking import checkpoint_eval


def test_synthetic_checkpoint_evaluation_reports_two_configs(monkeypatch) -> None:
	query_ids = ["q1", "q2"]
	corpus_ids = ["c1", "c2", "c3"]
	queries = ["find one", "find two"]
	corpus_codes = ["one", "two", "unrelated"]
	qrels: Dict[str, Set[str]] = {"q1": {"c1"}, "q2": {"c2"}}

	def fake_encode(texts, batch_size=None):
		if list(texts) == queries:
			return np.asarray([[1.0, 0.0], [0.0, 1.0]])
		return np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

	def fake_rerank(query: str, candidates):
		if query == "find one":
			return sorted(candidates, key=lambda candidate: candidate["id"] != "c1")
		return sorted(candidates, key=lambda candidate: candidate["id"] != "c2")

	monkeypatch.setattr(checkpoint_eval, "encode", fake_encode)
	monkeypatch.setattr(checkpoint_eval, "rerank", fake_rerank)
	monkeypatch.setattr(
		checkpoint_eval,
		"calibrate",
		lambda candidates: {"should_abstain": False},
	)

	report = checkpoint_eval.evaluate_checkpoint(
		query_ids,
		corpus_ids,
		queries,
		corpus_codes,
		qrels,
		config={"finetuning": {"eval_batch_size": 2}},
	)

	assert report["config_a"]["metrics"]["ndcg@10"] == 1.0
	assert report["config_c"]["metrics"]["ndcg@10"] == 1.0
	assert report["config_c"]["calibration"]["abstention_rate"] == 0.0
	assert "NOT AVAILABLE" in report["config_b"]["status"]