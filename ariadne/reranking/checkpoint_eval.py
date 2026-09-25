"""Day 4-5 checkpoint evaluation for dense retrieval and cross-encoder reranking."""

from __future__ import annotations

import json
import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

_repo_root = Path(__file__).resolve().parent.parent
_workspace_root = _repo_root.parent
for path in [str(_workspace_root), str(_repo_root)]:
	if path not in sys.path:
		sys.path.insert(0, path)

from ariadne.eval.metrics import evaluate_ranking
from ariadne.finetuning.embedder import encode
from ariadne.reranking.calibration import calibrate
from ariadne.reranking.cross_encoder import rerank
from ariadne.retrieval.pipeline import HybridPipeline


CONFIG_B_STATUS = (
	"Config B (dense+BM25 fusion): NOT AVAILABLE — "
	"retrieval/pipeline.py not yet implemented"
)


def load_valid_records() -> Tuple[List[str], List[str], List[str], List[str]]:
	"""Loads only the configured valid split using the validation convention."""
	from ariadne.finetuning.validate import load_config, verify_safety_rails

	config = load_config()
	data_config = config.get("data", {})
	raw_dir = _repo_root / data_config.get("raw_dir", "data/raw")
	valid_split_name = data_config.get("valid_split", "valid")
	valid_path = raw_dir / valid_split_name
	verify_safety_rails(valid_path, valid_split_name)

	if not valid_path.exists():
		raise FileNotFoundError(f"Validation dataset split not found at {valid_path}")

	from datasets import load_from_disk

	valid_dataset = load_from_disk(str(valid_path))
	return (
		list(valid_dataset["query_id"]),
		list(valid_dataset["corpus_id"]),
		list(valid_dataset["query"]),
		list(valid_dataset["code"]),
	)


def build_qrels(query_ids: Sequence[str], corpus_ids: Sequence[str]) -> Dict[str, Set[str]]:
	"""Builds query-to-relevant-corpus mappings from validation pairs."""
	qrels: Dict[str, Set[str]] = {}
	for query_id, corpus_id in zip(query_ids, corpus_ids):
		qrels.setdefault(query_id, set()).add(corpus_id)
	return qrels


def unique_corpus(corpus_ids: Sequence[str], codes: Sequence[str]) -> Tuple[List[str], List[str]]:
	"""Keeps the first code occurrence for every corpus ID."""
	cid_to_code: Dict[str, str] = {}
	for corpus_id, code in zip(corpus_ids, codes):
		cid_to_code.setdefault(corpus_id, code)
	unique_ids = list(cid_to_code)
	return unique_ids, [cid_to_code[corpus_id] for corpus_id in unique_ids]


def dense_rankings(
	query_embeddings: np.ndarray,
	corpus_embeddings: np.ndarray,
	corpus_ids: Sequence[str],
	corpus_codes: Sequence[str],
) -> List[List[Dict[str, Any]]]:
	"""Builds dense-ranked candidate dictionaries for every query."""
	similarities = np.matmul(query_embeddings, corpus_embeddings.T)
	rankings: List[List[Dict[str, Any]]] = []
	for query_scores in similarities:
		indices = np.argsort(-query_scores)
		rankings.append(
			[
				{
					"id": str(corpus_ids[index]),
					"text": str(corpus_codes[index]),
					"fusion_score": float(query_scores[index]),
				}
				for index in indices
			]
		)
	return rankings


def _evaluate_ranked_orders(
	ranked_ids: Sequence[Sequence[str]],
	query_ids: Sequence[str],
	qrels: Dict[str, Set[str]],
) -> Dict[str, float]:
	"""Evaluates per-query ID orders through the shared evaluate_ranking function."""
	metric_names = ["ndcg@10", "mrr@10", "recall@1", "recall@5", "recall@10"]
	per_query_metrics: List[Dict[str, float]] = []
	for query_id, ordered_ids in zip(query_ids, ranked_ids):
		if query_id not in qrels or not ordered_ids:
			continue
		candidate_count = len(ordered_ids)
		query_vector = np.arange(candidate_count, 0, -1, dtype=np.float32)[None, :]
		corpus_vectors = np.eye(candidate_count, dtype=np.float32)
		per_query_metrics.append(
			evaluate_ranking(
				query_embeddings=query_vector,
				corpus_embeddings=corpus_vectors,
				query_ids=[query_id],
				corpus_ids=list(ordered_ids),
				qrels=qrels,
				top_k=10,
			)
		)

	if not per_query_metrics:
		return {name: 0.0 for name in metric_names}
	return {
		name: float(np.mean([metrics[name] for metrics in per_query_metrics]))
		for name in metric_names
	}


def _print_calibration_diagnostic(
	query_id: str,
	reranked_candidates: Sequence[Dict[str, Any]],
	calibration_result: Dict[str, Any],
	config_name: str = "Config C",
) -> None:
	"""Prints the exact result returned by calibrate() without recomputing it."""
	raw_scores = [
		float(candidate["rerank_score"])
		for candidate in reranked_candidates
		if "rerank_score" in candidate
	]
	top_candidate = calibration_result.get("top_candidate")
	print(f"\n{config_name} diagnostic query_id={query_id!r}", flush=True)
	print(
		f"  candidate_ids={[str(candidate['id']) for candidate in reranked_candidates]!r}",
		flush=True,
	)
	print(f"  raw_rerank_scores={raw_scores!r}", flush=True)
	print(
		"  calibration_result="
		f"{{'confidence_variance': {calibration_result['confidence_variance']!r}, "
		f"'should_abstain': {calibration_result['should_abstain']!r}, "
		f"'top_candidate_id': "
		f"{None if top_candidate is None else str(top_candidate['id'])!r}}}",
		flush=True,
	)


def _print_fusion_rank_diagnostic(
	query_ids: Sequence[str],
	queries: Sequence[str],
	config_a_ids: Sequence[Sequence[str]],
	config_b_ids: Sequence[Sequence[str]],
	qrels: Dict[str, Set[str]],
	hybrid_pipeline: HybridPipeline,
) -> None:
	"""Prints dense-versus-fused relevant-document ranks and RRF settings."""
	print("\nConfig B fusion diagnostic", flush=True)
	print(
		"  rrf_config="
		f"{{'dense_weight': {hybrid_pipeline.config['dense_weight']!r}, "
		f"'sparse_weight': {hybrid_pipeline.config['sparse_weight']!r}, "
		f"'rrf_k': {hybrid_pipeline.config['rrf_k']!r}}}",
		flush=True,
	)
	demotions: List[int] = []
	shared_query_ids: List[str] = []
	shared_dense_ids: List[Sequence[str]] = []
	shared_fused_ids: List[Sequence[str]] = []
	for query_id, query, dense_ids, fused_ids in zip(
		query_ids, queries, config_a_ids, config_b_ids
	):
		relevant_ids = {str(candidate_id) for candidate_id in qrels.get(str(query_id), set())}
		dense_ranks = [
			dense_ids.index(candidate_id) + 1
			for candidate_id in relevant_ids
			if candidate_id in dense_ids
		]
		fused_ranks = [
			fused_ids.index(candidate_id) + 1
			for candidate_id in relevant_ids
			if candidate_id in fused_ids
		]
		best_dense_rank = min(dense_ranks) if dense_ranks else None
		best_fused_rank = min(fused_ranks) if fused_ranks else None
		sparse_ids = [
			str(candidate_id)
			for candidate_id, _ in hybrid_pipeline.sparse.retrieve(
				query, k=len(hybrid_pipeline.corpus)
			)
		]
		sparse_ranks = [
			sparse_ids.index(candidate_id) + 1
			for candidate_id in relevant_ids
			if candidate_id in sparse_ids
		]
		best_sparse_rank = min(sparse_ranks) if sparse_ranks else None
		if best_dense_rank is not None and best_fused_rank is not None:
			demotions.append(best_fused_rank - best_dense_rank)
		if best_fused_rank is not None:
			shared_query_ids.append(str(query_id))
			shared_dense_ids.append(dense_ids)
			shared_fused_ids.append(fused_ids)
		print(
			f"  query_id={str(query_id)!r}: "
			f"Config A rank={best_dense_rank!r}, "
			f"BM25 rank={best_sparse_rank!r}, Config B rank={best_fused_rank!r}, "
			f"delta={None if best_dense_rank is None or best_fused_rank is None else best_fused_rank - best_dense_rank!r}",
			flush=True,
		)
	print(
		"  average_rank_delta_B_minus_A="
		f"{float(np.mean(demotions)) if demotions else None!r} "
		f"(n={len(demotions)})",
		flush=True,
	)
	shared_dense_metrics = _evaluate_ranked_orders(
		shared_dense_ids, shared_query_ids, qrels
	)
	shared_fused_metrics = _evaluate_ranked_orders(
		shared_fused_ids, shared_query_ids, qrels
	)
	print(
		f"  shared_candidate_subset_metrics (n={len(shared_query_ids)}): "
		f"Config A ndcg@10={shared_dense_metrics['ndcg@10']:.4f}, "
		f"mrr@10={shared_dense_metrics['mrr@10']:.4f}; "
		f"Config B ndcg@10={shared_fused_metrics['ndcg@10']:.4f}, "
		f"mrr@10={shared_fused_metrics['mrr@10']:.4f}",
		flush=True,
	)


def evaluate_checkpoint(
	query_ids: Sequence[str],
	corpus_ids: Sequence[str],
	queries: Sequence[str],
	corpus_codes: Sequence[str],
	qrels: Dict[str, Set[str]],
	config: Dict[str, Any] | None = None,
	verbose: bool = False,
) -> Dict[str, Any]:
	"""Evaluates dense, hybrid, and reranked/calibrated configurations."""
	if config is None:
		from ariadne.finetuning.validate import load_config

		config = load_config()
	eval_batch_size = int(config.get("finetuning", {}).get("eval_batch_size", 32))
	print(f"Encoding {len(queries)} queries...", flush=True)
	query_embeddings = encode(list(queries), batch_size=eval_batch_size)
	print(f"Encoding {len(corpus_codes)} corpus candidates...", flush=True)
	corpus_embeddings = encode(list(corpus_codes), batch_size=eval_batch_size)
	print("Dense embeddings ready; starting Config A and Config C evaluation.", flush=True)
	all_dense_candidates = dense_rankings(
		query_embeddings, corpus_embeddings, corpus_ids, corpus_codes
	)

	config_a_ids = [[candidate["id"] for candidate in candidates] for candidates in all_dense_candidates]
	config_a_metrics = _evaluate_ranked_orders(config_a_ids, query_ids, qrels)

	hybrid_corpus = {
		str(corpus_id): str(code)
		for corpus_id, code in zip(corpus_ids, corpus_codes)
	}
	hybrid_pipeline = HybridPipeline(hybrid_corpus)
	hybrid_rankings = [
		hybrid_pipeline.retrieve(query, k=50)
		for query in queries
	]
	config_b_ids = [
		[str(candidate["id"]) for candidate in candidates]
		for candidates in hybrid_rankings
	]
	config_b_metrics = _evaluate_ranked_orders(config_b_ids, query_ids, qrels)
	_print_fusion_rank_diagnostic(
		query_ids, queries, config_a_ids, config_b_ids, qrels, hybrid_pipeline
	)

	config_c_ids: List[List[str]] = []
	config_c_query_ids: List[str] = []
	abstentions = 0
	pool_hit_query_count = 0
	pool_hit_reranked_top10_count = 0
	pool_hit_original_top10_count = 0
	pool_hit_original_ranks: List[int] = []
	for query_id, query, candidates in tqdm(
		list(zip(query_ids, queries, all_dense_candidates)),
		desc="Evaluating Config A/C",
		unit="query",
	):
		candidate_ids = [str(candidate["id"]) for candidate in candidates[:50]]
		try:
			pre_rerank_pool_ids = [str(candidate["id"]) for candidate in candidates[:20]]
			reranked_candidates = rerank(query, candidates[:50])
			reranked_pool_ids = [str(candidate["id"]) for candidate in reranked_candidates[:20]]
			assert set(pre_rerank_pool_ids) == set(reranked_pool_ids), (
				f"Candidate membership drift for query_id={query_id!r}: "
				f"pre_rerank={pre_rerank_pool_ids!r}, reranked={reranked_pool_ids!r}"
			)
			relevant_ids = {str(candidate_id) for candidate_id in qrels.get(str(query_id), set())}
			pool_relevant_ids = relevant_ids.intersection(pre_rerank_pool_ids)
			if pool_relevant_ids:
				pool_hit_query_count += 1
				original_ranks = [
					pre_rerank_pool_ids.index(candidate_id) + 1
					for candidate_id in pool_relevant_ids
				]
				reranked_ranks = [
					reranked_pool_ids.index(candidate_id) + 1
					for candidate_id in pool_relevant_ids
				]
				best_original_rank = min(original_ranks)
				pool_hit_original_ranks.append(best_original_rank)
				pool_hit_original_top10_count += int(best_original_rank <= 10)
				pool_hit_reranked_top10_count += int(min(reranked_ranks) <= 10)
			calibration_result = calibrate(reranked_candidates)
			if verbose:
				_print_calibration_diagnostic(
					str(query_id), reranked_candidates, calibration_result
				)
			abstentions += int(calibration_result["should_abstain"])
			config_c_ids.append([candidate["id"] for candidate in reranked_candidates])
			config_c_query_ids.append(str(query_id))
		except Exception as error:
			print(
				f"\nERROR evaluating query_id={query_id!r}, "
				f"candidate_ids={candidate_ids!r}: {error!r}",
				file=sys.stderr,
				flush=True,
			)
			traceback.print_exc(file=sys.stderr)
			print("Continuing with the next query.", file=sys.stderr, flush=True)

	config_c_metrics = _evaluate_ranked_orders(config_c_ids, config_c_query_ids, qrels)
	query_count = len(config_c_query_ids)

	config_d_ids: List[List[str]] = []
	config_d_query_ids: List[str] = []
	config_d_abstentions = 0
	for query_id, query, candidates in tqdm(
		list(zip(query_ids, queries, hybrid_rankings)),
		desc="Evaluating Config D",
		unit="query",
	):
		try:
			reranked_candidates = rerank(query, candidates)
			calibration_result = calibrate(reranked_candidates)
			if verbose:
				_print_calibration_diagnostic(
					str(query_id), reranked_candidates, calibration_result, "Config D"
				)
			config_d_abstentions += int(calibration_result["should_abstain"])
			config_d_ids.append([candidate["id"] for candidate in reranked_candidates])
			config_d_query_ids.append(str(query_id))
		except Exception as error:
			print(
				f"\nERROR evaluating Config D query_id={query_id!r}: {error!r}",
				file=sys.stderr,
				flush=True,
			)
			traceback.print_exc(file=sys.stderr)
			print("Continuing with the next query.", file=sys.stderr, flush=True)

	config_d_metrics = _evaluate_ranked_orders(config_d_ids, config_d_query_ids, qrels)
	config_d_query_count = len(config_d_query_ids)
	pool_diagnostic = {
		"queries_with_relevant_in_rerank_pool": pool_hit_query_count,
		"reranked_top10_fraction": (
			pool_hit_reranked_top10_count / pool_hit_query_count
			if pool_hit_query_count
			else 0.0
		),
		"original_top10_fraction": (
			pool_hit_original_top10_count / pool_hit_query_count
			if pool_hit_query_count
			else 0.0
		),
		"original_best_rank_mean": (
			float(np.mean(pool_hit_original_ranks)) if pool_hit_original_ranks else None
		),
	}
	return {
		"config_a": {
			"name": "Config A (dense retrieval only)",
			"metrics": config_a_metrics,
		},
		"config_c": {
			"name": "Config C (dense + cross-encoder reranking + calibration)",
			"metrics": config_c_metrics,
			"calibration": {
				"abstention_count": abstentions,
				"abstention_rate": abstentions / query_count if query_count else 0.0,
			},
			"pool_diagnostic": pool_diagnostic,
		},
		"config_b": {
			"name": "Config B (dense + BM25 fusion)",
			"metrics": config_b_metrics,
		},
		"config_d": {
			"name": "Config D (dense + BM25 fusion + cross-encoder reranking + calibration)",
			"metrics": config_d_metrics,
			"calibration": {
				"abstention_count": config_d_abstentions,
				"abstention_rate": (
					config_d_abstentions / config_d_query_count
					if config_d_query_count
					else 0.0
				),
			},
		},
	}


def main(limit: int | None = None, verbose: bool = False) -> Path:
	"""Runs the valid-split checkpoint evaluation and writes its JSON report."""
	query_ids, corpus_ids, queries, codes = load_valid_records()
	selected_corpus_ids = list(corpus_ids)
	if limit is not None:
		if limit <= 0:
			raise ValueError("--limit must be a positive integer")
		selected_query_ids = set(query_ids[:limit])
		selected_rows = [
			index for index, query_id in enumerate(query_ids) if query_id in selected_query_ids
		]
		query_ids = [query_ids[index] for index in selected_rows]
		selected_corpus_ids = [corpus_ids[index] for index in selected_rows]
		queries = [queries[index] for index in selected_rows]
		print(f"Applying --limit {limit}: evaluating {len(query_ids)} query records.", flush=True)
	qrels = build_qrels(query_ids, selected_corpus_ids)
	unique_ids, unique_codes = unique_corpus(corpus_ids, codes)
	report = evaluate_checkpoint(
		query_ids,
		unique_ids,
		queries,
		unique_codes,
		qrels,
		verbose=verbose,
	)
	timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	results_dir = _repo_root / "eval" / "results"
	results_dir.mkdir(parents=True, exist_ok=True)
	report_payload = {
		"title": "Ariadne Day 4-5 Reranking Checkpoint Evaluation",
		"timestamp": timestamp,
		"evaluation_split": "valid (data/raw/valid)",
		"num_queries": len(set(query_ids)),
		"num_candidates": len(unique_ids),
		"models": report,
	}
	report_path = results_dir / f"checkpoint_eval_{timestamp}.json"
	with open(report_path, "w", encoding="utf-8") as file:
		json.dump(report_payload, file, indent=2)

	config_ndcg = {
		"Config A": report["config_a"]["metrics"]["ndcg@10"],
		"Config B": report["config_b"]["metrics"]["ndcg@10"],
		"Config C": report["config_c"]["metrics"]["ndcg@10"],
		"Config D": report["config_d"]["metrics"]["ndcg@10"],
	}
	winner = max(config_ndcg, key=config_ndcg.get)
	print("=" * 90, flush=True)
	print("ARIADNE DAY 4-5 CHECKPOINT EVALUATION", flush=True)
	print("=" * 90, flush=True)
	print(f"{'Metric':<14} | {'Config A':>12} | {'Config B':>12} | {'Config C':>12} | {'Config D':>12}", flush=True)
	print("-" * 75, flush=True)
	for metric in ["ndcg@10", "mrr@10", "recall@1", "recall@5", "recall@10"]:
		print(
			f"{metric:<14} | {report['config_a']['metrics'][metric]:>12.4f} | "
			f"{report['config_b']['metrics'][metric]:>12.4f} | "
			f"{report['config_c']['metrics'][metric]:>12.4f} | "
			f"{report['config_d']['metrics'][metric]:>12.4f}",
			flush=True,
		)
	print(f"CURRENT WINNER: {winner}", flush=True)
	print(f"Config C abstention rate: {report['config_c']['calibration']['abstention_rate']:.2%}", flush=True)
	print(f"Config D abstention rate: {report['config_d']['calibration']['abstention_rate']:.2%}", flush=True)
	pool_diagnostic = report["config_c"]["pool_diagnostic"]
	print(
		"Relevant-in-top-20 diagnostic: "
		f"{pool_diagnostic['queries_with_relevant_in_rerank_pool']} queries; "
		f"original top-10 fraction={pool_diagnostic['original_top10_fraction']:.2%}; "
		f"reranked top-10 fraction={pool_diagnostic['reranked_top10_fraction']:.2%}; "
		f"original mean best rank={pool_diagnostic['original_best_rank_mean']}",
		flush=True,
	)
	print(f"Checkpoint report saved to: {report_path}", flush=True)
	return report_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run the valid-split reranking checkpoint evaluation.")
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Evaluate only the first N query records; default evaluates the full valid split.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print per-query Config C score and calibration diagnostics.",
	)
	arguments = parser.parse_args()
	main(limit=arguments.limit, verbose=arguments.verbose)
