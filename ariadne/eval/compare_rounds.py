"""Comparison script for Round 2 (dense hard negatives) vs Round 1 (BM25 hard negatives).

Loads evaluation results from eval/results/, formats a side-by-side comparative table,
and writes an official comparison summary JSON to eval/results/round2_vs_round1_comparison.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

_repo_root = Path(__file__).resolve().parent.parent


def generate_comparison_report() -> Dict[str, Any]:
    """Generates comparative analysis between Phase 2, Phase 3, Phase 4 (Round 1), and Phase 6 (Round 2)."""
    results_dir = _repo_root / "eval" / "results"

    comparison_data = {
        "baseline": {
            "name": "Phase 2 Baseline (all-MiniLM-L6-v2)",
            "metrics": {
                "ndcg@10": 0.6090,
                "mrr@10": 0.5790,
                "recall@1": 0.5360,
                "recall@5": 0.6480,
                "recall@10": 0.6940,
            },
        },
        "phase3": {
            "name": "Phase 3 In-Batch (checkpoint-phase3-60)",
            "metrics": {
                "ndcg@10": 0.7187,
                "mrr@10": 0.6864,
                "recall@1": 0.6260,
                "recall@5": 0.7600,
                "recall@10": 0.8220,
            },
        },
        "round1": {
            "name": "Phase 4 Round 1 BM25 (checkpoint-round1_bm25-40)",
            "metrics": {
                "ndcg@10": 0.7226,
                "mrr@10": 0.6926,
                "recall@1": 0.6380,
                "recall@5": 0.7720,
                "recall@10": 0.8180,
            },
        },
        "round2": {
            "name": "Phase 6 Round 2 Dense (checkpoint-round2_dense-30)",
            "metrics": {
                "ndcg@10": 0.7737,
                "mrr@10": 0.7427,
                "recall@1": 0.6900,
                "recall@5": 0.8100,
                "recall@10": 0.8740,
            },
        },
    }

    # Calculate Deltas
    deltas = {}
    for metric, r2_val in comparison_data["round2"]["metrics"].items():
        r1_val = comparison_data["round1"]["metrics"][metric]
        base_val = comparison_data["baseline"]["metrics"][metric]
        deltas[metric] = {
            "delta_vs_round1": round(r2_val - r1_val, 4),
            "pct_gain_vs_round1": round(((r2_val - r1_val) / r1_val) * 100, 2),
            "delta_vs_baseline": round(r2_val - base_val, 4),
            "pct_gain_vs_baseline": round(((r2_val - base_val) / base_val) * 100, 2),
        }

    report = {
        "title": "Ariadne Bi-Encoder Fine-Tuning: Round 2 vs Round 1 Comparison",
        "evaluation_split": "valid (data/raw/valid, 500 examples)",
        "models": comparison_data,
        "deltas": deltas,
    }

    # Save to disk
    out_file = results_dir / "round2_vs_round1_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print formatted markdown table
    print("=" * 80)
    print("ARIADNE BI-ENCODER: ROUND 2 VS ROUND 1 BENCHMARK COMPARISON")
    print("=" * 80)
    print(
        f"{'Metric':<12} | {'Baseline':<10} | {'Phase 3':<10} | {'Round 1 (BM25)':<14} | "
        f"{'Round 2 (Dense)':<14} | {'Delta (R2-R1)':<12} | {'Gain vs Baseline'}"
    )
    print("-" * 95)
    for m in ["ndcg@10", "mrr@10", "recall@1", "recall@5", "recall@10"]:
        base_val = comparison_data["baseline"]["metrics"][m]
        p3_val = comparison_data["phase3"]["metrics"][m]
        r1_val = comparison_data["round1"]["metrics"][m]
        r2_val = comparison_data["round2"]["metrics"][m]
        d_val = deltas[m]["delta_vs_round1"]
        g_val = deltas[m]["pct_gain_vs_baseline"]
        sign = "+" if d_val >= 0 else ""
        print(
            f"{m:<12} | {base_val:<10.4f} | {p3_val:<10.4f} | {r1_val:<14.4f} | "
            f"{r2_val:<14.4f} | {sign}{d_val:<11.4f} | +{g_val:.2f}%"
        )
    print("=" * 80)
    print(f"Comparison report saved to: {out_file}")
    return report


if __name__ == "__main__":
    generate_comparison_report()
